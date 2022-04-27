import gzip
import hashlib
import multiprocessing
import os
import shutil
import time

import numpy as np
from datasets import load_dataset

from arguments import PreprocessingArguments
from transformers import AutoTokenizer, HfArgumentParser


def get_hash(example):
    """Get hash of content field."""
    return {"hash": hashlib.md5(example["content"].strip().encode("utf-8")).hexdigest()}


def line_stats(example):
    """Calculates mean and max line length of file."""
    line_lengths = [len(line) for line in example["content"].splitlines()]
    return {"line_mean": np.mean(line_lengths), "line_max": max(line_lengths)}


def alpha_stats(example):
    """Calculates mean and max line length of file."""
    alpha_frac = np.mean([c.isalnum() for c in example["content"]])
    return {"alpha_frac": alpha_frac}


def check_uniques(example, uniques):
    """Check if current hash is still in set of unique hashes and remove if true."""
    if example["hash"] in uniques:
        uniques.remove(example["hash"])
        return True
    else:
        return False


def is_autogenerated(example, scan_width=5):
    """Check if file is autogenerated by looking for keywords in the first few lines of the file."""
    keywords = ["auto-generated", "autogenerated", "automatically generated"]
    lines = example["content"].splitlines()
    for _, line in zip(range(scan_width), lines):
        for keyword in keywords:
            if keyword in line.lower():
                return {"autogenerated": True}
    else:
        return {"autogenerated": False}


def is_config_test(example, scan_width=5, coeff=5):
    """Check if file is a configuration file or a unit test by :
    1- looking for keywords in the first few lines of the file.
    2- counting number of occurence of the words 'config' and 'test' with respect to number of lines.
    """

    keywords = ["unit tests", "test file", "configuration file"]
    lines = example["content"].splitlines()
    count_config = 0
    count_test = 0
    # first test
    for _, line in zip(range(scan_width), lines):
        for keyword in keywords:
            if keyword in line.lower():
                return {"config_test": True}
    # second test
    nlines = example["content"].count("\n")
    threshold = coeff if nlines < 300 else coeff * (nlines // 300)
    for line in lines:
        count_config += line.lower().count("config")
        count_test += line.lower().count("test")
        if count_config > threshold or count_test > threshold:
            return {"config_test": True}
    return {"config_test": False}


def has_no_keywords(example):
    """Check if a python file has none of the keywords for: funcion, class, for loop, while loop."""
    keywords = ["def ", "class ", "for ", "while "]
    lines = example["content"].splitlines()
    for line in lines:
        for keyword in keywords:
            if keyword in line.lower():
                return {"has_no_keywords": False}
    return {"has_no_keywords": True}


def few_assignments(example, minimum=4):
    """Check if file uses symbol '=' less than `minimum` times."""
    lines = example["content"].splitlines()
    counter = 0
    for line in lines:
        counter += line.lower().count("=")
        if counter > minimum:
            return {"few_assignments": False}
    return {"few_assignments": True}


def char_token_ratio(example):
    """Compute character/token ratio of the file with tokenizer."""
    input_ids = tokenizer(example["content"], truncation=False)["input_ids"]
    ratio = len(example["content"]) / len(input_ids)
    return {"ratio": ratio}


def preprocess(example):
    """Chain all preprocessing steps into one function to not fill cache."""
    results = dict()
    results.update(get_hash(example))
    results.update(line_stats(example))
    results.update(alpha_stats(example))
    results.update(char_token_ratio(example))
    results.update(is_autogenerated(example))
    results.update(is_config_test(example))
    results.update(has_no_keywords(example))
    results.update(few_assignments(example))
    return results


def filter(example, uniques, args):
    """Filter dataset with heuristics. Config, test and has_no_keywords files are removed with a given probability."""
    if not check_uniques(example, uniques):
        return False
    elif example["autogenerated"]:
        return False
    elif example["line_max"] > args.line_max:
        return False
    elif example["line_mean"] > args.line_mean:
        return False
    elif example["alpha_frac"] < args.alpha_frac:
        return False
    elif example["ratio"] < args.min_token_ratio:
        return False
    elif example["config_test"] and np.random.rand() <= args.filter_proba:
        return False
    elif example["has_no_keywords"] and np.random.rand() <= args.filter_proba:
        return False
    elif example["few_assignments"]:
        return False
    else:
        return True


def compress_file(file_path):
    """Compress a file with g-zip."""
    with open(file_path, "rb") as f_in:
        with gzip.open(file_path + ".gz", "wb", compresslevel=6) as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.unlink(file_path)


# Settings
parser = HfArgumentParser(PreprocessingArguments)
args = parser.parse_args()
if args.num_workers is None:
    args.num_workers = multiprocessing.cpu_count()
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

# Load dataset
t_start = time.time()
ds = load_dataset(args.dataset_name, split="train")
print(f"Time to load dataset: {time.time()-t_start:.2f}")

# Run preprocessing
t_start = time.time()
ds = ds.map(preprocess, num_proc=args.num_workers)
print(f"Time to preprocess dataset: {time.time()-t_start:.2f}")

# Deduplicate hashes
uniques = set(ds.unique("hash"))
frac = len(uniques) / len(ds)
print(f"Fraction of duplicates: {1-frac:.2%}")

# Deduplicate data and apply heuristics
t_start = time.time()
ds_filter = ds.filter(filter, fn_kwargs={"uniques": uniques, "args": args})
print(f"Time to filter dataset: {time.time()-t_start:.2f}")
print(f"Size of filtered dataset: {len(ds_filter)}")

# Save data in batches of samples_per_file
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
t_start = time.time()
for file_number, index in enumerate(range(0, len(ds_filter), args.samples_per_file)):
    file_path = f"{args.output_dir}/file-{file_number+1:012}.json"
    end_index = min(len(ds_filter), index + args.samples_per_file)
    ds_filter.select(list(range(index, end_index))).to_json(file_path)
    compress_file(file_path)
print(f"Time to save dataset: {time.time()-t_start:.2f}")
