from time import perf_counter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from matplotlib import pyplot as plt
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16, attn_implementation="sdpa").to("cuda:0")


class TorchTracemalloc():
    track_memory_consumption = []
    def __enter__(self):
        self.begin = torch.cuda.memory_allocated()
        torch.cuda.reset_max_memory_allocated() # reset the peak gauge to zero
        return self

    def __exit__(self, *exc):
        peak = torch.cuda.max_memory_allocated()
        peaked = (peak - self.begin) // 1024 ** 2
        TorchTracemalloc.track_memory_consumption.append(peaked)
        print(f"peak: {peaked}; reserved: {torch.cuda.max_memory_reserved() // 1024 ** 2}")


def collate_fn(example):
        prompt=f"Question: {example['input']}\nContext: {example['context']}\nAnswer:"
        example['prompt'] = prompt
        return example


def update_model_kwargs(model_kwargs, model_output):
        attention_mask = model_kwargs["attention_mask"]
        model_kwargs["attention_mask"] = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
        model_kwargs["past_key_values"] = model_output.past_key_values
        model_kwargs["use_cache"] = True
        return model_kwargs


dataset = load_dataset('THUDM/LongBench', "samsum", split='test')
dataset = dataset.map(collate_fn, batched=False)


@torch.no_grad()
def prefill(inputs):
    outputs = model(**inputs, use_cache=True)
    next_token_logits = outputs.logits[:, -1, :]
    next_tokens = torch.argmax(next_token_logits, dim=-1)
    next_input_ids = torch.cat([inputs["input_ids"], next_tokens[:, None]], dim=-1)
    next_model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            inputs,
            is_encoder_decoder=False,
        )
    return next_input_ids, next_model_kwargs


def save_bar_chart(title, x, y, ylabel, xlabel, save_path):
    width = 0.4
    xs = np.arange(len(x))
    plt.bar(xs, height=y, width=width)
    plt.title(title)
    plt.xticks(xs, x)
    plt.xlabel(ylabel)
    plt.ylabel(xlabel)
    plt.savefig(save_path)


def eval_generated_lengths(cache):
    batch_size = 1
    num_batches = 2 # NOTE: 200 samples total only in dataset

    generate_kwargs = {"do_sample": False, "temperature": 1.0, "top_p": 1.0}
    if cache == "quantized":
        generate_kwargs["cache_implementation"] = "quantized"

    # warm up
    for _ in range(3):
        inputs_warmup = tokenizer(["Today a dragon flew over Paris"] * 2, return_tensors="pt").to(model.device)
        model.generate(**inputs_warmup, max_new_tokens=20, **generate_kwargs)

    memory_avg, tokens_per_sec_avg = [], []
    time_to_first_token_avg = []
    TTFT, TIME_PER_DECODING = [], []
    #xs = [500, 1000, 4000, 10_000]
    bs = [1, 20, 50, 100, 300]
    gen_length = 500
    for batch_size in bs:
        with TorchTracemalloc() as tt:
            for batch in range(num_batches):
                start = perf_counter()
                torch.cuda.synchronize()
                curr_chunk = dataset[batch: batch+batch_size]
                inputs = tokenizer(curr_chunk['prompt'], padding=True, max_length=100, truncation=True, return_tensors="pt").to(model.device)
                next_input_ids, next_model_kwargs = prefill(inputs)
                TTFT.append(perf_counter() - start)
                next_model_kwargs.pop("input_ids")

                torch.cuda.synchronize()
                out, _ = model.generate(next_input_ids, **next_model_kwargs, min_new_tokens=gen_length-1, max_new_tokens=gen_length, **generate_kwargs)
                TIME_PER_DECODING.append((perf_counter() - start - TTFT[-1]) / batch_size / gen_length)

                del out
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
        memory_avg.append(TorchTracemalloc.track_memory_consumption[-1])
        tokens_per_sec_avg.append(1 / (sum(TIME_PER_DECODING) / len(TIME_PER_DECODING)))
        time_to_first_token_avg.append(sum(TTFT) / len(TTFT))

    save_bar_chart(
         title="Memory consumption for different batch sizes",
         x=bs,
         y=memory_avg,
         ylabel="Batch size",
         xlabel="GPU Memory comsumption in MiB",
         save_path="memory_int4.png",
        )
    print(f"tokens_per_sec_avg - one per condition: {tokens_per_sec_avg}")
    print(f"time_to_first_token_avg - one per condition: {tokens_per_sec_avg}")


eval_generated_lengths(cache="quantized")
