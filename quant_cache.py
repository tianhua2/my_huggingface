from time import perf_counter
import torch
from torch.profiler import profile, ProfilerActivity
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import gc
import random


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16, attn_implementation="eager").to("cuda:0")

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
        print(f"peaked: {peaked}")


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
batch_size = 1
num_batches = 5 # NOTE: 200 samples total only in dataset

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
TTFT, TIME_PER_DECODING = [], []
TOTAL_TIME_PER_TOKEN = []

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
    for batch in range(num_batches):
        # warm up
        inputs = tokenizer(["Today a dragon flew over Paris"] * 5, return_tensors="pt").to(model.device)
        out, ttft = model.generate(**inputs, do_sample=False, max_new_tokens=50)

        # benchmark
        start = perf_counter()
        input_length = random.randint(500, 2000)
        gen_length = random.randint(500, 2000)
        curr_chunk = dataset[batch: batch+batch_size]
        inputs = tokenizer(curr_chunk['prompt'], padding=True, max_length=input_length, truncation=True, return_tensors="pt").to(model.device)
        with TorchTracemalloc() as tt:
            out, ttft = model.generate(**inputs, do_sample=False, min_new_tokens=gen_length-1, max_new_tokens=gen_length)

        TTFT.append(ttft)
        TIME_PER_DECODING.append((perf_counter() - start - ttft) / batch_size / gen_length)
        TOTAL_TIME_PER_TOKEN.append((perf_counter() - start) / batch_size / gen_length)
        del out
        gc.collect()

#print(prof.key_averages())
end_event.record()
torch.cuda.synchronize()
print(f"Total time: {start_event.elapsed_time(end_event) / 1000}")

print(TTFT, TIME_PER_DECODING)
print(TorchTracemalloc.track_memory_consumption)
