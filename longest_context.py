import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

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
        reserved = torch.cuda.max_memory_reserved() // 1024 ** 2
        peaked = (peak - self.begin) // 1024 ** 2
        TorchTracemalloc.track_memory_consumption.append(peaked)
        print(f"peaked: {peaked}; reserved: {reserved}")


def collate_fn(example):
        prompt=f"Question: {example['input']}\nContext: {example['context']}\nAnswer:"
        example['prompt'] = prompt
        return example


#torch.cuda.set_per_process_memory_fraction(0.2, device="cuda:0")
# max tokens -> 6k for dynamic and 12k for int4

dataset = load_dataset('THUDM/LongBench', "samsum", split='test')
dataset = dataset.map(collate_fn, batched=False)

#inputs = tokenizer(["\n".join(dataset["prompt"][:3])], max_length=7000, truncation=True, return_tensors="pt").to(model.device)
inputs = tokenizer("Hello, how are you?", truncation=True, return_tensors="pt").to(model.device)

with TorchTracemalloc() as tt:
    out = model.generate(**inputs, do_sample=False, max_new_tokens=20, cache_implementation="quantized")
    out_fp16 = model.generate(**inputs, do_sample=False, max_new_tokens=20)

print(f"text with quant cache: {tokenizer.batch_decode(out)}")
print(f"tetx with fp16: {tokenizer.batch_decode(out_fp16)}")

#print(torch.cuda.memory_summary("cuda:0"))
