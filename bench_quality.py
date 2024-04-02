import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16, attn_implementation="eager").to("cuda:0")



# ---------------------------------------------------------------------------------LAMBADA -------------------------------------------------------------------------------------------

def calculate_perplexity():
    #dataset = load_dataset("lambada", split="test")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")

    max_length = model.config.max_position_embeddings
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"Perplexity: {ppl}")

# Perplexity: 6.09 for DynamicCache, same for quant

# ---------------------------------------------------------------------------------_MMLU -------------------------------------------------------------------------------------------

def collate_fn(examples):
    prompts = []
    for i in range(len(examples['question'])):
        prompt = f"{examples['question'][i].strip()}\nA. {examples['choices'][i][0]}\nB. {examples['choices'][i][1]}\nC. {examples['choices'][i][2]}\nD. {examples['choices'][i][3]}\nAnswer:"
        prompts.append(prompt)
    examples['prompt'] = prompts
    return examples


def calculate_mmlu():
    dataset = load_dataset("cais/mmlu", "all", split="test")
    dataset = dataset.map(collate_fn, batched=True)

    correct = 0
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    for example in tqdm(dataset):
        input_text = [f'{example["prompt"]} {choice}' for choice in example["choices"]]
        inputs = tokenizer(input_text, padding=True, return_tensors="pt").to(model.device)
        prompt_length = len(tokenizer(example["prompt"]).input_ids)
        target_ids = inputs["input_ids"].clone()
        target_ids[:, :prompt_length] = -100 # loss only on the choice/answer
        target_ids[target_ids == 2] = -100
        with torch.no_grad():
            outputs = model(**inputs)
        
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, model.config.vocab_size)
        shift_labels_reshaped = shift_labels.view(-1)
        loss = loss_fct(shift_logits, shift_labels_reshaped)
        loss = loss.view(shift_labels.shape).mean(dim=-1)

        pred = loss.argmin()
        gold = example["answer"]
        correct += (pred == gold)

    print(f"Accuracy on MMLU {correct / len(dataset) * 100}%")

# 41,2% for int4, and same for Dynamic but why the paper says it's 45.3%? 
# quanto settings: qint4, group_size=32, residual_size=128
calculate_perplexity()