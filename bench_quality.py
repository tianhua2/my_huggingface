import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm, trange

from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16, attn_implementation="eager").to("cuda:0")



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

# ---------------------------------------------------------------------------------LAMBADA -------------------------------------------------------------------------------------------

# this cannot be run as is, I tweaked manually the cache so that it quantized and dequantizes in the forward pass
# we do not support quant cache for forward, only in generate
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


# Perplexity: 6.09 (wikitext)
# Preplexity 16.1758 on lambada


# ---------------------------------------------------------------------------------ARC Challenge -------------------------------------------------------------------------------------------

# same as above since we do forward

def collate_fn_arc(examples):
    prompts = []
    for i in range(len(examples['question'])):
        choices_len = len(examples['choices'][i])
        temp = ""
        for j in range(choices_len):
            temp += f"\n{examples['choices'][i]['label'][j]}.{examples['choices'][i]['text'][j]} "
        prompt = f"{examples['question'][i].strip()}{temp}\nAnswer:"
        prompts.append(prompt)
    examples['prompt'] = prompts
    return examples


def calculate_arc():
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    dataset = dataset.map(collate_fn_arc, batched=True)

    correct = 0
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    for example in tqdm(dataset):
        input_text = [f'{example["prompt"]} {choice}' for choice in example["choices"]['label']]
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

        label2id = {lbl: idx for idx, lbl in enumerate(example["choices"]["label"])}
        pred = loss.argmin()
        gold = label2id[example["answerKey"]]
        correct += (pred == gold)

    print(f"Accuracy on ARC Challenge {(correct / len(dataset) * 100):.04f}%")


# ---------------------------------------------------------------------------------_MMLU -------------------------------------------------------------------------------------------

def collate_fn_mmlu(examples):
    prompts = []
    for i in range(len(examples['question'])):
        prompt = f"{examples['question'][i].strip()}\nA. {examples['choices'][i][0]}\nB. {examples['choices'][i][1]}\nC. {examples['choices'][i][2]}\nD. {examples['choices'][i][3]}\nAnswer:"
        prompts.append(prompt)
    examples['prompt'] = prompts
    return examples

# was same as above and using foward pass gives equal results for all evals
# below is a new version with generate ans constraints on generated text for match the format 
# when generating the overall score is less but the equivalence fp16-int4-int2 is almost the same

def calculate_mmlu(cache_implementation = None):
    dataset = load_dataset("cais/mmlu", "all", split="test")
    dataset = dataset.map(collate_fn_mmlu, batched=True)
    
    correct = 0
    map_id2choice = {0: "A", 1: "B", 2: "C", 3: "D"}
    bs = 16
    num_batches = 500
    for batch in trange(num_batches):
        curr_chunk = dataset[batch*bs : (batch+1)*bs]

        # init here, we need new grammar every call
        grammar = IncrementalGrammarConstraint("root ::= [A-D]", "root", tokenizer)
        grammar_processor = GrammarConstrainedLogitsProcessor(grammar)
        generation_kwargs = {"max_new_tokens": 1, "logits_processor": [grammar_processor]}
        if cache_implementation == "quantized":
            generation_kwargs["cache_implementation"] = "quantized"

        text = [f"{prompt} " for prompt in curr_chunk["prompt"]]
        inputs = tokenizer(text, padding=True, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, **generation_kwargs)
        seq_length = inputs.input_ids.shape[-1]

        preds = tokenizer.batch_decode(outputs[:, seq_length:])
        golds = [map_id2choice[ans] for ans in curr_chunk["answer"]]
        correct += sum((pred == gold) for gold, pred in zip(golds, preds))

    print(f"Accuracy on MMLU {(correct / (num_batches * bs) * 100):.04f}%")

# Acc MMLU: 41,2% 
# Accuracy on MMLU 25.1000% vs 24.7125% (fp16 vs int2)

calculate_arc()
