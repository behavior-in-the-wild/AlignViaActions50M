import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import evaluate

from argparse import ArgumentParser
# import yake
import re
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams

parser = ArgumentParser()
parser.add_argument("--file", nargs='+', default=[])
parser.add_argument("--model", type=str)
parser.add_argument("--compute_perplexity_diff", action='store_true')
parser.add_argument("--num_splits", type=int, default=1, help="Number of batches to split evaluation into, when computing perplexities")
args = parser.parse_args()

data = None
for file in args.file:
print("Reading", file)
new_data = pd.read_json(file, lines=True)
if data is None:
    data = new_data[new_data['response'] != '']
else:
    data = pd.concat([data, new_data])

if args.compute_perplexity_diff:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = LLM(args.model)
    sp = SamplingParams(max_tokens=1, detokenize=True, prompt_logprobs=1)

@torch.no_grad()
def perplexity(prompt, response):
    orig_content = []
    new_content = []
    for p, r in zip(prompt, response):
        try:
            orig_ad = p.split('Current Advertisement:')[1].split('Answer:')[0]
        except:
            orig_ad = p.split('Below is the original advertisement:\nAdvertisement:')[1].split('Answer:')[0]
        target_ad = r
        new_prompt = p.split(',')[0].replace('Given an advertisement ', 'Generate an advertisement') + '. Answer: '
        orig_content.append(new_prompt + orig_ad)
        new_content.append(new_prompt + target_ad)
    old_answers = model.generate(orig_content, sp, use_tqdm=False)
    new_answers = model.generate(new_content, sp, use_tqdm=False)
    old_ppls = []
    new_ppls = []
    
    for ao, an in zip(old_answers, new_answers):
        ans_old = ao.outputs[0].text.strip()
        ans_new = an.outputs[0].text.strip()
        logprobs_old = ao.prompt_logprobs
        logprobs_new = an.prompt_logprobs
        if logprobs_old is None or logprobs_new is None:
            continue
        logprobs_old = [lp for lp in logprobs_old if lp is not None]
        logprobs_new = [lp for lp in logprobs_new if lp is not None]
        ppl = np.exp(-np.mean([list(entry.values())[0].logprob for entry in logprobs_old]))
        old_ppls.append(ppl)
        ppl = np.exp(-np.mean([list(entry.values())[0].logprob for entry in logprobs_new]))
        new_ppls.append(ppl)
        
    return old_ppls, new_ppls
        
    
bleu = evaluate.load('bleu')
rouge = evaluate.load('rouge')
bertscore = evaluate.load("bertscore")

preds = data['ans'].tolist()
refs = data['response'].tolist()

if args.compute_perplexity_diff:
    try:
        ppl_data = data['prompt_new'].tolist()
    except:
        ppl_data = data['prompt'].tolist()
    
    if args.num_splits > 1:
        ppl_data_parts = split_list(ppl_data, args.num_splits)
        ref_parts = split_list(refs, args.num_splits)
        old_ppls = []
        new_ppls = []
        i = 0
        for part, ref in tqdm(zip(ppl_data_parts, ref_parts)):
            old_part, new_part = perplexity(part, ref)
            old_ppls.extend(old_part)
            new_ppls.extend(new_part)
            i += 1
        
    else:
        old_ppls, new_ppls = perplexity(ppl_data, refs)

refs = [[r] for r in refs]
bleu_score = bleu.compute(predictions=preds, references=refs)
rouge_score = rouge.compute(predictions=preds, references=refs)
bert_score = bertscore.compute(predictions=preds, references=refs, lang='en')
print("BLEU:", bleu_score["precisions"])
print("Rouge:", rouge_score['rougeLsum'])
print("BERTScore:", np.mean(bert_score['f1']))
print("Delta PPL:", np.mean(old_ppls) - np.mean(new_ppls))

if args.compute_perplexity_diff:
    print("Old perplexities: ", np.mean(old_ppls), "New perplexities: ", np.mean(new_ppls))
