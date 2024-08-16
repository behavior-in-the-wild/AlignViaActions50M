import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import ast
import json
import pandas as pd
import sys

model_name = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(name, device_map="auto", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(name, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)

def get_log_prob(inputs, outputs):

    input_tokens = tokenizer.encode(inputs, add_special_tokens=False, return_tensors='pt')
    output_tokens = tokenizer.encode(outputs, add_special_tokens=False, return_tensors='pt')
    
    tokens = torch.cat([input_tokens, output_tokens], dim=1)
    
    with torch.no_grad():
        outputs = model(tokens)
        logits = outputs.logits
    
    logits = logits.float()
    log_sum = 0
    range_index = range(input_tokens.shape[1] - 1, tokens.shape[1] - 1)
    for i in range_index:
        past_tok, current_tok = i, i + 1
        token_logit = logits[0, past_tok, :]
        token_log_probs = torch.nn.functional.log_softmax(token_logit, dim=-1)
        log_token_prob = token_log_probs[tokens[0, current_tok]].item()
        log_sum += log_token_prob
    
        token = tokenizer.decode(tokens[:, current_tok])
    return log_sum

survey_data = {}
with open('opinionqa-xl.json','r') as f:
    data = json.load(f)

for surveys in tqdm(data):
    if(surveys not in survey_data):
        survey_data[surveys] = []

    for codes in tqdm(data[surveys]):            
        ques = data[surveys][codes]['question']

        if(len(ques.split(' '))<=5):
            continue

        ref = data[surveys][codes]['options']

        prompt = f'Question: {ques}'
        counter = ord('A') 
        l = []

        if(len(ref) > 50 and all(isinstance(item, float) for item in ref)):
            continue

        if(len(ref) > 50):
            continue

        for val in ref:

            if(val=='Refused' or val.lower()=='nan'):
                continue
            prompt+='\n'+chr(counter)+'. '+val
            l.append(chr(counter))
            counter+=1

        prompt+=f'\nAnswer only with {",".join(l[:-1])} or {l[-1]} and nothing else :'

        log_probs = {}
        counter = ord('A')
        if(len(ref) > 50 and all(isinstance(item, float) for item in ref)):
            continue
        
        if(len(ref) > 50):
            continue

        for val in ref:
            if(val=='Refused' or val.lower()=='nan'):
                continue
            log_prob = get_log_prob(prompt, chr(counter))
            log_probs[val] = log_prob
            counter+=1

        survey_data[surveys].append({'question':ques,'prompt':prompt, 'options':ref,'log_prob':log_probs, 'qcode': codes})
        with open(f'results_oaxl/{model_name}_oaxl_responses}.json','w') as f:
            json.dump(survey_data,f)

