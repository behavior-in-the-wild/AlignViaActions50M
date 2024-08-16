import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import ast
import sys
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch, dispatch_model

model_name = sys.argv[1]
keys = sys.argv[2]
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, device_map="auto",)

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")

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
    
audience_factor = {
    "CREGION": ["Northeast USA", "South USA", 'Midwest USA', 'West USA'],
    "AGE": ["18 to 29","30 to 49","50 to 64", "above 65"],
    "EDUCATION": ["College graduate/some postgrad", "Less than high school","High school graduate","Some college, no degree","Associate\'s degree","Postgraduate"],
    "GENDER": ["Male", "Female"],
    "POLIDEOLOGY": ["Liberal", "Conservative", "Moderate"],
    "INCOME": ["$100K+", "<$30,000"],
    "POLPARTY": ["Democrat", "Republican"],
    "RACE": ["Black", "White", "Asian", "Hispanic"],
    "RELIG": ["Protestant", "Jewish", "Hindu", "Atheist", "Muslim"]
}

texts = {
    "CREGION" : 'who is a resident of',
    "AGE": 'whose age is',
    "EDUCATION": 'whose level of education is',
    "GENDER": "who is",
    "POLIDEOLOGY": 'consider yourself a',
    "INCOME": 'whose income is',
    "POLPARTY": 'who voted for',
    "RACE": 'whose race is',
    "RELIG": 'whose religion is'
}

num_questions = 0

for factors in audience_factor[keys]:
    for surveys in tqdm(data):
        if(surveys not in survey_data):
            survey_data[surveys] = []

        for codes in tqdm(data[surveys]):                
            ques = data[surveys][codes]['question']

            if(len(ques.split(' '))<=5):
                continue

            ref = data[surveys][codes]['options']

            prompt = f'Answer the following question considering yourself as a person {texts[keys]} {factors}.'
            prompt += f' Question: {ques}\nPossible answers:'
            counter = ord('A') 
            l = []

            if(len(ref) > 50 and all(isinstance(item, float) for item in ref)):
                continue

            if(len(ref) > 50):
                continue

            num_questions += 1
            
            for val in ref:
                if(val=='Refused' or val.lower()=='nan'):
                    continue
                prompt+='\n'+chr(counter)+'. '+val
                l.append(chr(counter))
                counter+=1
    
            prompt+=f'\nAnswer only with {",".join(l[:-1])} or {l[-1]} and nothing else :'
    
            log_probs = {}
            counter = ord('A') 
            for val in ref:
                if(val=='Refused'):
                    continue

                log_prob = get_log_prob(prompt, chr(counter))
                log_probs[val] = log_prob
                counter+=1

            survey_data[surveys].append({'question':ques,'prompt':prompt, 'options':ref,'log_prob':log_probs, 'factor_value': factors, 'factor': keys})
            os.makedirs(f'results_pref_xl/{model_name.split("/")[1]}', exist_ok=True)

            with open(f'results_pref_xl/{model_name.split("/")[1]}/{model_name.split("/")[1]}_{keys}.json', 'w') as f:
                json.dump(survey_data, f)

print(f"Number of questions processed: {num_questions}")
