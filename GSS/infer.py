import os

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import ast
import sys

print("model_loading_started")

os.system('huggingface-cli login --token <TOKEN>')

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

config = PeftConfig.from_pretrained("../CultureBank-Mixtral-DPO")

base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", device_map="auto")
model = PeftModel.from_pretrained(base_model, "../CultureBank-Mixtral-DPO", device_map="auto")

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

from transformers import AutoModelForCausalLM
from peft import PeftModel

model_name = sys.argv[1]

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM
from peft import PeftModel, AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print("model_loading_ended")

def get_log_prob(inputs, outputs, tokenizer, model):

    # inputs = "How safe, if at all, would you say your local community is from crime? Would you say it is"
    # outputs = 'Very safe'
    
    input_tokens = tokenizer.encode(inputs, add_special_tokens=False, return_tensors='pt')
    output_tokens = tokenizer.encode(outputs, add_special_tokens=False, return_tensors='pt')
    
    # Concatenate input and output tokens
    tokens = torch.cat([input_tokens, output_tokens], dim=1)
    
    # Get model predictions for the entire sequence at once
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
    #     print([tokenizer.decode(token) for token in input_tokens_updated])
    #     print(f"Token: {last_token}, Log Prob: {log_token_prob}")
    # print(f"Total Log Sum Probability: {log_sum}")
    return log_sum

from tqdm import tqdm
import os
import pandas as pd
import ast
import json

survey_data = []
quest_dict = {}

with open('questions_final.json', 'r') as f:
    quest_dict = json.load(f)

for keys in tqdm(quest_dict.keys()):

    try:
        r = quest_dict[keys]
        ques = r['Question']
        ref = r['Options']
    
        if(len(ref)>=50):
            continue
    
        # prompt = 'Question: Please tell us whether you are satisfied or dissatisfied with your social life.\nA. Very satisfied\nB. Somewhat satisfied\nC. Somewhat dissatisfied\nD. Very dissatisfied\nAnswer:'
    
        prompt = f'Question: {ques}'
        
        counter = ord('A') 
        l = []
        for val in ref:
    
            if(val=='Refused'):
                continue
            prompt+='\n'+chr(counter)+'. '+str(val)
            l.append(chr(counter))
            counter+=1
    
        # prompt+='\nAnswer:'
        prompt+=f'\nAnswer only with {",".join(l[:-1])} or {l[-1]} and nothing else :'
    
        log_probs = []
        # print(prompt)
        # break
        counter = ord('A') 
        for val in ref:
    
            
            if(val=='Refused'):
                continue
    
            # log_prob = to_tokens_and_logprobs(model, tokenizer, [prompt+' '+chr(counter)])[0][-1]
            log_prob = get_log_prob(prompt, chr(counter), tokenizer, model)
            log_probs.append(log_prob)
            counter+=1
    
    
        survey_data.append({'question':ques,'prompt':prompt, 'options':ref,'log_prob':log_probs})
    
        import json
    
        with open(f'results/GSS_{model_name.split("/")[-1]}.json','w') as f:
            json.dump(survey_data,f)

    except:
        continue


