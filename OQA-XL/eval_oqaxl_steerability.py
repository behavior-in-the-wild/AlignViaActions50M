import pandas as pd
import traceback
from tqdm import tqdm
from scipy.stats import wasserstein_distance
import numpy as np
import math
import json
import os
from pprint import pprint
import glob


generations_path = "/sensei-fs/users/susmita/AVA/AlignViaActions50M/OQA-XL/results_pref_xl/vicuna-7b-v1.5"
if generations_path[-1] == '/':
    generations_path = generations_path[:-1]

files = [file for file in glob.glob(f'{generations_path}/*.json')]
with open('meta-data-ques.json','r') as f:
    meta_ques = json.load(f)
with open('survey-details.json','r') as f:
    survey_details = json.load(f)
with open('opinionqa-xl.json', 'r') as f:
    data = json.load(f)

question_details = json.load(open('oa-xl.json', 'r'))
question_code_dict = {}
for k, v in question_details.items():
    for k_, v_ in v.items():
        question_code_dict[v_['question']] = k_

survey_details_new = {}

for sur in survey_details:
    survey_details_n = sur
    survey_details_n['key'] = sur['name'].replace('ave ','')
    survey_details_n['year'] = int(sur['end_date'].split(',')[1])
    survey_details_new[sur['name'].replace('ave ','')] = survey_details_n


question_code_dict = {}
for k, v in data.items():
    for k_, v_ in v.items():
        question_code_dict[v_['question']] = k_

all_survey_ques_code = {}
all_survey_ques_details = {}

for folders in tqdm(os.listdir('ATP_data/PEW_ATP_Cleaned')):
    try:
        if '.zip' not in folders:
            survey = folders.split('_')[0]
            sav_path = f'ATP_data/PEW_ATP_Cleaned/{folders}/{folders}/ATP {survey}.sav'
            if not os.path.exists(sav_path):
                sav_path = f'/disks/1/aanisha/ATP_data/PEW_ATP_Cleaned/{folders}/ATP {survey}.sav'
                if not os.path.exists(sav_path):
                    continue
            df = pd.read_spss(sav_path)

            cols = list(df.columns)
            meta_cols = list(meta_ques.keys())
            cat_cols = []
            col_mappings = {}

            all_survey_ques_details[folders.split('_')[0]] = {
                "column_mappings": {},
                "unique_options": {},
                "sav_path": sav_path,
                "survey_details": {}
            }

            for d in cols[4:]:
                if d not in meta_cols and 'FORM' != d and d[:2] != 'F_' and 'WEIGHT' not in d:
                    if '_' in d:
                        processed_col_name = d[:d.rindex('_')]
                    else:
                        processed_col_name = d
                    cat_cols.append(processed_col_name)
                    col_mappings[processed_col_name] = d
                    unique_options = list(df[d].unique())
                    all_survey_ques_details[folders.split('_')[0]]["unique_options"][d] = unique_options

            all_survey_ques_code[folders.split('_')[0]] = cat_cols
            all_survey_ques_details[folders.split('_')[0]]["column_mappings"] = col_mappings
            if(folders.split('_')[0] in survey_details_new.keys()):
                all_survey_ques_details[folders.split('_')[0]]["survey_details"] = survey_details_new[folders.split('_')[0]]

    except Exception as e:
        try:
            survey = folders.split('_')[0]
            sav_path = f'ATP_data/PEW_ATP_Cleaned/{folders}/ATP {survey}.sav'

            df = pd.read_spss(sav_path)

            cols = list(df.columns)
            meta_cols = list(meta_ques.keys())
            cat_cols = []
            col_mappings = {}

            all_survey_ques_details[folders.split('_')[0]] = {
                "column_mappings": {},
                "unique_options": {},
                "sav_path": sav_path,
                "survey_details": {}
            }

            for d in cols[4:]:
                if d not in meta_cols and 'FORM' != d and d[:2] != 'F_' and 'WEIGHT' not in d:
                    if '_' in d:
                        processed_col_name = d[:d.rindex('_')]
                    else:
                        processed_col_name = d
                    cat_cols.append(processed_col_name)
                    col_mappings[processed_col_name] = d
                    unique_options = list(df[d].unique())
                    all_survey_ques_details[folders.split('_')[0]]["unique_options"][d] = unique_options

            all_survey_ques_code[folders.split('_')[0]] = cat_cols
            all_survey_ques_details[folders.split('_')[0]]["column_mappings"] = col_mappings
            if(folders.split('_')[0] in survey_details_new.keys()):
                all_survey_ques_details[folders.split('_')[0]]["survey_details"] = survey_details_new[folders.split('_')[0]]

        except Exception as e:
            print(f"An error occurred with folder {folders}: {e}")
            traceback.print_exc()

concept_dict = {}

for keys in all_survey_ques_details.keys():
    if('concept' in all_survey_ques_details[keys]['survey_details']):
        concept = all_survey_ques_details[keys]['survey_details']['concept']
        if(concept==''):
            concept='Unk'
        if(concept not in concept_dict):
            concept_dict[concept] = []
        concept_dict[concept].append(keys)
    else:
        concept = "Unk"
        if(concept not in concept_dict):
            concept_dict[concept] = []
        concept_dict[concept].append(keys)


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

model_steerability_dict = {}

for file in files:
    factor = [k for k in audience_factor if k in file.split('/')[-1].split('.json')[0]][0]

    with open(file,'r') as f:
        try:
            data = json.load(f)
        except:
            continue
    for factor_value in audience_factor[factor]:
        response_comparison_model = {}
        try:
            steerability_dict_key = factor + '_' + str(factor_value)
        except:
            continue

        for key in all_survey_ques_details.keys():
            sav_path = all_survey_ques_details[key]['sav_path']
        
            df = pd.read_spss(sav_path)
        
            response_comparison_model[key] = {}

            if(key not in data):
                continue
        
            for resp in data[key]:
                question_factor_value = resp['prompt'].split(f"Answer the following question considering yourself as a person {texts[factor]} ")[1].split('.')[0]
                if question_factor_value != factor_value:
                    continue

                resp['qcode'] = question_code_dict[resp['question']]
                if(resp['qcode'] in df.columns):
        
        
                    if('F_CREGION_FINAL' in df.columns):
                        df['F_CREGION'] = data['F_CREGION_FINAL']
                
                    elif('F_CREGION_RECRUITMENT' in df.columns):
                        df['F_CREGION'] = data['F_CREGION_RECRUITMENT']
                
                    elif('F_CREGION_TYPOGRAPHY' in df.columns):
                        df['F_CREGION'] = data['F_CREGION_TYPOGRAPHY']
                
                    elif('F_CREGION' in df.columns):
                        df['F_CREGION'] = data['F_CREGION']
                
                    resp_values = df[resp['qcode']].tolist()        
                    frequency_dict = {}
            
                    for response in resp_values:
                        if response in frequency_dict:
                            frequency_dict[response] += 1
                        else:
                            frequency_dict[response] = 1
        
                    response_comparison_model[key][resp['qcode']+'_'+key] = {'model_response':resp['log_prob'], 'audience_response':frequency_dict}
            
                elif(resp['qcode']+'_'+key in df.columns):
        
                    if('F_CREGION_FINAL' in df.columns):
                        df['F_CREGION'] = df['F_CREGION_FINAL']
                
                    elif('F_CREGION_RECRUITMENT' in df.columns):
                        df['F_CREGION'] = df['F_CREGION_RECRUITMENT']
                
                    elif('F_CREGION_TYPOGRAPHY' in df.columns):
                        df['F_CREGION'] = df['F_CREGION_TYPOGRAPHY']
                
                    elif('F_CREGION' in df.columns):
                        df['F_CREGION'] = df['F_CREGION']
                
                    resp_values = df[resp['qcode']+'_'+key].tolist()        
                    frequency_dict = {}
        
                    for response in resp_values:
        
                        if(str(response).lower()!='refused' and str(response).lower()!='nan'):
                            if response in frequency_dict:
                                frequency_dict[response] += 1
                            else:
                                frequency_dict[response] = 1
                    response_comparison_model[key][resp['qcode']+'_'+key] = {'model_response':resp['log_prob'], 'audience_response':frequency_dict}            
            
        sum = 0
        q = []
        c = 0
        for surveys in response_comparison_model:
            for question in response_comparison_model[surveys]:
                c+=1
                
                dist_audience = []
                dist_llm = []
        
                q.append(surveys+'_'+question)

                for keys in response_comparison_model[surveys][question]['model_response']:
        
                    if(keys not in response_comparison_model[surveys][question]['audience_response']):
                        dist_audience.append(0)
                    else:
        
                        dist_audience.append(response_comparison_model[surveys][question]['audience_response'][keys])
                    
                    dist_llm.append(math.exp(response_comparison_model[surveys][question]['model_response'][keys]))
        
                if(len(dist_audience)==0):
                    continue
        
                dist_llm = [d/np.sum(dist_llm) for d in dist_llm]
                dist_audience = [d/np.sum(dist_audience) for d in dist_audience]
                    
                w = wasserstein_distance(dist_audience, dist_llm)
                n = len(response_comparison_model[surveys][question]['model_response'].keys())
                v = 1 - (w/(n-1))
                if(~np.isnan(v)):
                    sum+=v
        
        try:
            scale_factor = 1/6188
            model_steerability_dict[steerability_dict_key] = sum*scale_factor
        except:
            pass


out_dict = {}
for k, v in model_steerability_dict.items():
    if v > 0:
        out_dict[k] = v
print("Steerability Score:", np.mean(list(out_dict.values())))
