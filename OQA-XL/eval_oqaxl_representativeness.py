import os
from tqdm import tqdm
from scipy.stats import wasserstein_distance
import numpy as np
import math
import json
import sys
import os
from tqdm import tqdm
from scipy.stats import wasserstein_distance
import numpy as np
import math
import pandas as pd


response_file = sys.argv[1]
with open(response_file,'r') as f:
    data = json.load(f)

with open('meta-data-ques.json','r') as f:
    meta_ques = json.load(f)

with open('survey-details.json','r') as f:
    survey_details = json.load(f)

all_survey_ques_code = {}
all_survey_ques_details = {}

for folders in tqdm(os.listdir('/disks/1/aanisha/ATP_data/PEW_ATP_Cleaned')):
    try:
        if '.zip' not in folders:
            survey = folders.split('_')[0]
            sav_path = f'/disks/1/aanisha/ATP_data/PEW_ATP_Cleaned/{folders}/{folders}/ATP {survey}.sav'

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

        try:
            survey = folders.split('_')[0]
            sav_path = f'/disks/1/aanisha/ATP_data/PEW_ATP_Cleaned/{folders}/ATP {survey}.sav'
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


response_comparison_model = {}
for key in tqdm(data.keys()):
    sav_path = all_survey_ques_details[key]['sav_path']
    df = pd.read_spss(sav_path)
    response_comparison_model[key] = {}

    for resp in data[key]:
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

            # Iterate over the list and count the frequencies
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

            # Iterate over the list and count the frequencies
            for response in resp_values:
                if(str(response).lower()!='refused' and str(response).lower()!='nan'):
                    if response in frequency_dict:
                        frequency_dict[response] += 1
                    else:
                        frequency_dict[response] = 1
            response_comparison_model[key][resp['qcode']+'_'+key] = {'model_response':resp['log_prob'], 'audience_response':frequency_dict}

sum = 0
q = []

for surveys in response_comparison_model:
    for question in response_comparison_model[surveys]:
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

print('Alignment= ',sum/6188)
print(sum)

            