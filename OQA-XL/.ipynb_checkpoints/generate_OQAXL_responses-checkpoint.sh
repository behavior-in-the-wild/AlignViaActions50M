#!/bin/bash

MODEL=$1

python3 get_log_prob_hf_pref_xl.py $MODEL CREGION &

python3 get_log_prob_hf_pref_xl.py $MODEL AGE &

python3 get_log_prob_hf_pref_xl.py $MODEL EDUCATION &

python3 get_log_prob_hf_pref_xl.py $MODEL GENDER &

python3 get_log_prob_hf_pref_xl.py $MODEL POLIDEOLOGY &

python3 get_log_prob_hf_pref_xl.py $MODEL INCOME &

python3 get_log_prob_hf_pref_xl.py $MODEL POLPARTY &

python3 get_log_prob_hf_pref_xl.py $MODEL RACE &

python3 get_log_prob_hf_pref_xl.py $MODEL RELIG &

wait

echo "Finished"