import numpy as np
import pandas as pd
from argparse import ArgumentParser
import warnings


def calculate_accuracies(list_pairs):
    count = 0
    total_pairs = len(list_pairs)

    for list1, list2 in list_pairs:
        try:
            # First element being the same between the two lists
            if list1[0] == list2[0]:
                count += 1 
        except:
            warnings.warn(f"Unrecognized format, skipping sample")
    accuracy = count / total_pairs
    return accuracy


def eval(file):
    data = pd.read_json(file, lines=True)
    data = data[data['response'] != '']

    preds = data['ans'].tolist()
    refs = data['response'].tolist()

    if file.split('/')[-1].endswith('-r.jsonl'):
        new_preds = []
        new_refs = []
        for p, r in zip(preds, refs):
            try:
                p = p.split('Given')[0].strip().split('are ')[1].split('\n')[0].split(', ')
                r = r.split('are ')[1].split(', ')
                new_preds.append(p)
                new_refs.append(r)
            except:
                warnings.warn(f"Unrecognized format, skipping sample")
                
    elif file.split('/')[-1].endswith('-g.jsonl') or file.split('/')[-1].endswith('-a.jsonl'):
        new_preds = []
        new_refs = []
        for p, r in zip(preds, refs):
            try:
                p = p.split('Given')[0].strip().split('is ')[1].lower().split('\n')[0].split('.')[0]
                r = r.split('is ')[1].strip().lower()
                new_preds.append(p)
                new_refs.append(r)
            except:
                warnings.warn(f"Unrecognized format, skipping sample")

    preds = new_preds
    refs = new_refs

    pairs = list(zip(preds, refs))
    return calculate_accuracies(pairs)
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file", type=str)
    args = parser.parse_args()

    print("Accuracy:", eval(args.file))
   
