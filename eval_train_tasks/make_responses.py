import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" 
os.environ["WORLD_SIZE"] = "1"

from transformers import AutoTokenizer
import transformers
import torch
import json
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyPairDataset
import numpy as np
from argparse import ArgumentParser
import gc
import glob
from peft import PeftModel
from vllm import LLM, SamplingParams


def format_prompt(prompt: str) -> str:
    return prompt.strip()

def append_in_context(prompt, samples, responses):
    prefix = '\n'.join([(s + r) for s, r in zip(samples, responses)])
    return prefix + prompt


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="/sensei-fs/users/susmita/hf_llama13b_fb_transcreation_ver3_aud")
    parser.add_argument("--task_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--max_new_tokens", type=int, default = 5)
    parser.add_argument("--in_context_samples", type=int, default=0)
    parser.add_argument("--is-peft", action="store_true")
    args = parser.parse_args()
            
    task_file = args.task_file
    
    print("Loading model")
    if args.is_peft:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-chat-hf", device_map="auto", trust_remote_code=True, torch_dtype=torch.float16, offload_folder="offload")
        peft_model_id = args.model
        model = PeftModel.from_pretrained(base_model, peft_model_id, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16, offload_folder="offload")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf",  device_map="auto")
        model = model.merge_and_unload()
        model.save_pretrained(args.model + '_merged')
        tokenizer.save_pretrained(args.model + '_merged')
        model = args.model + '_merged'
    else:
        model = args.model
        
    tensor_parallel_size=len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if tensor_parallel_size > 1:
        print(f"Using {tensor_parallel_size} GPUs")
        model = LLM(model, tensor_parallel_size=tensor_parallel_size, dtype="bfloat16")
    else:
        model = LLM(model)
    
    print("Processing", task_file)
    if task_file.startswith('pred') or task_file.startswith('cmp'):
        args.max_new_tokens = 10
    else:
        args.max_new_tokens = 100
        
    test_task = pd.read_json(task_file, lines=task_file.endswith('.jsonl'))
    try:
        prompts = test_task['prompt_new'].tolist()
        responses = test_task['response'].tolist()
    except KeyError:
        try:
            prompts = test_task['prompt'].tolist()
            responses = test_task['response'].tolist()
        except KeyError:
            raise ValueError("Unrecognized Format")
            exit()
        
    formatted_prompts = [format_prompt(p) for p in prompts]
    sampling_params = SamplingParams(max_tokens=args.max_new_tokens, detokenize=True)
    response_dicts = []

    if args.in_context_samples > 0:
        print("Reached")
        in_context_samples = formatted_prompts[:args.in_context_samples]
        in_context_responses = responses[:args.in_context_samples]
        formatted_prompts = formatted_prompts[args.in_context_samples:]
        responses = responses[args.in_context_samples:]
        formatted_prompts = [append_in_context(p, in_context_samples, in_context_responses) for p in formatted_prompts]
    
    print("Starting generation")
    num_chunks = max(len(formatted_prompts)//500, 1)
    formatted_prompt_batches = np.array_split(formatted_prompts, num_chunks)
    response_batches = np.array_split(responses, num_chunks)

    # Use at most 10 batches (5000 samples)
    while len(formatted_prompt_batches) > 10:
        formatted_prompt_batches = formatted_prompt_batches[::2]
        response_batches = response_batches[::2]

    os.system(f'rm {args.output_file}')

    for i, (bp, br) in enumerate(tqdm(zip(formatted_prompt_batches, response_batches), total=len(formatted_prompt_batches))):
        answers = model.generate(bp, sampling_params, use_tqdm=False)
        for p, r, a in zip(bp, br, answers):
            ans = a.outputs[0].text.strip()
            with open(args.output_file, 'a', encoding='utf8') as f:
                f.write(json.dumps({'prompt': p, 'response': r, 'ans': ans}, ensure_ascii=False) + '\n')
    
