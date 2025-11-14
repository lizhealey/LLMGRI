import pickle
from datetime import datetime
import json

import os
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

os.environ["ANTHROPIC_API_KEY"] = api_key

def make_batch_json_anthropic(prompts, batch_dir , model_anthropic):
    timestamp = datetime.now().strftime("%Y_%m_%d_%Hh")
    i=0
    requests = []
    for i in range(len(prompts)):
        prompt = prompts[i]
        requests.append(
            Request(
                custom_id=f"file-{i}",
                params=MessageCreateParamsNonStreaming(
                    model=model_anthropic,
                    max_tokens=1024,
                    messages=[{
                        "role": "user",
                        "content": prompt,
                    }]
                )
            ))
    
    
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=requests)
       
    return batch,requests

def save_json(data, directory,  custom_ending):
    os.makedirs(directory, exist_ok=True)
    filename = f"{custom_ending}"
    filepath = os.path.join(directory, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"JSON saved to: {filepath}")
    return filepath


def load_json(directory, filename="batch_sent.jsonl"):
    with open(directory+'/' +filename, 'r') as f:
        data = json.load(f)
    return data


#############################################################################
# Perturbations
#############################################################################

def make_batch_json_anthropic_perturbation(prompt_list, batch_dir , model_anthropic):

    requests = []
    i=0
    save_dict = {}
    for (hypo, hyper,tir,id), prompt in prompt_list.items():
        save_dict[i] = {
            "hypo": hypo,
            "hyper": hyper,
            "tir": tir,
            "id": id,
            "prompt": prompt
        }
        i+=1
        
        requests.append(
            Request(
                custom_id=f"file-{i}",
                params=MessageCreateParamsNonStreaming(
                    model=model_anthropic,
                    max_tokens=1024,
                    messages=[{
                        "role": "user",
                        "content": prompt,
                    }]
                )
            ))
    
    api_key = api_key
    pickle.dump(save_dict, open(f"{batch_dir}/batch_input.pickle", "wb"))
    os.environ["ANTHROPIC_API_KEY"] = api_key
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=requests)
       
    return batch,requests