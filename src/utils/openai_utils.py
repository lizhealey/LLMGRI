from openai import OpenAI
import os
import pickle
from datetime import datetime
import json


OPENAI_API_KEY =api_key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def make_batch_json_openai(prompts, batch_dir, model ):
    timestamp = datetime.now().strftime("%Y_%m_%d_%Hh")
    output_file = f"{batch_dir}/batch_input_at_{timestamp}.jsonl"
    i=0
    with open(output_file, "w") as f:
        for i in range(len(prompts)):
            prompt = prompts[i]
            batch_item = {"custom_id": f"file-{i}", 
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": model,
                    "input":  prompt, 
                       
                }
            }
            f.write(json.dumps(batch_item) + "\n")
            i+=1
    print(f"Saved {len(prompts)} batch requests to {output_file}")
    return output_file

def save_input_file_openai(batch_input_file, batch_dir, batch_json):
    input_file_data = {
        "file_id": batch_input_file.id,
        "filename": batch_input_file.filename,
        "batch_input_file_path": batch_json,
    }
    with open(os.path.join(batch_dir, "input_file_metadata.json"), "w") as f:
        json.dump(input_file_data, f, indent=2)
    print('Saved')

def save_batch_file_openai(batch_input_batch, batch_dir):
    batch_metadata = {
        "batch_id": batch_input_batch.id,
        "status": batch_input_batch.status,
        "input_file_id": batch_input_batch.input_file_id,
        "output_file_id": batch_input_batch.output_file_id,
        "endpoint": batch_input_batch.endpoint,
        "metadata": batch_input_batch.metadata }
    with open(os.path.join(batch_dir, "input_batch_metadata.json"), "w") as f:
        json.dump(batch_metadata, f, indent=2)
    print('Saved')
    

def load_batch_file_openai(batch_dir):
    metadata_path = os.path.join(batch_dir, "input_batch_metadata.json")
    with open(metadata_path, "r") as f:
        batch_metadata = json.load(f)
    return batch_metadata


def make_batch_json_openai_perturbation(prompt_list, batch_dir , model ):
    # different for perturbations
    timestamp = datetime.now().strftime("%Y_%m_%d_%Hh")
    output_file = f"{batch_dir}/batch_input_at_{timestamp}.jsonl"
    i=0
    save_dict = {}
    with open(output_file, "w") as f:

        for (hypo, hyper,tir,id), prompt in prompt_list.items():
            save_dict[i] = {
                "hypo": hypo,
                "hyper": hyper,
                "tir": tir,
                "id": id,
                "prompt": prompt
            }
            
            i+=1
    
        
            batch_item = {"custom_id": f"file-{i}", 
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": model,
                    "input":  prompt, 
                       
                }
            }
            
            f.write(json.dumps(batch_item) + "\n")
    pickle.dump(save_dict, open(f"{batch_dir}/batch_input.pickle", "wb"))
        
    print(f"Saved {len(prompt_list)} batch requests to {output_file}")
    return output_file