import os
import json
import yaml
import argparse
from tqdm import tqdm
from PIL import Image
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.llava_wrapper import LLaVAWrapper
from models.qwen2vl_wrapper import Qwen2VLWrapper

def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def run_beaf(model_wrapper, data_paths, opera_config, use_opera, output_dir):
    beaf_dir = data_paths['beaf_data_dir']
    qna_file = os.path.join(beaf_dir, "beaf_qna.json")
    
    with open(qna_file, 'r') as f:
        beaf_data = json.load(f)
        
    results = []
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "raw_outputs.json")
    
    with open(out_file, 'w') as f_out:
        for item in tqdm(beaf_data, desc="Running BEAF Benchmark"):
            image_name = item['image']
            question = item['question']
            
            # Formatting prompt. BEAF is usually Yes/No QA.
            prompt = f"{question} Please answer yes or no."
            
            image_path = os.path.join(beaf_dir, image_name)
            image = Image.open(image_path).convert("RGB")
            
            if use_opera:
                pred = model_wrapper.generate_with_opera(prompt, image, opera_config, max_new_tokens=10)
            else:
                pred = model_wrapper.generate_baseline(prompt, image, max_new_tokens=10)
                
            # Add prediction to item
            result = item.copy()
            result['answer'] = pred
            results.append(result)
            
        json.dump(results, f_out, indent=4)
            
    print(f"Finished BEAF. Results saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["llava", "qwen2vl"], required=True)
    parser.add_argument("--use_opera", action="store_true")
    args = parser.parse_args()
    
    data_paths = load_config("../../data_paths.yaml")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "opera" if args.use_opera else "base"
    output_dir = f"../../results/{args.model}_beaf_{mode}_{timestamp}"
    
    if args.model == "llava":
        opera_config = load_config("../../configs/opera_llava.yaml")
        model = LLaVAWrapper(data_paths["llava_1_5_7b_ckpt"])
    else:
        opera_config = load_config("../../configs/opera_qwen2vl.yaml")
        model = Qwen2VLWrapper(data_paths["qwen2vl_7b_instruct_ckpt"])
        
    # Log config
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "run_config.json"), 'w') as f:
        json.dump({"model": args.model, "use_opera": args.use_opera, "opera_config": opera_config}, f, indent=4)
        
    run_beaf(model, data_paths, opera_config, args.use_opera, output_dir)
