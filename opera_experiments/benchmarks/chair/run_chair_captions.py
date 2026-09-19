import os
import json
import yaml
import argparse
import random
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

def run_chair(model_wrapper, data_paths, opera_config, use_opera, output_dir, num_samples=500, seed=42):
    coco_image_dir = data_paths['coco_val2014_images']
    chair_annotation_file = data_paths['chair_annotation_file']
    
    with open(chair_annotation_file, 'r') as f:
        coco_anno = json.load(f)
        
    images_info = coco_anno['images']
    
    # Fix seed to ensure both models/methods evaluate on the exact same 500 images
    random.seed(seed)
    sampled_images = random.sample(images_info, min(num_samples, len(images_info)))
    
    results = []
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "raw_outputs.jsonl")
    
    prompt = "Describe this image."
    max_tokens = opera_config.get("max_new_tokens", 128)
    
    with open(out_file, 'w') as f_out:
        for img_info in tqdm(sampled_images, desc="Running CHAIR (Image Captioning)"):
            image_name = img_info['file_name']
            image_id = img_info['id']
            
            image_path = os.path.join(coco_image_dir, image_name)
            image = Image.open(image_path).convert("RGB")
            
            if use_opera:
                pred = model_wrapper.generate_with_opera(prompt, image, opera_config, max_new_tokens=max_tokens)
            else:
                pred = model_wrapper.generate_baseline(prompt, image, max_new_tokens=max_tokens)
                
            result = {
                "image_id": image_id,
                "file_name": image_name,
                "prompt": prompt,
                "caption": pred
            }
            results.append(result)
            f_out.write(json.dumps(result) + "\n")
            f_out.flush()
            
    print(f"Finished generating captions for {len(sampled_images)} images. Saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["llava", "qwen2vl"], required=True)
    parser.add_argument("--use_opera", action="store_true")
    args = parser.parse_args()
    
    data_paths = load_config("../../data_paths.yaml")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "opera" if args.use_opera else "base"
    output_dir = f"../../results/{args.model}_chair_{mode}_{timestamp}"
    
    if args.model == "llava":
        opera_config = load_config("../../configs/opera_llava.yaml")
        model = LLaVAWrapper(data_paths["llava_1_5_7b_ckpt"])
    else:
        opera_config = load_config("../../configs/opera_qwen2vl.yaml")
        model = Qwen2VLWrapper(data_paths["qwen2vl_7b_instruct_ckpt"])
        
    # Log config
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "run_config.json"), 'w') as f:
        json.dump({"model": args.model, "use_opera": args.use_opera, "seed": opera_config.get("seed", 42), "opera_config": opera_config}, f, indent=4)
        
    run_chair(model, data_paths, opera_config, args.use_opera, output_dir, num_samples=500, seed=opera_config.get("seed", 42))
