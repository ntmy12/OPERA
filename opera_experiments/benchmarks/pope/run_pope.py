import os
import json
import yaml
import argparse
import random
import time
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from datetime import datetime
from typing import Dict, Any, List, Optional

import sys
# Set up paths to import models and eval modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
REPO_ROOT = os.path.dirname(EXPERIMENTS_DIR)

if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from models.llava_wrapper import LLaVAWrapper, resolve_dtype
from models.qwen2vl_wrapper import Qwen2VLWrapper
from eval_pope import eval_pope


def set_seed(seed: int = 42):
    """Set random seed for reproducible benchmark evaluation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml_config(yaml_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    if not os.path.exists(yaml_path):
        return {}
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def auto_detect_coco_dir(override_path: Optional[str] = None, config_path: Optional[str] = None) -> str:
    """
    Intelligently detect COCO val2014 images directory.
    Priority:
      1. Explicit CLI override
      2. Known Kaggle dataset paths
      3. Deep scan inside /kaggle/input/
      4. Config path from YAML
      5. Fallback relative paths
    """
    if override_path and os.path.isdir(override_path):
        print(f"[Data Detection] Using user-specified COCO image dir: {override_path}")
        return override_path

    known_kaggle_paths = [
        "/kaggle/input/datasets/biminhco/val2014/val2014",
        "/kaggle/input/val2014/val2014",
        "/kaggle/input/coco-2014-val/val2014",
        "/kaggle/input/coco-val2014/val2014",
        "/kaggle/input/coco2014/val2014",
    ]
    for p in known_kaggle_paths:
        if os.path.isdir(p):
            print(f"[Data Detection] Found COCO images at standard Kaggle path: {p}")
            return p

    # Deep scan /kaggle/input/
    kaggle_input = "/kaggle/input"
    if os.path.isdir(kaggle_input):
        print("[Data Detection] Scanning /kaggle/input/ for COCO_val2014_*.jpg...")
        for root, _, files in os.walk(kaggle_input):
            for f in files[:100]:
                if f.startswith("COCO_val2014_") and f.lower().endswith((".jpg", ".jpeg", ".png")):
                    print(f"[Data Detection] Auto-detected COCO images at: {root}")
                    return root

    # Config path
    if config_path and os.path.isdir(config_path):
        print(f"[Data Detection] Using COCO images from config: {config_path}")
        return config_path

    # Local workspace fallbacks
    local_fallbacks = [
        os.path.join(REPO_ROOT, "dataset", "val2014"),
        os.path.join(REPO_ROOT, "val2014"),
        os.path.join(EXPERIMENTS_DIR, "dataset", "val2014"),
    ]
    for p in local_fallbacks:
        if os.path.isdir(p):
            print(f"[Data Detection] Found local fallback COCO images at: {p}")
            return p

    raise FileNotFoundError(
        "COCO val2014 image directory not found! Please specify via --coco_dir or mount the dataset in Kaggle."
    )


def auto_detect_pope_anno_dir(override_path: Optional[str] = None, config_path: Optional[str] = None) -> str:
    """
    Intelligently detect POPE annotations directory containing coco_pope_*.json files.
    """
    if override_path and os.path.isdir(override_path):
        return override_path

    candidates = []
    if config_path:
        candidates.append(config_path)
        candidates.append(os.path.join(REPO_ROOT, config_path))
        candidates.append(os.path.join(EXPERIMENTS_DIR, config_path))

    candidates.extend([
        os.path.join(REPO_ROOT, "pope_coco"),
        os.path.join(EXPERIMENTS_DIR, "pope_coco"),
        os.path.join(REPO_ROOT, "dataset", "pope_coco"),
        "/kaggle/input/pope-coco",
        "/kaggle/input/pope-annotations",
        "pope_coco",
    ])

    for c in candidates:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "coco_pope_random.json")):
            print(f"[Data Detection] Found POPE annotations at: {c}")
            return c

    raise FileNotFoundError(
        "POPE annotation directory not found! Ensure coco_pope_random.json is present in pope_coco/ or provide --pope_dir."
    )


def format_pope_prompt(question: str, model_name: str) -> str:
    """
    Format prompt per model requirements:
    - Qwen2-VL: append suffix ' Please answer with yes or no.'
    - LLaVA: retain original question without extra suffix.
    """
    q = question.strip()
    if model_name.lower() in ["qwen2vl", "qwen", "qwen2-vl"]:
        return f"{q} Please answer with yes or no."
    else:
        return q


def run_pope_split(
    model_wrapper,
    split: str,
    coco_image_dir: str,
    pope_anno_dir: str,
    output_dir: str,
    use_opera: bool,
    opera_config: Dict[str, Any],
    max_new_tokens: int = 6,
    model_name: str = "llava",
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run POPE benchmark for a single split.
    """
    pope_anno_file = os.path.join(pope_anno_dir, f"coco_pope_{split}.json")
    if not os.path.exists(pope_anno_file):
        raise FileNotFoundError(f"POPE annotation file not found: {pope_anno_file}")

    with open(pope_anno_file, 'r', encoding='utf-8') as f:
        pope_data = [json.loads(line) for line in f if line.strip()]

    if max_samples is not None and max_samples > 0:
        pope_data = pope_data[:max_samples]
        print(f"     [Sampling] Subsampled to first {len(pope_data)} samples for split '{split}'")

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "raw_outputs.jsonl")

    print(f"\n---> Starting POPE Split: '{split}' ({len(pope_data)} questions) | Model: {model_name} | OPERA: {use_opera}")
    print(f"     Max New Tokens: {max_new_tokens} | Decoding: Greedy (do_sample=False, temp=0.0)")

    sample_latencies = []
    with open(out_file, 'w', encoding='utf-8') as f_out:
        for idx, item in enumerate(tqdm(pope_data, desc=f"POPE {split} [{model_name}]")):
            image_name = item['image']
            question = item['text']
            label = item['label']

            prompt = format_pope_prompt(question, model_name)
            image_path = os.path.join(coco_image_dir, image_name)
            
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Warning: Failed to load image {image_path}: {e}")
                continue

            # Synchronize CUDA to measure precise inference latency per sample
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            # Greedy generation: do_sample=False, temperature=0.0, max_new_tokens=6
            if use_opera:
                pred = model_wrapper.generate_with_opera(
                    prompt,
                    image,
                    opera_config,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0
                )
            else:
                pred = model_wrapper.generate_baseline(
                    prompt,
                    image,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            latency = t_end - t_start
            sample_latencies.append(latency)

            # Record standard and compatible keys
            result = {
                "question_id": item.get('question_id', idx),
                "image": image_name,
                "question": question,
                "prompt": prompt,
                "label": label,
                "pred": pred,
                "answer": pred,
                "text": pred,
                "latency_s": round(latency, 4)
            }
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            f_out.flush()

    total_split_time = sum(sample_latencies)
    avg_split_time = (total_split_time / len(sample_latencies)) if sample_latencies else 0.0

    print(f"Finished split '{split}'. Raw outputs saved to: {out_file}")
    print(f"⏱️ Split '{split}' Timing: {len(sample_latencies)} samples | Total: {total_split_time:.2f}s | Avg: {avg_split_time:.4f}s/sample")

    # Immediately evaluate this split
    metrics = eval_pope(out_file)
    metrics["avg_time_per_sample_s"] = round(avg_split_time, 4)
    metrics["total_inference_time_s"] = round(total_split_time, 4)
    metrics["num_evaluated"] = len(sample_latencies)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
    print(f"Evaluated split '{split}': Accuracy={metrics['Accuracy']}%, F1={metrics['F1']}%, Avg Time/Sample={metrics['avg_time_per_sample_s']}s")

    return metrics


def print_summary_table(
    all_metrics: Dict[str, Dict[str, Any]],
    model_name: str,
    mode: str,
    overall_metrics: Optional[Dict[str, Any]] = None
):
    """Print an aesthetic summary table of benchmark results and timing across all splits."""
    col_split = 14
    col_metric = 10
    col_time = 18
    
    header = (
        f"{'Split':<{col_split}} | "
        f"{'Accuracy':>{col_metric}} | "
        f"{'Precision':>{col_metric}} | "
        f"{'Recall':>{col_metric}} | "
        f"{'F1-Score':>{col_metric}} | "
        f"{'Yes-Ratio':>{col_metric}} | "
        f"{'Unknowns':>{col_metric}} | "
        f"{'Avg Time/Sample':>{col_time}}"
    )
    sep = "=" * len(header)
    dash_sep = "-" * len(header)

    print("\n" + sep)
    print(f"   POPE BENCHMARK SUMMARY TABLE | Model: {model_name.upper()} | Mode: {mode.upper()}")
    print(sep)
    print(header)
    print(dash_sep)
    for split in ["random", "popular", "adversarial"]:
        if split not in all_metrics:
            continue
        m = all_metrics[split]
        acc = f"{m.get('Accuracy', 0):.2f}%"
        prec = f"{m.get('Precision', 0):.2f}%"
        rec = f"{m.get('Recall', 0):.2f}%"
        f1 = f"{m.get('F1', 0):.2f}%"
        yes_r = f"{m.get('Yes_ratio', 0):.2f}%"
        unk = str(m.get('Unknown_answers', 0))
        avg_t = m.get('avg_time_per_sample_s', None)
        time_str = f"{avg_t:.4f} s" if avg_t is not None else "N/A"
        print(
            f"{split:<{col_split}} | "
            f"{acc:>{col_metric}} | "
            f"{prec:>{col_metric}} | "
            f"{rec:>{col_metric}} | "
            f"{f1:>{col_metric}} | "
            f"{yes_r:>{col_metric}} | "
            f"{unk:>{col_metric}} | "
            f"{time_str:>{col_time}}"
        )
    print(dash_sep)
    if overall_metrics:
        tot_s = overall_metrics.get("total_samples", 0)
        tot_t = overall_metrics.get("total_inference_time_s", 0.0)
        avg_s = overall_metrics.get("overall_avg_time_per_sample_s", 0.0)
        print(f"⏱️ OVERALL TIMING ({tot_s} samples): Total Inference Time = {tot_t:.2f}s | Average Time / Sample = {avg_s:.4f}s")
    print(sep + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run POPE benchmark for OPERA with multi-GPU support")
    parser.add_argument("--model", type=str, choices=["llava", "qwen2vl"], required=True,
                        help="Model to benchmark (llava or qwen2vl)")
    parser.add_argument("--use_opera", action="store_true",
                        help="Enable OPERA hallucination mitigation")
    parser.add_argument("--split", type=str, choices=["random", "popular", "adversarial", "all"], default="all",
                        help="POPE split to evaluate, or 'all' to run all 3 splits sequentially")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate per split (e.g. 10 for latency profiling)")
    parser.add_argument("--max_new_tokens", type=int, default=6,
                        help="Max new tokens to generate (enforced 6 for POPE)")
    parser.add_argument("--coco_dir", type=str, default=None,
                        help="Path to COCO val2014 images directory")
    parser.add_argument("--pope_dir", type=str, default=None,
                        help="Path to POPE annotations directory")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to data paths YAML config")
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp16", "auto"], default="bf16",
                        help="Precision for model weights (BF16 with FP16 fallback)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device or device_map strategy (default: 'auto')")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Custom root directory to store benchmark results")

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # 1. Locate data paths configuration
    if args.config:
        config_path = args.config
    elif os.path.exists(os.path.join(EXPERIMENTS_DIR, "configs", "data_paths_kaggle.yaml")):
        config_path = os.path.join(EXPERIMENTS_DIR, "configs", "data_paths_kaggle.yaml")
    elif os.path.exists(os.path.join(EXPERIMENTS_DIR, "data_paths.yaml")):
        config_path = os.path.join(EXPERIMENTS_DIR, "data_paths.yaml")
    else:
        config_path = ""

    data_paths = load_yaml_config(config_path) if config_path else {}

    # 2. Auto-detect directories
    coco_image_dir = auto_detect_coco_dir(
        override_path=args.coco_dir,
        config_path=data_paths.get("coco_val2014_images")
    )
    pope_anno_dir = auto_detect_pope_anno_dir(
        override_path=args.pope_dir,
        config_path=data_paths.get("pope_coco_annotation_dir")
    )

    # 3. Determine splits
    if args.split == "all":
        splits_to_run = ["random", "popular", "adversarial"]
    else:
        splits_to_run = [args.split]

    # 4. Resolve precision & device
    dtype = resolve_dtype(args.dtype)
    print(f"\n[Environment Setup]")
    print(f"  PyTorch Version: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"  Detected {gpu_count} GPU(s):")
        for i in range(gpu_count):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Resolved Precision: {dtype}")
    print(f"  Device Strategy: {args.device}")

    # 5. Output directory setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "opera" if args.use_opera else "baseline"
    if args.output_dir:
        base_output_dir = args.output_dir
    else:
        base_output_dir = os.path.join(EXPERIMENTS_DIR, "results", f"{args.model}_pope_{mode}_{timestamp}")
    os.makedirs(base_output_dir, exist_ok=True)

    # 6. Load Model ONCE
    print(f"\n[Model Initialization] Loading model '{args.model}' (Once for splits: {splits_to_run})...")
    if args.model == "llava":
        opera_config_file = os.path.join(EXPERIMENTS_DIR, "configs", "opera_llava.yaml")
        opera_config = load_yaml_config(opera_config_file)
        model_ckpt = data_paths.get("llava_1_5_7b_ckpt", "llava-hf/llava-1.5-7b-hf")
        model = LLaVAWrapper(model_ckpt, device=args.device, dtype=dtype)
    else:
        opera_config_file = os.path.join(EXPERIMENTS_DIR, "configs", "opera_qwen2vl.yaml")
        opera_config = load_yaml_config(opera_config_file)
        model_ckpt = data_paths.get("qwen2vl_7b_instruct_ckpt", "Qwen/Qwen2-VL-7B-Instruct")
        model = Qwen2VLWrapper(model_ckpt, device=args.device, dtype=dtype)

    # 7. Execute all splits sequentially
    all_metrics = {}
    for current_split in splits_to_run:
        split_out_dir = os.path.join(base_output_dir, current_split)
        
        # Save run_config for this split
        run_config = {
            "model": args.model,
            "model_ckpt": model_ckpt,
            "use_opera": args.use_opera,
            "split": current_split,
            "max_samples": args.max_samples,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "temperature": 0.0,
            "seed": args.seed,
            "dtype": str(dtype),
            "device_map": args.device,
            "attn_implementation": "eager",
            "coco_image_dir": coco_image_dir,
            "pope_anno_dir": pope_anno_dir,
            "opera_config": opera_config if args.use_opera else {},
            "timestamp": timestamp,
            "environment": {
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
            }
        }
        os.makedirs(split_out_dir, exist_ok=True)
        with open(os.path.join(split_out_dir, "run_config.json"), 'w', encoding='utf-8') as f:
            json.dump(run_config, f, indent=4)

        metrics = run_pope_split(
            model_wrapper=model,
            split=current_split,
            coco_image_dir=coco_image_dir,
            pope_anno_dir=pope_anno_dir,
            output_dir=split_out_dir,
            use_opera=args.use_opera,
            opera_config=opera_config,
            max_new_tokens=args.max_new_tokens,
            model_name=args.model,
            max_samples=args.max_samples
        )
        all_metrics[current_split] = metrics

    # 8. Save overall summary with timing and print table
    total_samples_all = sum(m.get("num_evaluated", m.get("Total", 0)) for m in all_metrics.values())
    total_time_all = sum(m.get("total_inference_time_s", 0.0) for m in all_metrics.values())
    overall_avg_time = (total_time_all / total_samples_all) if total_samples_all > 0 else 0.0

    overall_metrics = {
        "total_samples": total_samples_all,
        "total_inference_time_s": round(total_time_all, 4),
        "overall_avg_time_per_sample_s": round(overall_avg_time, 4),
    }

    summary_data = {
        **all_metrics,
        "overall": overall_metrics
    }

    summary_path = os.path.join(base_output_dir, "summary_metrics.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=4)

    print_summary_table(all_metrics, model_name=args.model, mode=mode, overall_metrics=overall_metrics)
    print(f"All benchmarks finished successfully! Results saved in: {base_output_dir}\n")

