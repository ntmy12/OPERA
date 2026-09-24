import os
import sys
import time
import json
import yaml
import glob
import random
import argparse
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import torch
import numpy as np

# Ensure line buffering on stdout/stderr for immediate streaming on Kaggle / Colab
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

# Set up paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
REPO_ROOT = os.path.dirname(EXPERIMENTS_DIR)

if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from models.llava_wrapper import LLaVAWrapper, resolve_dtype
from models.qwen2vl_wrapper import Qwen2VLWrapper
from eval_chair import evaluate_chair


def set_seed(seed: int = 42):
    """Set random seed for reproducible sampling and evaluation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml_config(yaml_path: str) -> dict:
    """Load configuration from a YAML file."""
    if not os.path.exists(yaml_path):
        return {}
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def auto_detect_coco_dir(override_path: str = None, config_path: str = None) -> str:
    """Auto-detect COCO val2014 images directory on Kaggle, Colab, or local."""
    if override_path and os.path.isdir(override_path):
        print(f"[Data Detection] Using user-specified COCO image dir: {override_path}", flush=True)
        return override_path

    known_kaggle_paths = [
        "/kaggle/input/datasets/biminhco/val2014/val2014",
        "/kaggle/input/datasets/biminhco/val2014",
        "/kaggle/input/val2014/val2014",
        "/kaggle/input/coco-2014-val/val2014",
        "/kaggle/input/coco-val2014/val2014",
        "/kaggle/input/coco2014/val2014",
    ]
    for p in known_kaggle_paths:
        if os.path.isdir(p):
            print(f"[Data Detection] Found COCO images at standard Kaggle path: {p}", flush=True)
            return p

    # Deep scan /kaggle/input/
    kaggle_input = "/kaggle/input"
    if os.path.isdir(kaggle_input):
        for root, _, files in os.walk(kaggle_input):
            if any(f.startswith("COCO_val2014_") for f in files[:50]):
                print(f"[Data Detection] Auto-detected COCO images at: {root}", flush=True)
                return root

    # Config path
    if config_path and os.path.isdir(config_path):
        print(f"[Data Detection] Using COCO images from config: {config_path}", flush=True)
        return config_path

    # Local workspace fallbacks
    local_fallbacks = [
        os.path.join(REPO_ROOT, "dataset", "val2014"),
        os.path.join(REPO_ROOT, "val2014"),
        os.path.join(EXPERIMENTS_DIR, "dataset", "val2014"),
    ]
    for p in local_fallbacks:
        if os.path.isdir(p):
            print(f"[Data Detection] Found local fallback COCO images at: {p}", flush=True)
            return p

    raise FileNotFoundError("COCO val2014 image directory not found! Provide via --coco_dir or mount in Kaggle.")


def find_image_file(image_dir: str, image_name: str) -> str:
    """Find image file directly or in subdirectories."""
    direct = os.path.join(image_dir, image_name)
    if os.path.isfile(direct):
        return direct
    sub = os.path.join(image_dir, "val2014", image_name)
    if os.path.isfile(sub):
        return sub
    return direct


def extract_image_id(file_name: str) -> int:
    """Extract numeric image id from COCO image filename."""
    base = os.path.basename(file_name).split(".")[0]
    num_part = base.split("_")[-1]
    return int(num_part)


def load_chair_samples(image_dir: str, num_samples: int = 10, seed: int = 2027, manifest_path: str = None) -> list:
    """
    Load samples for CHAIR benchmark.
    Priority:
      1. Pre-computed manifest (e.g. selected_chair_val2014_seed2027.json)
      2. Deterministic sampling from image directory with fixed seed
    """
    candidates = []
    if manifest_path and os.path.isfile(manifest_path):
        candidates.append(manifest_path)
    default_manifest = os.path.join(CURRENT_DIR, f"selected_chair_val2014_seed{seed}.json")
    if os.path.isfile(default_manifest):
        candidates.append(default_manifest)

    for m in candidates:
        try:
            with open(m, "r", encoding="utf-8") as f:
                data = json.load(f)
            samples = data.get("samples", data)
            if isinstance(samples, list) and len(samples) >= num_samples:
                print(f"[Sampling] Using verified manifest ({len(samples[:num_samples])}/{len(samples)} images): {m}", flush=True)
                return samples[:num_samples]
        except Exception as e:
            print(f"[Sampling Warning] Failed loading manifest {m}: {e}", flush=True)

    # Fallback: Sample directly from image_dir
    print(f"[Sampling] Sampling {num_samples} images from {image_dir} with seed {seed}...", flush=True)
    all_files = sorted([f for f in os.listdir(image_dir) if f.startswith("COCO_val2014_") and f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if not all_files:
        val_sub = os.path.join(image_dir, "val2014")
        if os.path.isdir(val_sub):
            all_files = sorted([f for f in os.listdir(val_sub) if f.startswith("COCO_val2014_") and f.lower().endswith((".jpg", ".jpeg", ".png"))])

    chosen = random.Random(seed).sample(all_files, min(num_samples, len(all_files)))
    return [{"image_id": extract_image_id(f), "file_name": f} for f in chosen]


def print_chair_summary_table(metrics: dict, model_name: str, mode: str, num_samples: int):
    """Print clean aesthetic summary table of CHAIR results and timing."""
    sep = "=" * 92
    dash = "-" * 92

    chairs = f"{metrics.get('CHAIRs', 0):.2f}%"
    chairi = f"{metrics.get('CHAIRi', 0):.2f}%"
    recall = f"{metrics.get('Recall', 0):.2f}%"
    cap_len = f"{metrics.get('Caption_Length', 0):.2f}"
    avg_t = metrics.get("avg_time_per_sample_s", 0.0)
    tot_t = metrics.get("total_inference_time_s", 0.0)
    throughput = f"{1.0 / avg_t:.2f} it/s" if avg_t > 0 else "N/A"

    print("\n" + sep, flush=True)
    print(f"   CHAIR BENCHMARK RESULTS | Model: {model_name.upper()} | Method: {mode.upper()} | Samples: {num_samples}", flush=True)
    print(sep, flush=True)
    header = (
        f"{'Model':<10} | {'Method':<10} | {'CHAIRs':>9} | {'CHAIRi':>9} | {'Recall':>9} | "
        f"{'Cap Len':>8} | {'Avg Time/Sample':>16} | {'Total Time':>12}"
    )
    print(header, flush=True)
    print(dash, flush=True)
    row = (
        f"{model_name.upper():<10} | {mode.upper():<10} | {chairs:>9} | {chairi:>9} | {recall:>9} | "
        f"{cap_len:>8} | {f'{avg_t:.4f} s':>16} | {f'{tot_t:.2f} s':>12}"
    )
    print(row, flush=True)
    print(dash, flush=True)
    print(f"⏱️ TIMING SUMMARY: {num_samples} samples | Total Time: {tot_t:.2f}s | Avg Time/Sample: {avg_t:.4f}s | Speed: {throughput}", flush=True)
    print(sep + "\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run CHAIR benchmark for OPERA with multi-GPU support & timing")
    parser.add_argument("--model", type=str, choices=["llava", "qwen2vl"], required=True,
                        help="Model to benchmark (llava or qwen2vl)")
    parser.add_argument("--use_opera", action="store_true",
                        help="Enable OPERA hallucination mitigation")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of images to evaluate (default: 10 for latency testing)")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Max new tokens to generate (standard: 128 for CHAIR)")
    parser.add_argument("--prompt", type=str, default="Describe this image.",
                        help="Prompt for caption generation")
    parser.add_argument("--coco_dir", type=str, default=None,
                        help="Path to COCO val2014 images directory")
    parser.add_argument("--manifest_path", type=str, default=None,
                        help="Path to pre-computed CHAIR samples JSON")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to data paths YAML config")
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp16", "auto"], default="bf16",
                        help="Precision for model weights (BF16 with FP16 fallback)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device or device_map strategy (default: 'auto')")
    parser.add_argument("--seed", type=int, default=2027,
                        help="Random seed for sampling (default: 2027)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Custom root directory to store benchmark results")

    args = parser.parse_args()

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

    # 2. Detect COCO directory & CHAIR manifest
    coco_image_dir = auto_detect_coco_dir(
        override_path=args.coco_dir,
        config_path=data_paths.get("coco_val2014_images")
    )

    manifest_file = args.manifest_path or data_paths.get("chair_manifest", None)
    if manifest_file and not os.path.isabs(manifest_file):
        manifest_file = os.path.join(EXPERIMENTS_DIR, manifest_file)

    cache_path = data_paths.get("chair_eval_cache", None)
    if cache_path and not os.path.isabs(cache_path):
        cache_path = os.path.join(EXPERIMENTS_DIR, cache_path)

    # 3. Load samples (subsampled to --num_samples, default 10)
    samples = load_chair_samples(
        image_dir=coco_image_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        manifest_path=manifest_file
    )

    # 4. Resolve precision & device
    dtype = resolve_dtype(args.dtype)
    print(f"\n[Environment Setup]", flush=True)
    print(f"  PyTorch Version: {torch.__version__}", flush=True)
    print(f"  CUDA Available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}", flush=True)
    print(f"  Resolved Precision: {dtype}", flush=True)
    print(f"  Device Map: {args.device}", flush=True)
    print(f"  Samples to evaluate: {len(samples)} (seed: {args.seed})", flush=True)

    # 5. Output directory setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "opera" if args.use_opera else "baseline"
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(EXPERIMENTS_DIR, "results", f"{args.model}_chair_{mode}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    raw_outputs_file = os.path.join(output_dir, "raw_outputs.jsonl")
    metrics_file = os.path.join(output_dir, "metrics.json")
    summary_file = os.path.join(output_dir, "summary_metrics.json")

    # 6. Load Model
    print(f"\n[Model Initialization] Loading model '{args.model}'...", flush=True)
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

    # Save run config
    run_config = {
        "benchmark": "chair",
        "model": args.model,
        "model_ckpt": model_ckpt,
        "use_opera": args.use_opera,
        "num_samples": len(samples),
        "max_new_tokens": args.max_new_tokens,
        "prompt": args.prompt,
        "do_sample": False,
        "temperature": 0.0,
        "seed": args.seed,
        "dtype": str(dtype),
        "device_map": args.device,
        "coco_image_dir": coco_image_dir,
        "opera_config": opera_config if args.use_opera else {},
        "timestamp": timestamp,
    }
    with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=4)

    # 7. Generate captions & measure precise GPU latency
    sample_latencies = []
    results = []

    print(f"\n---> Starting CHAIR Caption Generation ({len(samples)} samples) | Model: {args.model.upper()} | OPERA: {args.use_opera}", flush=True)
    print(f"     Prompt: '{args.prompt}' | max_new_tokens: {args.max_new_tokens} | Greedy Decoding", flush=True)

    with open(raw_outputs_file, "w", encoding="utf-8") as f_out:
        pbar = tqdm(samples, desc=f"CHAIR [{args.model} - {mode}]")
        for idx, item in enumerate(pbar):
            img_id = item["image_id"]
            img_name = item["file_name"]

            img_path = find_image_file(coco_image_dir, img_name)
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Warning: Failed to load image {img_path}: {e}", flush=True)
                continue

            # Precise CUDA inference timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            if args.use_opera:
                pred = model.generate_with_opera(
                    args.prompt,
                    image,
                    opera_config,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=0.0
                )
            else:
                pred = model.generate_baseline(
                    args.prompt,
                    image,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=0.0
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_end = time.perf_counter()

            latency = t_end - t_start
            sample_latencies.append(latency)

            record = {
                "image_id": img_id,
                "file_name": img_name,
                "prompt": args.prompt,
                "caption": pred,
                "latency_s": round(latency, 4)
            }
            results.append(record)
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()

    total_time = sum(sample_latencies)
    avg_time = (total_time / len(sample_latencies)) if sample_latencies else 0.0

    print(f"\nCaption generation completed for {len(sample_latencies)} images.", flush=True)
    print(f"⏱️ Total Inference Time: {total_time:.2f}s | Average Time / Sample: {avg_time:.4f}s", flush=True)

    # 8. Evaluate CHAIR metrics
    print(f"\n[CHAIR Evaluation] Computing CHAIR metrics (CHAIRs, CHAIRi, Recall, Caption Length)...", flush=True)
    try:
        metrics = evaluate_chair(
            results_file=raw_outputs_file,
            cache_path=cache_path,
            output_metrics_file=metrics_file,
            save_details_file=os.path.join(output_dir, "chair_details.json")
        )
    except Exception as e:
        print(f"[CHAIR Evaluation Notice] Evaluator error: {e}. Fallback to basic metrics logging.", flush=True)
        metrics = {"total_evaluated": len(sample_latencies)}

    # Attach timing metrics
    metrics["avg_time_per_sample_s"] = round(avg_time, 4)
    metrics["total_inference_time_s"] = round(total_time, 4)
    metrics["num_evaluated"] = len(sample_latencies)
    metrics["samples_tested"] = len(samples)

    # Save final metrics & summary
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    summary_data = {
        "model": args.model,
        "mode": mode,
        "metrics": metrics,
        "overall": {
            "total_samples": len(sample_latencies),
            "total_inference_time_s": round(total_time, 4),
            "overall_avg_time_per_sample_s": round(avg_time, 4)
        }
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)

    print_chair_summary_table(metrics, model_name=args.model, mode=mode, num_samples=len(sample_latencies))
    print(f"All CHAIR benchmarks finished successfully! Results saved in: {output_dir}\n", flush=True)


if __name__ == "__main__":
    main()
