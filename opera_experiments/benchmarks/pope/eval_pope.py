import json
import argparse
import os

def parse_pred(pred_text):
    """
    Parse model prediction string into 'yes', 'no', or 'unknown'.
    Handles common prefixes and variations.
    """
    if not isinstance(pred_text, str):
        return "unknown"
    text = pred_text.lower().strip()
    # Check startswith first for standard responses
    if text.startswith("yes"):
        return "yes"
    elif text.startswith("no"):
        return "no"
    
    # Check within first 5 tokens for conversational responses
    tokens = text.replace(".", " ").replace(",", " ").replace("!", " ").split()[:5]
    if "yes" in tokens and "no" not in tokens:
        return "yes"
    elif "no" in tokens and "yes" not in tokens:
        return "no"
    return "unknown"

def extract_prediction(item: dict) -> str:
    """Extract model prediction checking 'answer', 'pred', 'text' keys."""
    for key in ["answer", "pred", "text", "model_prediction"]:
        if key in item and item[key] is not None:
            return str(item[key])
    return ""

def eval_pope(results_file: str):
    """
    Evaluate POPE benchmark predictions and return metrics dict.
    """
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
        
    TP = TN = FP = FN = 0
    unknowns = 0
    yes_count = 0
    
    for item in data:
        label = item.get('label', '').lower().strip()
        raw_pred = extract_prediction(item)
        pred = parse_pred(raw_pred)
        
        if pred == "yes":
            yes_count += 1
            
        if pred == "unknown":
            unknowns += 1
            # Treat unknown as false prediction for benchmark rigor
            if label == "yes":
                FN += 1
            else:
                FP += 1
            continue
            
        if label == "yes" and pred == "yes":
            TP += 1
        elif label == "no" and pred == "no":
            TN += 1
        elif label == "yes" and pred == "no":
            FN += 1
        elif label == "no" and pred == "yes":
            FP += 1
        
    total = len(data)
    accuracy = (TP + TN) / total if total > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    yes_ratio = yes_count / total if total > 0 else 0.0
    
    metrics = {
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1": round(f1 * 100, 2),
        "Yes_ratio": round(yes_ratio * 100, 2),
        "Total": total,
        "Unknown_answers": unknowns,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
    }
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate POPE predictions")
    parser.add_argument("--results_dir", type=str, default=None, help="Directory containing raw_outputs.jsonl")
    parser.add_argument("--results_file", type=str, default=None, help="Direct path to raw_outputs.jsonl")
    args = parser.parse_args()
    
    if args.results_file:
        res_file = args.results_file
        save_dir = os.path.dirname(res_file)
    elif args.results_dir:
        res_file = os.path.join(args.results_dir, "raw_outputs.jsonl")
        save_dir = args.results_dir
    else:
        raise ValueError("Must provide either --results_dir or --results_file")
        
    metrics = eval_pope(res_file)
    
    print(f"\nPOPE Metrics for {res_file}:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
        
    if save_dir:
        metrics_save_path = os.path.join(save_dir, "metrics.json")
        with open(metrics_save_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {metrics_save_path}\n")
