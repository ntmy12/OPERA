import json
import argparse
import os

def parse_pred(pred_text):
    # Robust parsing as requested
    text = pred_text.lower().strip()
    if text.startswith("yes") or "yes" in text.split()[:5]:
        return "yes"
    elif text.startswith("no") or "no" in text.split()[:5]:
        return "no"
    return "unknown"

def eval_pope(results_file):
    with open(results_file, 'r') as f:
        data = [json.loads(line) for line in f]
        
    TP = TN = FP = FN = 0
    unknowns = 0
    
    for item in data:
        label = item['label'].lower().strip()
        pred = parse_pred(item['pred'])
        
        if pred == "unknown":
            unknowns += 1
            # Treat unknown as false prediction for safety
            if label == "yes": FN += 1
            else: FP += 1
            continue
            
        if label == "yes" and pred == "yes": TP += 1
        elif label == "no" and pred == "no": TN += 1
        elif label == "yes" and pred == "no": FN += 1
        elif label == "no" and pred == "yes": FP += 1
        
    accuracy = (TP + TN) / len(data) if data else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1": round(f1 * 100, 2),
        "Unknown_answers": unknowns
    }
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()
    
    results_file = os.path.join(args.results_dir, "raw_outputs.jsonl")
    metrics = eval_pope(results_file)
    
    print(f"Metrics for {args.results_dir}:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
        
    with open(os.path.join(args.results_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
