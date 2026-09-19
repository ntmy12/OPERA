import json
import argparse

def evaluate(results_path):
    with open(results_path, 'r') as f:
        results = json.load(f)

    tp, tn, fp, fn = 0, 0, 0, 0

    for item in results:
        gt = item['gt'].strip().lower()
        pred = item['pred'].strip().lower()
        
        # Sạch hóa pred
        if 'yes' in pred:
            pred = 'yes'
        elif 'no' in pred:
            pred = 'no'

        if gt == 'yes' and pred == 'yes':
            tp += 1
        elif gt == 'no' and pred == 'no':
            tn += 1
        elif gt == 'no' and pred == 'yes':
            fp += 1
        elif gt == 'yes' and pred == 'no':
            fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    yes_ratio = (tp + fp) / total if total > 0 else 0

    print("="*30)
    print("BEAF EVALUATION METRICS")
    print("="*30)
    print(f"Total Samples: {total}")
    print(f"Accuracy:    {accuracy*100:.2f}%")
    print(f"Precision:   {precision*100:.2f}%")
    print(f"Recall:      {recall*100:.2f}%")
    print(f"F1-Score:    {f1*100:.2f}%")
    print(f"Yes Ratio:   {yes_ratio*100:.2f}%")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    args = parser.parse_args()
    evaluate(args.results)
