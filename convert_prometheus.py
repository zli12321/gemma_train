from datasets import load_dataset
import json

def main():
    print("Loading Prometheus Feedback Collection from HuggingFace...")
    ds = load_dataset("prometheus-eval/Feedback-Collection", split="train")
    print(f"Loaded {len(ds)} examples")

    sft_data = []
    for example in ds:
        sft_data.append({
            "instruction": example["instruction"],
            "input": example["input"],
            "output": example["output"],
        })

    output_path = "data/prometheus_feedback.json"
    print(f"Writing {len(sft_data)} examples to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    print("Done! Now register the dataset in data/dataset_info.json")

if __name__ == "__main__":
    main()
