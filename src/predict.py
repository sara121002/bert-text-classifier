import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--text", required=True)
    return p.parse_args()

def main():
    args = get_args()
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    inputs = tok(args.text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze()
        pred = int(torch.argmax(probs))

    print(f"Text: {args.text}")
    print(f"Pred class: {pred}")
    print(f"Probs: {probs.tolist()}")

if __name__ == "__main__":
    main()
