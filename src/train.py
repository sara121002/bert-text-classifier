import argparse, os
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, TrainingArguments, Trainer)
import numpy as np
import evaluate

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="bert-base-uncased")
    p.add_argument("--dataset", default="ag_news", choices=["ag_news","sst2"])
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--out_dir", default="models/bert-agnews")
    # 1) add to get_args()
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()

def main():
    args = get_args()
    # 1) data
    if args.dataset == "ag_news":
        ds = load_dataset("ag_news")
        num_labels = 4
    else:  # sst2 (GLUE)
        ds = load_dataset("glue", "sst2")
        num_labels = 2

    # 2) tokenizer + preprocess
    tok = AutoTokenizer.from_pretrained(args.model_name)
    def tok_fn(batch): return tok(batch["text"] if "text" in batch else batch["sentence"],
                                  truncation=True)
    ds_enc = ds.map(tok_fn, batched=True)
    ds_enc = ds_enc.rename_column("label","labels") if "label" in ds_enc["train"].features else ds_enc

      # 2.5) optional: limit dataset size for fast tests
    from datasets import DatasetDict
    ds_enc = ds_enc.shuffle(seed=args.seed)

    def maybe_slice(split_name, k):
        if k is None:
            return ds_enc[split_name]
        n = min(k, len(ds_enc[split_name]))
        return ds_enc[split_name].select(range(n))

    train_ds = maybe_slice("train", args.max_train_samples)
    eval_split = "test" if "test" in ds_enc else "validation"
    eval_ds = maybe_slice(eval_split, args.max_eval_samples)

    # 3) model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_labels
    )

    # 4) metrics
    acc = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
            "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
        }

    # 5) trainer
    collator = DataCollatorWithPadding(tokenizer=tok)
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # 6) save model + tokenizer
    os.makedirs(args.out_dir, exist_ok=True)
    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)

if __name__ == "__main__":
    main()
