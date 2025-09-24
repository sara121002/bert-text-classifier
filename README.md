# BERT Text Classifier (AG News / SST-2)

**Stack:** Hugging Face Transformers, PyTorch, Datasets, Accelerate  
**Goal:** Fine-tune BERT for news/sentiment classification.  
**Key Metrics:** Accuracy, Weighted F1

## Quickstart

### Option 1: Using Conda (if installed)

```bash
conda create -n bertcls python=3.10 -y
conda activate bertcls
pip install -r requirements.txt
```

### Option 2: Using Python venv (recommended if conda not available)

```bash
python -m venv bertcls
bertcls\Scripts\activate  # On Windows
# source bertcls/bin/activate  # On Linux/Mac
pip install -r requirements.txt
```

### Training and Inference

```bash
# Train
python src/train.py --model_name bert-base-uncased --dataset ag_news --epochs 2 --batch_size 16

# Inference
python src/predict.py --model_dir models/bert-agnews --text "Apple unveils new AI features"
```
