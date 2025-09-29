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

## Example Predictions

After training the model, you can run inference with:

```bash
python src/predict.py --model_dir models/bert-agnews --text "Apple unveils new AI features for iPhone"
```

Expected Output

Text: Apple unveils new AI features for iPhone

Pred class: 3

Probs: [0.07217928767204285, 0.07576016336679459, 0.12794804573059082, 0.7241125702857971]

Interpretation: The model assigns 72% confidence to the Sci/Tech class.

These predictions match the 4 classes in the AG News dataset:
0 = World,
1 = Sports,
2 = Business,
3 = Sci/Tech,

