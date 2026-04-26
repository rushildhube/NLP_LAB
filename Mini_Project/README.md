# Fine-Tuned Transformer for Abstractive Summarization

## 📋 Problem Statement
Fine-tune a pre-trained transformer model for **abstractive summarization** on a relevant dataset.

## ✅ Solution Overview

This project implements a complete pipeline to fine-tune **BART (facebook/bart-large-cnn)**, a pre-trained sequence-to-sequence transformer, on the **CNN/DailyMail** dataset for abstractive text summarization.

### Architecture & Components

```
Mini_Project/
├── train.py              # Fine-tuning script with early stopping
├── evaluate_model.py     # Comprehensive evaluation with multiple metrics
├── app.py                # Streamlit web application for inference
├── common.py             # Shared utilities (model loading, generation)
└── README.md             # This file
```

---

## 🔧 Setup & Installation

### Requirements
All dependencies are in `requirements.txt`. Install them:

```bash
cd "d:\NLP LAB"
pip install -r requirements.txt
```

### Key Dependencies
- **transformers**: Pre-trained BART model
- **datasets**: CNN/DailyMail dataset loading
- **torch**: Deep learning framework
- **evaluate**: ROUGE, BLEU, METEOR metrics
- **streamlit**: Web UI for inference

---

## 🚀 How to Use

### 1️⃣ Fine-Tune the Model
```bash
python Mini_Project/train.py
```

**What it does:**
- Loads CNN/DailyMail dataset (3,000 training samples, 800 validation samples)
- Fine-tunes BART on summarization task
- Implements early stopping (stops if no improvement for 2 epochs)
- Saves best model to `Mini_Project/model/`
- Saves training config to `Mini_Project/model/training_config.json`

**Training Configuration:**
- **Base Model**: facebook/bart-large-cnn (pre-trained)
- **Learning Rate**: 5e-5 (typical for fine-tuning)
- **Batch Size**: 2 (low-memory systems)
- **Epochs**: 3 (with early stopping)
- **Warmup Steps**: 500
- **Gradient Accumulation**: 2 steps
- **Evaluation**: Every 100 steps
- **Mixed Precision**: Auto-enabled on GPU

**Expected Output:**
```
✅ Training complete. Model saved in Mini_Project/model/
```

---

### 2️⃣ Evaluate the Fine-Tuned Model
```bash
python Mini_Project/evaluate_model.py
```

**What it does:**
- Loads 200 test samples from CNN/DailyMail
- Generates abstractive summaries using the fine-tuned model
- Computes multiple evaluation metrics:
  - **ROUGE-1, ROUGE-2, ROUGE-L**: Overlap-based metrics (standard for summarization)
  - **BLEU**: Machine translation metric (if available)
  - **METEOR**: Semantic similarity metric (if available)
- Performs error analysis (shows worst 3 predictions)
- Saves detailed results to `evaluation_results.json`

**Expected Metrics:**
- ROUGE-1: 0.35-0.45 (after fine-tuning)
- ROUGE-2: 0.15-0.25
- ROUGE-L: 0.32-0.42

**Output Includes:**
- Per-metric scores
- Average generation time per sample
- Top 3 worst predictions with reference vs. model output
- Sample-wise ROUGE scores (for error analysis)

---

### 3️⃣ Run Interactive Web App
```bash
streamlit run Mini_Project/app.py
```

**Features:**
- Enter any article text
- Click "Summarize" to generate abstractive summary
- Model loads once and is cached for fast inference
- Runs on `http://localhost:8501`

---

## 📊 Project Details

### Model Architecture
- **Type**: Sequence-to-Sequence (Encoder-Decoder)
- **Base Model**: BART (Bidirectional Auto-Regressive Transformer)
- **Pre-training**: Trained on large text corpus with denoising objectives
- **Fine-tuning Task**: Abstractive summarization (article → summary)

### Dataset
- **Name**: CNN/DailyMail
- **Size Used**: 3,000 training + 800 validation + 200 test samples
- **Features**: 
  - `article`: Full news article (512 tokens max)
  - `highlights`: Reference summary (128 tokens max)
- **Domain**: News articles and their reference summaries
- **Rationale**: Standard benchmark for summarization research

### Key Improvements Over Baseline
1. **Early Stopping**: Prevents overfitting
2. **Learning Rate Warmup**: Stabilizes fine-tuning
3. **Gradient Accumulation**: Better gradient estimates on low-memory systems
4. **Frequent Evaluation**: Catches convergence issues early
5. **Comprehensive Metrics**: ROUGE + BLEU + METEOR for thorough evaluation
6. **Error Analysis**: Identifies failure cases for debugging

---

## 📈 Expected Results

After running the full pipeline:

### Training Metrics
- Train loss: 2.0 → 1.5 (across 3 epochs)
- Validation loss: 2.2 → 1.6 (with early stopping)
- Training time: ~2-4 hours on GPU (longer on CPU)

### Evaluation Metrics
```
ROUGE Scores:
  rouge1: 0.38
  rouge2: 0.18
  rougeL: 0.35

Generation Time: ~0.5s per article
```

### Key Outputs
```
Mini_Project/
├── model/
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── tokenizer.json
│   └── training_config.json
├── results/
│   ├── trainer_state.json
│   └── checkpoint-*/
└── evaluation_results.json
```

---

## 🔍 File Descriptions

### `train.py`
Implements the fine-tuning pipeline:
- Dataset loading and preprocessing
- Model and tokenizer initialization
- Training loop with Trainer API
- Early stopping callback
- Model checkpointing
- Results saving

### `evaluate_model.py`
Comprehensive evaluation script:
- Test set generation with progress tracking
- Multi-metric computation
- Error analysis with worst predictions
- Performance timing
- Results persistence as JSON

### `app.py`
Streamlit web application:
- User-friendly text input
- Real-time summary generation
- Model caching for performance
- Clean UI with Markdown output

### `common.py`
Shared utilities used by all scripts:
- Centralized model loading with device management
- Unified summary generation function
- Consistent path management
- Device detection (GPU/CPU)

---

## 💡 Customization Options

### Change Dataset Size
Edit `train.py` line 31-32:
```python
train_data = dataset["train"].select(range(5000))  # Larger dataset
val_data = dataset["validation"].select(range(1000))
```

### Adjust Training Hyperparameters
Edit `train.py` lines 73-91:
```python
learning_rate=1e-4,           # Higher LR for larger datasets
warmup_steps=1000,            # More warmup
num_train_epochs=5,           # More epochs
per_device_train_batch_size=4 # Larger batch if GPU available
```

### Use Different Base Model
Edit `train.py` line 46:
```python
model_name = "facebook/bart-base"  # Smaller, faster
model_name = "t5-base"             # Alternative: T5 model
model_name = "pegasus-arxiv"       # Alternative: PEGASUS
```

### Change Summary Length
Edit `common.py` line 43:
```python
max_summary_length: int = 150,  # Longer summaries
```

---

## ⚡ Performance Tips

### For GPU Systems
- Increase batch size: `per_device_train_batch_size=8`
- Enable mixed precision: `fp16=True`
- Use more samples: `train_data = dataset["train"].select(range(10000))`

### For CPU Systems
- Reduce batch size: `per_device_train_batch_size=1`
- Reduce dataset: `train_data = dataset["train"].select(range(1000))`
- Disable mixed precision: `fp16=False` (already auto-disabled)

### For Memory Constraints
- Reduce sequence lengths in preprocessing
- Use gradient accumulation (already set to 2)
- Enable disk-offloading in Trainer args

---

## 📚 References

### Model
- BART Paper: [Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461)
- Hugging Face Docs: https://huggingface.co/facebook/bart-large-cnn

### Dataset
- CNN/DailyMail: [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)

### Metrics
- ROUGE: [ROUGE: A Package for Automatic Evaluation of Summaries](https://arxiv.org/abs/W04-1013)
- BLEU: [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/)

---

## ✨ Requirements Met

✅ **Fine-tune a pre-trained transformer**: BART (facebook/bart-large-cnn)  
✅ **For summarization task**: Abstractive text-to-summary generation  
✅ **On a relevant dataset**: CNN/DailyMail (standard summarization benchmark)  
✅ **Complete pipeline**: Training → Evaluation → Inference  
✅ **Production-ready**: Error handling, logging, metrics, caching  

---

## 🐛 Troubleshooting

### "Model directory not found"
Run `train.py` first to create the fine-tuned model.

### "Out of memory" error
Reduce batch size or dataset size in `train.py`.

### Slow generation on CPU
Use a smaller model: `facebook/bart-base` instead of `facebook/bart-large-cnn`.

### Missing evaluation metrics
Some metrics may fail to download. The script gracefully continues with ROUGE.

---

## 📝 License
This project uses open-source models and datasets. See respective licenses for BART, CNN/DailyMail, and Hugging Face Transformers.

