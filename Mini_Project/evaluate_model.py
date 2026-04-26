# =========================
# EVALUATION SCRIPT
# =========================

import logging
import json
import time
from tqdm import tqdm
from datasets import load_dataset
import evaluate
import numpy as np

from common import generate_summary, load_model_and_tokenizer, MODEL_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting evaluation...")
    start_time = time.time()
    
    # Load small test dataset
    logger.info("Loading test dataset...")
    try:
        dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:200]")
        logger.info(f"Loaded {len(dataset)} test samples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # Load trained model
    logger.info("Loading model...")
    try:
        tokenizer, model, device = load_model_and_tokenizer()
        logger.info(f"Model loaded on device: {device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Load metrics
    logger.info("Loading evaluation metrics...")
    try:
        rouge = evaluate.load("rouge")
        bleu = evaluate.load("bleu")
        meteor = evaluate.load("meteor")
        logger.info("Metrics loaded: ROUGE, BLEU, METEOR")
    except Exception as e:
        logger.warning(f"Could not load all metrics: {e}. Using ROUGE only.")
        bleu = None
        meteor = None

    predictions = []
    references = []
    generation_times = []

# -------------------------
# GENERATE SUMMARIES WITH PROGRESS
# -------------------------
    logger.info("Generating summaries...")
    for i, item in enumerate(tqdm(dataset, desc="Generating predictions")):
        try:
            gen_start = time.time()
            pred = generate_summary(
                text=item["article"],
                tokenizer=tokenizer,
                model=model,
                device=device,
            )
            gen_time = time.time() - gen_start
            generation_times.append(gen_time)
            
            predictions.append(pred)
            references.append(item["highlights"])
        except Exception as e:
            logger.warning(f"Failed to generate summary for sample {i}: {e}")
            continue

    logger.info(f"Generated {len(predictions)} summaries")

# -------------------------
# CALCULATE MULTIPLE METRICS
# -------------------------
    logger.info("Computing metrics...")
    metrics_results = {}
    
    # ROUGE
    try:
        rouge_results = rouge.compute(
            predictions=predictions,
            references=references
        )
        metrics_results["ROUGE"] = rouge_results
        logger.info("✅ ROUGE computed")
    except Exception as e:
        logger.error(f"ROUGE computation failed: {e}")
    
    # BLEU
    if bleu:
        try:
            bleu_results = bleu.compute(
                predictions=predictions,
                references=[[ref] for ref in references]
            )
            metrics_results["BLEU"] = bleu_results
            logger.info("✅ BLEU computed")
        except Exception as e:
            logger.warning(f"BLEU computation failed: {e}")
    
    # METEOR
    if meteor:
        try:
            meteor_results = meteor.compute(
                predictions=predictions,
                references=references
            )
            metrics_results["METEOR"] = meteor_results
            logger.info("✅ METEOR computed")
        except Exception as e:
            logger.warning(f"METEOR computation failed: {e}")

# -------------------------
# ERROR ANALYSIS
# -------------------------
    logger.info("Analyzing errors...")
    
    # Calculate ROUGE-1 per sample for error analysis
    rouge_1_scores = []
    for pred, ref in zip(predictions, references):
        try:
            score = rouge.compute(
                predictions=[pred],
                references=[ref]
            )["rouge1"]
            rouge_1_scores.append(score)
        except:
            rouge_1_scores.append(0.0)
    
    # Find worst predictions
    worst_indices = np.argsort(rouge_1_scores)[:3]  # Top 3 worst
    
# -------------------------
# PRINT RESULTS SUMMARY
# -------------------------
    print("\n" + "="*70)
    print("EVALUATION RESULTS SUMMARY")
    print("="*70)
    
    print(f"\nDataset: {len(predictions)} samples")
    print(f"Avg generation time: {np.mean(generation_times):.2f}s per sample")
    
    print("\n" + "-"*70)
    print("ROUGE SCORES:")
    print("-"*70)
    if "ROUGE" in metrics_results:
        for key, value in metrics_results["ROUGE"].items():
            print(f"  {key}: {value:.4f}")
    
    if "BLEU" in metrics_results:
        print(f"\nBLEU Score: {metrics_results['BLEU']['bleu']:.4f}")
    
    if "METEOR" in metrics_results:
        print(f"METEOR Score: {metrics_results['METEOR']['meteor']:.4f}")
    
    print("\n" + "-"*70)
    print("TOP 3 WORST PREDICTIONS (by ROUGE-1):")
    print("-"*70)
    for rank, idx in enumerate(worst_indices, 1):
        print(f"\n{rank}. Sample #{idx} (ROUGE-1: {rouge_1_scores[idx]:.4f})")
        print(f"   Article: {dataset[idx]['article'][:100]}...")
        print(f"   Reference: {references[idx][:100]}...")
        print(f"   Prediction: {predictions[idx][:100]}...")
    
    # Save detailed results
    print("\n" + "-"*70)
    results_dict = {
        "metrics": metrics_results,
        "avg_generation_time": float(np.mean(generation_times)),
        "total_samples": len(predictions),
        "sample_wise_rouge1": [float(s) for s in rouge_1_scores],
    }
    
    results_file = MODEL_DIR.parent / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(results_dict, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")
    
    print("="*70)
    elapsed = time.time() - start_time
    logger.info(f"Evaluation complete in {elapsed:.2f}s")


if __name__ == "__main__":
    main()