#!/usr/bin/env python
# =========================
# QUICK START GUIDE
# =========================
"""
This script helps you get started with fine-tuning BART for summarization.
It verifies all dependencies are installed and shows you the next steps.
"""

import sys
from pathlib import Path

def check_dependencies():
    """Verify all required packages are installed."""
    print("🔍 Checking dependencies...\n")
    
    required_packages = {
        "torch": "PyTorch (deep learning framework)",
        "transformers": "Transformers (pre-trained models)",
        "datasets": "Datasets (data loading)",
        "evaluate": "Evaluate (metrics)",
        "streamlit": "Streamlit (web app)",
        "tqdm": "TQDM (progress bars)",
        "numpy": "NumPy (numerical computing)",
    }
    
    missing = []
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {package:20s} - {description}")
        except ImportError:
            print(f"❌ {package:20s} - {description}")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"\nInstall them with:")
        print(f"  pip install -r requirements.txt")
        return False
    else:
        print(f"\n✅ All dependencies installed!")
        return True

def show_project_structure():
    """Display the project structure."""
    print("\n📁 Project Structure:")
    print("""
    Mini_Project/
    ├── train.py              # Fine-tune BART on CNN/DailyMail
    ├── evaluate_model.py     # Evaluate with ROUGE, BLEU, METEOR
    ├── app.py                # Streamlit web app for inference
    ├── common.py             # Shared utilities
    └── README.md             # Full documentation
    """)

def show_workflow():
    """Display the complete workflow."""
    print("\n🚀 Complete Workflow:")
    print("""
    Step 1: Fine-tune the model
    ─────────────────────────────────
    $ python Mini_Project/train.py
    
    • Downloads CNN/DailyMail dataset (5K train, 1K val samples)
    • Fine-tunes BART with learning rate warmup
    • Implements early stopping to prevent overfitting
    • Saves model to Mini_Project/model/
    
    ⏱️  Expected time: 2-4 hours on GPU, longer on CPU
    
    
    Step 2: Evaluate the model
    ─────────────────────────────────
    $ python Mini_Project/evaluate_model.py
    
    • Generates summaries on 200 test samples
    • Computes ROUGE, BLEU, METEOR metrics
    • Analyzes worst predictions
    • Saves results to evaluation_results.json
    
    ⏱️  Expected time: 2-5 minutes
    
    
    Step 3: Run the web app
    ─────────────────────────────────
    $ streamlit run Mini_Project/app.py
    
    • Opens interactive web interface
    • Enter any article text
    • Get abstractive summaries in real-time
    • Model cached for fast inference
    
    🌐 Opens at: http://localhost:8501
    """)

def show_expected_results():
    """Show expected performance metrics."""
    print("\n📊 Expected Results:")
    print("""
    After fine-tuning and evaluation, you should see:
    
    ROUGE Scores (on 200 test samples):
    • ROUGE-1: 0.35-0.45  (word overlap)
    • ROUGE-2: 0.15-0.25  (bigram overlap)
    • ROUGE-L: 0.32-0.42  (longest common subsequence)
    
    Generation Speed:
    • ~0.5-1.0s per article (GPU)
    • ~2-5s per article (CPU)
    
    Output Files:
    • Mini_Project/model/              (fine-tuned model)
    • Mini_Project/results/            (training checkpoints)
    • evaluation_results.json          (detailed metrics)
    • training_config.json             (training hyperparameters)
    """)

def show_tips():
    """Show helpful tips."""
    print("\n💡 Tips & Troubleshooting:")
    print("""
    Running on GPU (NVIDIA):
    • Much faster (2-4 hours vs 12+ hours on CPU)
    • Check: python -c "import torch; print(torch.cuda.is_available())"
    
    Running on CPU:
    • Works fine, just slower
    • If memory issues: reduce batch size in train.py
    
    First time takes longer:
    • Models and datasets download first time only
    • Subsequent runs use cached data
    
    Customization:
    • Edit train.py to adjust dataset size, learning rate, epochs
    • Edit common.py to change summary length
    • Edit app.py to customize the UI
    
    Full documentation in: README.md
    """)

def main():
    """Run the quick start guide."""
    print("=" * 70)
    print("🎯 FINE-TUNED TRANSFORMER FOR SUMMARIZATION - QUICK START")
    print("=" * 70)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again.")
        sys.exit(1)
    
    # Show project structure
    show_project_structure()
    
    # Show workflow
    show_workflow()
    
    # Show expected results
    show_expected_results()
    
    # Show tips
    show_tips()
    
    print("\n" + "=" * 70)
    print("✨ Ready to start! Run the commands above to begin fine-tuning.")
    print("=" * 70)
    print("\n📖 For more details, see: Mini_Project/README.md\n")

if __name__ == "__main__":
    main()
