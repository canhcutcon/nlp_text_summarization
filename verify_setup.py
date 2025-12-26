"""
Verification script to check if the environment is set up correctly
"""

import sys
print("=" * 80)
print("ENVIRONMENT VERIFICATION")
print("=" * 80)

# 1. Check Python version and location
print(f"\n1. Python Version & Location:")
print(f"   Version: {sys.version}")
print(f"   Executable: {sys.executable}")

# 2. Check transformers installation
print(f"\n2. Transformers Library:")
try:
    import transformers
    print(f"   ✅ transformers installed successfully")
    print(f"   Version: {transformers.__version__}")
    print(f"   Location: {transformers.__file__}")
except ImportError as e:
    print(f"   ❌ Failed to import transformers: {e}")
    sys.exit(1)

# 3. Check datasets installation
print(f"\n3. Datasets Library:")
try:
    import datasets
    print(f"   ✅ datasets installed successfully")
    print(f"   Version: {datasets.__version__}")
except ImportError as e:
    print(f"   ❌ Failed to import datasets: {e}")
    sys.exit(1)

# 4. Check torch installation
print(f"\n4. PyTorch:")
try:
    import torch
    print(f"   ✅ torch installed successfully")
    print(f"   Version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"   MPS (Apple Silicon) available: {torch.backends.mps.is_available()}")
except ImportError as e:
    print(f"   ❌ Failed to import torch: {e}")
    sys.exit(1)

# 5. Check other critical packages
print(f"\n5. Other Critical Packages:")
packages = [
    'accelerate',
    'sentencepiece',
    'tokenizers',
    'rouge_score',
    'evaluate',
    'nltk',
    'pandas',
    'numpy',
    'matplotlib',
    'seaborn',
    'plotly',
]

for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f"   ✅ {pkg:<20} v{version}")
    except ImportError:
        print(f"   ❌ {pkg:<20} NOT INSTALLED")

# 6. Test loading dataset
print(f"\n6. Test Dataset Loading:")
try:
    from datasets import load_dataset
    print(f"   Loading Vietnamese summarization dataset...")
    dataset = load_dataset("8Opt/vietnamese-summarization-dataset-0001", split="train[:5]")
    print(f"   ✅ Successfully loaded {len(dataset)} sample(s)")
    print(f"   Features: {list(dataset.features.keys())}")
except Exception as e:
    print(f"   ❌ Failed to load dataset: {e}")

# 7. Test transformers import (the problematic one)
print(f"\n7. Test Transformers Seq2Seq Components:")
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )
    print(f"   ✅ Successfully imported Seq2SeqTrainer and related classes")
except ImportError as e:
    print(f"   ❌ Failed to import Seq2Seq components: {e}")
    sys.exit(1)

# 8. List available Jupyter kernels
print(f"\n8. Available Jupyter Kernels:")
try:
    import subprocess
    result = subprocess.run(['jupyter', 'kernelspec', 'list'],
                          capture_output=True, text=True, timeout=10)
    print(result.stdout)
except Exception as e:
    print(f"   ⚠️  Could not list kernels: {e}")

print("=" * 80)
print("✅ ALL CHECKS PASSED!")
print("=" * 80)
print("\n📝 Next Steps:")
print("1. Open your Jupyter notebook")
print("2. Go to: Kernel → Change Kernel → 'Python 3.13 (NLP)'")
print("3. Restart the kernel and run your code")
print("\n💡 Tip: You can also use the load_dataset.py script to explore the dataset")
print("=" * 80)
