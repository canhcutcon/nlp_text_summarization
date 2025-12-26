"""
KAGGLE STARTER CODE
Copy and paste these cells into your Kaggle notebook
"""

# ============================================================================
# CELL 1: Setup and Verify Environment
# ============================================================================
print("=" * 80)
print("VIETNAMESE TEXT SUMMARIZATION - KAGGLE SETUP")
print("=" * 80)

import sys
print(f"\nPython version: {sys.version}")

# Check CUDA availability
import torch
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Check transformers version
import transformers
print(f"\nTransformers version: {transformers.__version__}")

# Install/upgrade packages if needed - UPDATED VERSIONS
print("\nInstalling required packages...")
import subprocess
subprocess.run(['pip', 'install', '-q', '--upgrade',
                'transformers>=4.50.0',
                'datasets>=2.14.6',
                'sentencepiece>=0.1.99',
                'rouge-score>=0.1.2',
                'evaluate>=0.4.1',
                'accelerate>=0.26.0'])

print("✅ Setup complete!")
print("\n⚠️  IMPORTANT: After running this cell, please RESTART THE KERNEL!")
print("   Then continue with the remaining cells.")


# ============================================================================
# CELL 2: Load Dataset
# ============================================================================
print("\n" + "=" * 80)
print("LOADING DATASET")
print("=" * 80)

from datasets import load_dataset

# Load Vietnamese summarization dataset from Hugging Face
print("\nLoading Vietnamese Summarization Dataset...")
print("Dataset: 8Opt/vietnamese-summarization-dataset-0001")

dataset = load_dataset("8Opt/vietnamese-summarization-dataset-0001")

print("\n✅ Dataset loaded successfully!")
print(dataset)

# Display statistics
print(f"\nDataset splits:")
for split_name in dataset.keys():
    print(f"  - {split_name}: {len(dataset[split_name]):,} samples")

print(f"\nFeatures: {list(dataset['train'].features.keys())}")

# Show a sample
sample = dataset['train'][0]
print(f"\n📝 Sample Example:")
print(f"Document (first 200 chars): {sample['document'][:200]}...")
print(f"Summary: {sample['summary'][:150]}...")
print(f"Keywords: {sample['keywords']}")


# ============================================================================
# CELL 3: Load Model and Tokenizer
# ============================================================================
print("\n" + "=" * 80)
print("LOADING MODEL")
print("=" * 80)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Choose your model
MODEL_NAME = 'VietAI/vit5-base'  # Best for Vietnamese
# Alternatives:
# MODEL_NAME = 'google/mt5-base'  # Good multilingual option
# MODEL_NAME = 'VietAI/vit5-large'  # Better quality, slower

print(f"\nLoading model: {MODEL_NAME}")
print("This may take a few minutes...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print(f"\n✅ Model loaded successfully!")
print(f"Model parameters: {model.num_parameters():,}")

# Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model device: {device}")


# ============================================================================
# CELL 4: Preprocess Dataset
# ============================================================================
print("\n" + "=" * 80)
print("PREPROCESSING DATASET")
print("=" * 80)

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128

def preprocess_function(examples):
    """Tokenize inputs and targets"""
    # Add "summarize: " prefix for T5-based models
    inputs = [f"summarize: {doc}" for doc in examples['document']]

    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding='max_length'
    )

    # Tokenize targets (summaries)
    labels = tokenizer(
        examples['summary'],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding='max_length'
    )

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

print("\nTokenizing datasets...")
print("This may take several minutes...")

tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset['train'].column_names,
    desc="Tokenizing"
)

print("\n✅ Tokenization complete!")
print(f"Train samples: {len(tokenized_datasets['train']):,}")
print(f"Validation samples: {len(tokenized_datasets['validation']):,}")
print(f"Test samples: {len(tokenized_datasets['test']):,}")


# ============================================================================
# CELL 5: Configure Training
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING CONFIGURATION")
print("=" * 80)

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    # Output
    output_dir='./results',
    overwrite_output_dir=True,

    # Evaluation & Saving - UPDATED for transformers >= 4.50
    eval_strategy='steps',  # Changed from 'evaluation_strategy'
    eval_steps=500,
    save_strategy='steps',
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model='rouge1',
    greater_is_better=True,

    # Training hyperparameters
    learning_rate=5e-5,
    num_train_epochs=3,
    per_device_train_batch_size=8,  # Adjust if OOM
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,  # Effective batch size = 16
    warmup_steps=500,
    weight_decay=0.01,

    # Performance
    fp16=True,  # Mixed precision training
    dataloader_num_workers=2,

    # Logging
    logging_dir='./logs',
    logging_steps=100,
    report_to=['tensorboard'],

    # Generation (for evaluation)
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,
    generation_num_beams=4,

    # Misc
    push_to_hub=False,
    seed=42,
)

print("\n📋 Training Configuration:")
print(f"  Model: {MODEL_NAME}")
print(f"  Device: {device}")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  FP16 (mixed precision): {training_args.fp16}")
print(f"  Max input length: {MAX_INPUT_LENGTH}")
print(f"  Max target length: {MAX_TARGET_LENGTH}")


# ============================================================================
# CELL 6: Set Up Metrics
# ============================================================================
print("\n" + "=" * 80)
print("SETTING UP EVALUATION METRICS")
print("=" * 80)

import numpy as np
import evaluate

# Load ROUGE metric
rouge_metric = evaluate.load('rouge')

def compute_metrics(eval_pred):
    """Compute ROUGE scores for evaluation"""
    predictions, labels = eval_pred

    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in labels (padding token)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute ROUGE scores
    result = rouge_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    # Extract and format results
    return {
        'rouge1': round(result['rouge1'] * 100, 2),
        'rouge2': round(result['rouge2'] * 100, 2),
        'rougeL': round(result['rougeL'] * 100, 2),
    }

print("✅ Metrics configured (ROUGE-1, ROUGE-2, ROUGE-L)")


# ============================================================================
# CELL 7: Initialize Trainer and Start Training
# ============================================================================
print("\n" + "=" * 80)
print("INITIALIZING TRAINER")
print("=" * 80)

from transformers import Seq2SeqTrainer, DataCollatorForSeq2Seq

# Data collator for dynamic padding
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("✅ Trainer initialized")

print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80)
print("\n🚀 Training started...")
print("This will take several hours. You can close your browser - Kaggle will keep running.")
print("=" * 80 + "\n")

# Start training!
train_result = trainer.train()

print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE!")
print("=" * 80)

# Print training summary
print("\n📊 Training Summary:")
print(f"  Total training time: {train_result.metrics['train_runtime']:.2f} seconds")
print(f"  Training samples/second: {train_result.metrics['train_samples_per_second']:.2f}")
print(f"  Final training loss: {train_result.metrics['train_loss']:.4f}")

# Save the model
print("\n💾 Saving model...")
trainer.save_model('./final_model')
tokenizer.save_pretrained('./final_model')
print("✅ Model saved to ./final_model")


# ============================================================================
# CELL 8: Evaluate on Test Set
# ============================================================================
print("\n" + "=" * 80)
print("EVALUATING ON TEST SET")
print("=" * 80)

print("\n🔍 Running evaluation on test set...")
test_results = trainer.evaluate(tokenized_datasets['test'])

print("\n📊 Test Results:")
print("=" * 80)
for key, value in test_results.items():
    if 'rouge' in key:
        print(f"  {key.upper()}: {value:.2f}")

print("=" * 80)


# ============================================================================
# CELL 9: Generate Sample Predictions
# ============================================================================
print("\n" + "=" * 80)
print("SAMPLE PREDICTIONS")
print("=" * 80)

def generate_summary(text, max_length=128, num_beams=4):
    """Generate summary for a given text"""
    inputs = tokenizer(
        f"summarize: {text}",
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        return_tensors='pt'
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=num_beams,
        length_penalty=0.6,
        early_stopping=True,
        no_repeat_ngram_size=3
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# Test with examples
print("\n🔍 Testing with examples from test set:\n")

for i in range(3):
    sample = dataset['test'][i]

    print(f"\n{'=' * 80}")
    print(f"EXAMPLE {i+1}")
    print("=" * 80)

    print(f"\n📄 Document:")
    print(f"{sample['document'][:300]}...")

    print(f"\n✅ Reference Summary:")
    print(f"{sample['summary']}")

    print(f"\n🤖 Generated Summary:")
    predicted = generate_summary(sample['document'])
    print(f"{predicted}")

    print(f"\n🔑 Keywords: {', '.join(sample['keywords'])}")

print("\n" + "=" * 80)


# ============================================================================
# CELL 10: Save Results and Metrics
# ============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

import json

# Save test results
with open('test_results.json', 'w', encoding='utf-8') as f:
    json.dump(test_results, f, indent=2)

# Save training history
training_history = trainer.state.log_history
with open('training_history.json', 'w', encoding='utf-8') as f:
    json.dump(training_history, f, indent=2)

# Create summary report
summary = {
    'model': MODEL_NAME,
    'dataset': '8Opt/vietnamese-summarization-dataset-0001',
    'training_samples': len(tokenized_datasets['train']),
    'validation_samples': len(tokenized_datasets['validation']),
    'test_samples': len(tokenized_datasets['test']),
    'epochs': training_args.num_train_epochs,
    'batch_size': training_args.per_device_train_batch_size,
    'learning_rate': training_args.learning_rate,
    'test_results': {k: v for k, v in test_results.items() if 'rouge' in k},
}

with open('summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n✅ Results saved:")
print("  - test_results.json")
print("  - training_history.json")
print("  - summary.json")
print("  - ./final_model/ (trained model)")

print("\n" + "=" * 80)
print("✅ ALL DONE!")
print("=" * 80)
print("\nYou can download the trained model from the Output tab →")
print("The model is saved in: ./final_model/")
print("\n📊 Final Test Scores:")
for key, value in summary['test_results'].items():
    print(f"  {key.upper()}: {value:.2f}")
print("=" * 80)


# ============================================================================
# OPTIONAL CELL 11: Quick Test Function
# ============================================================================
"""
Use this cell to test the model with your own text:
"""

def test_your_text(text):
    """
    Test the model with custom Vietnamese text

    Usage:
        text = "Paste your Vietnamese article here..."
        summary = test_your_text(text)
    """
    print(f"\n📄 Input Text:")
    print(f"{text[:200]}..." if len(text) > 200 else text)

    summary = generate_summary(text)

    print(f"\n🤖 Generated Summary:")
    print(summary)

    return summary

# Example usage:
# your_text = "Hôm nay, Bộ Y tế công bố thêm 15.527 ca nhiễm COVID-19..."
# test_your_text(your_text)
