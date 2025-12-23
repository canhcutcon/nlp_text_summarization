"""
Vietnamese Text Summarization - Fine-tuning PhoBERT, mT5, ViT5
================================================================
Author: Yang - Master's Research Project
Environment: Kaggle / Google Colab
Dataset: VLSP 2021 Summarization Task

Mục tiêu:
- Fine-tune models cho extractive & abstractive summarization
- So sánh performance giữa PhoBERT, mT5, ViT5
- Implement best practices cho Vietnamese NLP
- Comprehensive evaluation với ROUGE scores
"""

import os
import json
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Frameworks
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup
)

# Evaluation Metrics
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import evaluate

# Utilities
from sklearn.model_selection import train_test_split
import pickle
import gc

print("=" * 80)
print("VIETNAMESE TEXT SUMMARIZATION - TRAINING PIPELINE")
print("=" * 80)

# ============================================================================
# PHẦN 1: LÝ THUYẾT VÀ NGUYÊN TẮC CƠ BẢN
# ============================================================================

"""
1. NGUYÊN TẮC CƠ BẢN CỦA TÓM TẮT VĂN BẢN
=========================================

A. HAI LOẠI TÓM TẮT CHÍNH:

1. EXTRACTIVE SUMMARIZATION (Tóm tắt trích xuất)
   - Chọn những câu quan trọng nhất từ văn bản gốc
   - Không tạo câu mới, chỉ sắp xếp lại câu có sẵn
   - Phù hợp với: PhoBERT, BERT-based models
   
   Ưu điểm:
   ✓ Đảm bảo tính chính xác về ngữ pháp
   ✓ Giữ nguyên thông tin từ nguồn
   ✓ Training nhanh hơn, ít tài nguyên hơn
   
   Nhược điểm:
   ✗ Thiếu tính linh hoạt
   ✗ Có thể không mạch lạc
   ✗ Không tạo được câu mới

2. ABSTRACTIVE SUMMARIZATION (Tóm tắt sinh tạo)
   - Hiểu nội dung và tạo ra câu tóm tắt mới
   - Giống cách con người tóm tắt
   - Phù hợp với: mT5, ViT5, BART
   
   Ưu điểm:
   ✓ Tóm tắt tự nhiên, mạch lạc hơn
   ✓ Có thể paraphrase và tổng hợp thông tin
   ✓ Linh hoạt về độ dài
   
   Nhược điểm:
   ✗ Có thể sinh ra thông tin sai (hallucination)
   ✗ Cần nhiều tài nguyên training hơn
   ✗ Phức tạp hơn trong evaluation

B. CÁC THÀNH PHẦN CHÍNH:

1. Encoder: Đọc và hiểu văn bản đầu vào
2. Decoder: Sinh ra văn bản tóm tắt
3. Attention Mechanism: Tập trung vào phần quan trọng

C. EVALUATION METRICS:

- ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
  * ROUGE-1: Unigram overlap
  * ROUGE-2: Bigram overlap  
  * ROUGE-L: Longest common subsequence
  
- BLEU: Precision-based metric
- BERTScore: Semantic similarity using embeddings
"""

# ============================================================================
# PHẦN 2: SETUP ENVIRONMENT & CONFIGURATION
# ============================================================================

class Config:
    """Configuration cho training pipeline"""
    
    # Device configuration
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_gpu = torch.cuda.device_count()
        
        # Data paths (Kaggle structure)
        self.data_dir = '/kaggle/input/vlsp2021-summarization'
        self.output_dir = '/kaggle/working/models'
        self.log_dir = '/kaggle/working/logs'
        
        # Model configurations
        self.models = {
            'phobert': {
                'name': 'vinai/phobert-base',
                'type': 'extractive',
                'max_length': 512
            },
            'mt5': {
                'name': 'google/mt5-base',
                'type': 'abstractive',
                'max_length': 512,
                'max_target_length': 128
            },
            'vit5': {
                'name': 'VietAI/vit5-base',
                'type': 'abstractive', 
                'max_length': 512,
                'max_target_length': 128
            }
        }
        
        # Training hyperparameters
        self.batch_size = 8  # Giảm nếu out of memory trên Kaggle
        self.learning_rate = 5e-5
        self.num_epochs = 3
        self.warmup_steps = 500
        self.weight_decay = 0.01
        self.gradient_accumulation_steps = 2
        
        # Early stopping
        self.patience = 3
        self.min_delta = 0.001
        
        # Logging
        self.logging_steps = 100
        self.eval_steps = 500
        self.save_steps = 1000
        
        print(f"\n{'='*60}")
        print(f"CONFIGURATION SUMMARY")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Number of GPUs: {self.n_gpu}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Epochs: {self.num_epochs}")
        print(f"{'='*60}\n")

config = Config()

# ============================================================================
# PHẦN 3: DATA LOADING & PREPROCESSING
# ============================================================================

class VLSPDataLoader:
    """
    Load và preprocess VLSP 2021 Summarization dataset
    
    Format expected:
    - article: Văn bản gốc cần tóm tắt
    - summary: Tóm tắt reference (ground truth)
    """
    
    def __init__(self, data_path, model_type='abstractive'):
        self.data_path = data_path
        self.model_type = model_type
        
    def load_data(self):
        """Load dataset từ VLSP format"""
        print("\n[1] Loading VLSP Dataset...")
        
        # VLSP format có thể là JSON hoặc CSV
        if self.data_path.endswith('.json'):
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        elif self.data_path.endswith('.csv'):
            df = pd.read_csv(self.data_path)
        else:
            # Load từ folder chứa files
            df = self._load_from_folder()
        
        print(f"✓ Loaded {len(df)} samples")
        print(f"✓ Columns: {df.columns.tolist()}")
        
        return df
    
    def _load_from_folder(self):
        """Load dataset từ folder structure"""
        articles = []
        summaries = []
        
        # Tìm các file trong folder
        for file in os.listdir(self.data_path):
            if file.endswith('.txt'):
                with open(os.path.join(self.data_path, file), 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Parse article và summary (format có thể thay đổi)
                    parts = content.split('\n\n')
                    if len(parts) >= 2:
                        articles.append(parts[0])
                        summaries.append(parts[1])
        
        return pd.DataFrame({
            'article': articles,
            'summary': summaries
        })
    
    def preprocess(self, df):
        """Clean và prepare data"""
        print("\n[2] Preprocessing data...")
        
        # Remove NaN values
        df = df.dropna(subset=['article', 'summary'])
        
        # Basic cleaning
        df['article'] = df['article'].apply(self._clean_text)
        df['summary'] = df['summary'].apply(self._clean_text)
        
        # Remove too short or too long samples
        df = df[df['article'].str.len() > 50]
        df = df[df['summary'].str.len() > 10]
        df = df[df['article'].str.len() < 5000]
        
        # Statistics
        print(f"\n✓ Clean samples: {len(df)}")
        print(f"✓ Average article length: {df['article'].str.len().mean():.0f} chars")
        print(f"✓ Average summary length: {df['summary'].str.len().mean():.0f} chars")
        print(f"✓ Compression ratio: {(df['summary'].str.len() / df['article'].str.len()).mean():.2%}")
        
        return df
    
    def _clean_text(self, text):
        """Basic text cleaning"""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters (giữ dấu tiếng Việt)
        text = text.strip()
        
        return text
    
    def split_data(self, df, test_size=0.15, val_size=0.15):
        """Split dataset into train/val/test"""
        print("\n[3] Splitting dataset...")
        
        # First split: train + val vs test
        train_val, test = train_test_split(
            df, test_size=test_size, random_state=42
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val, test_size=val_size_adjusted, random_state=42
        )
        
        print(f"✓ Train: {len(train)} samples")
        print(f"✓ Val: {len(val)} samples")
        print(f"✓ Test: {len(test)} samples")
        
        return train, val, test

# ============================================================================
# PHẦN 4: DATASET CLASS CHO PYTORCH
# ============================================================================

class SummarizationDataset(Dataset):
    """
    PyTorch Dataset cho text summarization
    Hỗ trợ cả extractive và abstractive models
    """
    
    def __init__(self, articles, summaries, tokenizer, max_length=512, 
                 max_target_length=128, model_type='abstractive'):
        self.articles = articles
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_target_length = max_target_length
        self.model_type = model_type
    
    def __len__(self):
        return len(self.articles)
    
    def __getitem__(self, idx):
        article = str(self.articles.iloc[idx])
        summary = str(self.summaries.iloc[idx])
        
        if self.model_type == 'abstractive':
            # Cho mT5, ViT5: seq2seq format
            # Add prefix for T5 models
            article = "summarize: " + article
            
            # Tokenize input
            inputs = self.tokenizer(
                article,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Tokenize target
            targets = self.tokenizer(
                summary,
                max_length=self.max_target_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'labels': targets['input_ids'].squeeze()
            }
        
        else:
            # Cho PhoBERT: extractive format (sentence classification)
            # Này phức tạp hơn, cần implement extractive approach
            # Để đơn giản, ta dùng format tương tự
            inputs = self.tokenizer(
                article,
                summary,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'token_type_ids': inputs.get('token_type_ids', torch.zeros_like(inputs['input_ids'])).squeeze()
            }

# ============================================================================
# PHẦN 5: MODEL IMPLEMENTATIONS
# ============================================================================

class PhoBERTSummarizer(nn.Module):
    """
    PhoBERT for Extractive Summarization
    Architecture: PhoBERT + Classification head
    """
    
    def __init__(self, model_name='vinai/phobert-base', num_labels=2):
        super().__init__()
        self.phobert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.phobert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits


class mT5Summarizer:
    """
    mT5 for Abstractive Summarization
    Google's multilingual T5 model
    """
    
    def __init__(self, model_name='google/mt5-base'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
    def prepare_data(self, train_df, val_df):
        """Prepare datasets"""
        train_dataset = SummarizationDataset(
            train_df['article'],
            train_df['summary'],
            self.tokenizer,
            model_type='abstractive'
        )
        
        val_dataset = SummarizationDataset(
            val_df['article'],
            val_df['summary'],
            self.tokenizer,
            model_type='abstractive'
        )
        
        return train_dataset, val_dataset


class ViT5Summarizer:
    """
    ViT5 for Abstractive Summarization
    VietAI's Vietnamese-optimized T5 model
    """
    
    def __init__(self, model_name='VietAI/vit5-base'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
    def prepare_data(self, train_df, val_df):
        """Prepare datasets"""
        train_dataset = SummarizationDataset(
            train_df['article'],
            train_df['summary'],
            self.tokenizer,
            model_type='abstractive'
        )
        
        val_dataset = SummarizationDataset(
            val_df['article'],
            val_df['summary'],
            self.tokenizer,
            model_type='abstractive'
        )
        
        return train_dataset, val_dataset

# ============================================================================
# PHẦN 6: TRAINING PIPELINE
# ============================================================================

class SummarizationTrainer:
    """
    Unified trainer cho text summarization
    Hỗ trợ cả PhoBERT, mT5, ViT5
    """
    
    def __init__(self, model_name='mt5', config=config):
        self.model_name = model_name
        self.config = config
        self.device = config.device
        
        # Initialize model
        if model_name == 'phobert':
            self.model = PhoBERTSummarizer()
            self.tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
            self.model_type = 'extractive'
        elif model_name == 'mt5':
            self.summarizer = mT5Summarizer()
            self.model = self.summarizer.model
            self.tokenizer = self.summarizer.tokenizer
            self.model_type = 'abstractive'
        elif model_name == 'vit5':
            self.summarizer = ViT5Summarizer()
            self.model = self.summarizer.model
            self.tokenizer = self.summarizer.tokenizer
            self.model_type = 'abstractive'
        
        self.model.to(self.device)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'rouge1': [],
            'rouge2': [],
            'rougeL': []
        }
        
    def train(self, train_df, val_df):
        """
        Main training loop
        """
        print(f"\n{'='*60}")
        print(f"TRAINING {self.model_name.upper()} MODEL")
        print(f"{'='*60}\n")
        
        if self.model_type == 'abstractive':
            # Use HuggingFace Seq2SeqTrainer for T5 models
            self._train_seq2seq(train_df, val_df)
        else:
            # Custom training loop for PhoBERT
            self._train_extractive(train_df, val_df)
    
    def _train_seq2seq(self, train_df, val_df):
        """Training cho abstractive models (mT5, ViT5)"""
        
        # Prepare datasets
        if self.model_name == 'mt5':
            train_dataset, val_dataset = self.summarizer.prepare_data(train_df, val_df)
        else:  # vit5
            train_dataset, val_dataset = self.summarizer.prepare_data(train_df, val_df)
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=f"{self.config.output_dir}/{self.model_name}",
            evaluation_strategy="steps",
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            weight_decay=self.config.weight_decay,
            save_total_limit=3,
            num_train_epochs=self.config.num_epochs,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),  # Mixed precision training
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            load_best_model_at_end=True,
            metric_for_best_model="rouge1",
            greater_is_better=True,
            report_to="none"  # Disable wandb
        )
        
        # Metric computation
        rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            
            # Decode predictions
            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            
            # Replace -100 in labels (used for padding)
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # Compute ROUGE scores
            rouge_scores = {
                'rouge1': [],
                'rouge2': [],
                'rougeL': []
            }
            
            for pred, label in zip(decoded_preds, decoded_labels):
                scores = rouge_scorer_obj.score(label, pred)
                rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
                rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
                rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
            
            return {
                'rouge1': np.mean(rouge_scores['rouge1']),
                'rouge2': np.mean(rouge_scores['rouge2']),
                'rougeL': np.mean(rouge_scores['rougeL'])
            }
        
        # Initialize Trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
        
        # Train
        print("\n🚀 Starting training...")
        train_result = trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(f"{self.config.output_dir}/{self.model_name}")
        
        # Save metrics
        metrics = train_result.metrics
        trainer.save_metrics("train", metrics)
        
        print(f"\n✅ Training completed!")
        print(f"Final training loss: {metrics['train_loss']:.4f}")
        
        return trainer
    
    def _train_extractive(self, train_df, val_df):
        """Training cho extractive models (PhoBERT)"""
        print("\n⚠️  Extractive training với PhoBERT cần implement riêng")
        print("Để đơn giản, bạn nên dùng mT5 hoặc ViT5 cho abstractive summarization")
        pass
    
    def evaluate(self, test_df):
        """Evaluate model trên test set"""
        print(f"\n{'='*60}")
        print(f"EVALUATING {self.model_name.upper()} MODEL")
        print(f"{'='*60}\n")
        
        self.model.eval()
        rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        
        results = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': [],
            'predictions': [],
            'references': []
        }
        
        with torch.no_grad():
            for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
                article = "summarize: " + str(row['article'])
                reference = str(row['summary'])
                
                # Generate summary
                inputs = self.tokenizer(
                    article,
                    max_length=512,
                    truncation=True,
                    return_tensors='pt'
                ).to(self.device)
                
                outputs = self.model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    length_penalty=0.6,
                    early_stopping=True
                )
                
                prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Compute ROUGE
                scores = rouge_scorer_obj.score(reference, prediction)
                results['rouge1'].append(scores['rouge1'].fmeasure)
                results['rouge2'].append(scores['rouge2'].fmeasure)
                results['rougeL'].append(scores['rougeL'].fmeasure)
                results['predictions'].append(prediction)
                results['references'].append(reference)
        
        # Aggregate results
        final_results = {
            'rouge1': np.mean(results['rouge1']),
            'rouge2': np.mean(results['rouge2']),
            'rougeL': np.mean(results['rougeL'])
        }
        
        print(f"\n📊 Evaluation Results:")
        print(f"ROUGE-1: {final_results['rouge1']:.4f}")
        print(f"ROUGE-2: {final_results['rouge2']:.4f}")
        print(f"ROUGE-L: {final_results['rougeL']:.4f}")
        
        return final_results, results

# ============================================================================
# PHẦN 7: VISUALIZATION & ANALYSIS
# ============================================================================

class ResultAnalyzer:
    """Phân tích và visualize kết quả"""
    
    def __init__(self, results, model_name):
        self.results = results
        self.model_name = model_name
        
    def plot_training_history(self, history):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title(f'{self.model_name} - Training & Validation Loss', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # ROUGE scores plot
        axes[1].plot(history['rouge1'], label='ROUGE-1', linewidth=2)
        axes[1].plot(history['rouge2'], label='ROUGE-2', linewidth=2)
        axes[1].plot(history['rougeL'], label='ROUGE-L', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Score', fontsize=12)
        axes[1].set_title(f'{self.model_name} - ROUGE Scores', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'/kaggle/working/{self.model_name}_training_history.png', dpi=300)
        plt.show()
    
    def plot_rouge_distribution(self):
        """Plot ROUGE score distributions"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        scores = ['rouge1', 'rouge2', 'rougeL']
        titles = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
        
        for idx, (score, title) in enumerate(zip(scores, titles)):
            axes[idx].hist(self.results[score], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            axes[idx].axvline(np.mean(self.results[score]), color='red', linestyle='--', 
                            linewidth=2, label=f'Mean: {np.mean(self.results[score]):.3f}')
            axes[idx].set_xlabel('Score', fontsize=12)
            axes[idx].set_ylabel('Frequency', fontsize=12)
            axes[idx].set_title(f'{self.model_name} - {title} Distribution', fontsize=14)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'/kaggle/working/{self.model_name}_rouge_distribution.png', dpi=300)
        plt.show()
    
    def show_sample_predictions(self, n_samples=5):
        """Show random sample predictions"""
        print(f"\n{'='*80}")
        print(f"SAMPLE PREDICTIONS - {self.model_name}")
        print(f"{'='*80}\n")
        
        indices = np.random.choice(len(self.results['predictions']), n_samples, replace=False)
        
        for i, idx in enumerate(indices):
            print(f"\n{'─'*80}")
            print(f"Sample {i+1}")
            print(f"{'─'*80}")
            print(f"\n📝 Reference Summary:")
            print(f"{self.results['references'][idx][:200]}...")
            print(f"\n🤖 Generated Summary:")
            print(f"{self.results['predictions'][idx][:200]}...")
            print(f"\n📊 ROUGE Scores:")
            print(f"ROUGE-1: {self.results['rouge1'][idx]:.3f} | "
                  f"ROUGE-2: {self.results['rouge2'][idx]:.3f} | "
                  f"ROUGE-L: {self.results['rougeL'][idx]:.3f}")
    
    def compare_models(self, all_results):
        """So sánh performance giữa các models"""
        models = list(all_results.keys())
        metrics = ['rouge1', 'rouge2', 'rougeL']
        
        data = []
        for model in models:
            for metric in metrics:
                data.append({
                    'Model': model,
                    'Metric': metric.upper(),
                    'Score': all_results[model][metric]
                })
        
        df_compare = pd.DataFrame(data)
        
        # Plot comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(metrics))
        width = 0.25
        
        for i, model in enumerate(models):
            model_data = df_compare[df_compare['Model'] == model]
            scores = [model_data[model_data['Metric'] == m.upper()]['Score'].values[0] for m in metrics]
            ax.bar(x + i * width, scores, width, label=model)
        
        ax.set_xlabel('Metrics', fontsize=14)
        ax.set_ylabel('Score', fontsize=14)
        ax.set_title('Model Comparison - ROUGE Scores', fontsize=16)
        ax.set_xticks(x + width)
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/kaggle/working/model_comparison.png', dpi=300)
        plt.show()
        
        return df_compare

# ============================================================================
# PHẦN 8: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main pipeline execution
    """
    print("\n" + "="*80)
    print("VIETNAMESE TEXT SUMMARIZATION - COMPLETE PIPELINE")
    print("="*80 + "\n")
    
    # -----------------------------------------------------------------------
    # STEP 1: Load và preprocess data
    # -----------------------------------------------------------------------
    print("\n🔹 STEP 1: DATA LOADING & PREPROCESSING")
    
    # Thay đổi path này theo cấu trúc Kaggle dataset của bạn
    data_path = '/kaggle/input/vlsp2021-summarization/train.csv'
    
    # Load data
    loader = VLSPDataLoader(data_path)
    df = loader.load_data()
    df = loader.preprocess(df)
    
    # Split data
    train_df, val_df, test_df = loader.split_data(df)
    
    # Save splits for reference
    train_df.to_csv('/kaggle/working/train_data.csv', index=False)
    val_df.to_csv('/kaggle/working/val_data.csv', index=False)
    test_df.to_csv('/kaggle/working/test_data.csv', index=False)
    
    # -----------------------------------------------------------------------
    # STEP 2: Train models
    # -----------------------------------------------------------------------
    print("\n🔹 STEP 2: MODEL TRAINING")
    
    # Chọn model để train (có thể train cả 3 để so sánh)
    models_to_train = ['vit5']  # hoặc ['mt5', 'vit5'] để train cả 2
    
    all_results = {}
    
    for model_name in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*60}")
        
        # Initialize trainer
        trainer = SummarizationTrainer(model_name=model_name, config=config)
        
        # Train
        trainer.train(train_df, val_df)
        
        # Evaluate
        final_results, detailed_results = trainer.evaluate(test_df)
        all_results[model_name] = final_results
        
        # Analyze results
        analyzer = ResultAnalyzer(detailed_results, model_name)
        analyzer.plot_rouge_distribution()
        analyzer.show_sample_predictions(n_samples=3)
        
        # Clean up GPU memory
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
    
    # -----------------------------------------------------------------------
    # STEP 3: Compare models
    # -----------------------------------------------------------------------
    if len(all_results) > 1:
        print("\n🔹 STEP 3: MODEL COMPARISON")
        comparison_analyzer = ResultAnalyzer({}, 'comparison')
        df_compare = comparison_analyzer.compare_models(all_results)
        print("\n📊 Final Comparison:")
        print(df_compare)
    
    # -----------------------------------------------------------------------
    # STEP 4: Save final results
    # -----------------------------------------------------------------------
    print("\n🔹 STEP 4: SAVING RESULTS")
    
    # Save results to JSON
    with open('/kaggle/working/final_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Pipeline completed successfully!")
    print("\n📁 Output files:")
    print("  - Models saved in: /kaggle/working/models/")
    print("  - Visualizations: /kaggle/working/*.png")
    print("  - Results: /kaggle/working/final_results.json")

if __name__ == "__main__":
    main()
