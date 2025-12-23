#!/usr/bin/env python3
"""
Quick Start Script for Vietnamese Text Summarization
=====================================================

Chạy script này để:
1. Check environment
2. Download sample data
3. Train ViT5 model
4. Evaluate results

Usage:
    python quick_start.py
    
Or in Kaggle/Colab:
    !python quick_start.py --data-path /kaggle/input/vlsp2021/train.csv
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

def check_environment():
    """Check if environment is ready"""
    print("\n" + "="*60)
    print("ENVIRONMENT CHECK")
    print("="*60)
    
    # Check Python version
    print(f"\n✓ Python version: {sys.version.split()[0]}")
    
    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    except ImportError:
        print("✗ PyTorch not found. Installing...")
        os.system("pip install torch -q")
    
    # Check Transformers
    try:
        import transformers
        print(f"✓ Transformers version: {transformers.__version__}")
    except ImportError:
        print("✗ Transformers not found. Installing...")
        os.system("pip install transformers==4.35.0 -q")
    
    # Check other dependencies
    required = ['pandas', 'numpy', 'tqdm', 'matplotlib', 'seaborn', 'rouge_score']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            missing.append(package)
            print(f"✗ {package} not found")
    
    if missing:
        print(f"\nInstalling missing packages: {', '.join(missing)}")
        os.system(f"pip install {' '.join(missing)} -q")
    
    print("\n✅ Environment check complete!\n")
    return True


def download_sample_data():
    """Download sample VLSP data if not exists"""
    print("\n" + "="*60)
    print("SAMPLE DATA DOWNLOAD")
    print("="*60)
    
    # Create sample data
    import pandas as pd
    
    # Sample Vietnamese news articles and summaries
    sample_data = {
        'article': [
            "Hôm nay, Bộ Y tế công bố thêm 15.527 ca nhiễm COVID-19 mới, nâng tổng số ca nhiễm tại Việt Nam lên 895.326 ca. TP.HCM tiếp tục dẫn đầu với 6.784 ca, tiếp theo là Bình Dương với 4.632 ca và Đồng Nai với 1.139 ca.",
            "Thủ tướng Chính phủ Phạm Minh Chính đã ký Nghị định về việc thực hiện giãn cách xã hội trên toàn quốc để phòng chống dịch COVID-19. Nghị định có hiệu lực từ ngày 23/7/2021 và áp dụng cho tất cả các tỉnh thành có dịch.",
            "Giá vàng trong nước hôm nay tăng mạnh, theo đà tăng của giá vàng thế giới. Vàng SJC được các doanh nghiệp niêm yết ở mức 56,5 triệu đồng/lượng mua vào và 57 triệu đồng/lượng bán ra.",
        ],
        'summary': [
            "Bộ Y tế công bố 15.527 ca COVID-19 mới, TP.HCM dẫn đầu với 6.784 ca.",
            "Thủ tướng ký Nghị định về giãn cách xã hội toàn quốc từ 23/7.",
            "Giá vàng trong nước tăng theo giá thế giới, vàng SJC 57 triệu/lượng.",
        ]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Save to file
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/sample_data.csv', index=False, encoding='utf-8')
    
    print("\n✓ Sample data created: data/sample_data.csv")
    print(f"✓ Total samples: {len(df)}")
    print("\nNote: Đây chỉ là sample data nhỏ để test.")
    print("Để training thực sự, hãy download VLSP 2021 dataset.\n")
    
    return 'data/sample_data.csv'


def quick_train(data_path, model_name='VietAI/vit5-base', epochs=1):
    """Quick training on sample data"""
    print("\n" + "="*60)
    print("QUICK TRAINING")
    print("="*60)
    
    import pandas as pd
    import torch
    from transformers import (
        AutoTokenizer, 
        AutoModelForSeq2SeqLM,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
        DataCollatorForSeq2Seq
    )
    from torch.utils.data import Dataset
    import numpy as np
    from rouge_score import rouge_scorer
    
    # Load data
    print(f"\n📂 Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} samples")
    
    # Split data
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"✓ Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Dataset class
    class QuickDataset(Dataset):
        def __init__(self, articles, summaries, tokenizer):
            self.articles = articles
            self.summaries = summaries
            self.tokenizer = tokenizer
        
        def __len__(self):
            return len(self.articles)
        
        def __getitem__(self, idx):
            article = "summarize: " + str(self.articles.iloc[idx])
            summary = str(self.summaries.iloc[idx])
            
            inputs = self.tokenizer(article, max_length=512, padding='max_length', 
                                   truncation=True, return_tensors='pt')
            targets = self.tokenizer(summary, max_length=128, padding='max_length',
                                    truncation=True, return_tensors='pt')
            
            return {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'labels': targets['input_ids'].squeeze()
            }
    
    # Load model
    print(f"\n🤖 Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"✓ Model loaded on {device}")
    
    # Create datasets
    train_dataset = QuickDataset(train_df['article'], train_df['summary'], tokenizer)
    val_dataset = QuickDataset(val_df['article'], val_df['summary'], tokenizer)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    
    # Metrics
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
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
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir='./quick_train_output',
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=5e-5,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to='none'
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # Train
    print(f"\n🚀 Starting training for {epochs} epoch(s)...")
    trainer.train()
    
    # Evaluate
    print("\n📊 Evaluating on validation set...")
    results = trainer.evaluate()
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nROUGE-1: {results['eval_rouge1']:.4f}")
    print(f"ROUGE-2: {results['eval_rouge2']:.4f}")
    print(f"ROUGE-L: {results['eval_rougeL']:.4f}")
    print(f"Loss: {results['eval_loss']:.4f}")
    
    # Save model
    output_dir = './quick_train_final'
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n✅ Model saved to: {output_dir}")
    
    # Test generation
    print("\n" + "="*60)
    print("SAMPLE GENERATION")
    print("="*60)
    
    test_article = df.iloc[0]['article']
    test_summary = df.iloc[0]['summary']
    
    inputs = tokenizer("summarize: " + test_article, max_length=512, 
                      truncation=True, return_tensors='pt').to(device)
    outputs = model.generate(**inputs, max_length=128, num_beams=4)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\n📝 Original Article:")
    print(f"{test_article[:200]}...")
    print(f"\n📄 Reference Summary:")
    print(f"{test_summary}")
    print(f"\n🤖 Generated Summary:")
    print(f"{generated}")
    
    print("\n✅ Quick training complete!")


def main():
    parser = argparse.ArgumentParser(description='Quick Start for Vietnamese Text Summarization')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to training data CSV file')
    parser.add_argument('--model', type=str, default='VietAI/vit5-base',
                       help='Model name or path')
    parser.add_argument('--epochs', type=int, default=1,
                       help='Number of training epochs')
    parser.add_argument('--skip-check', action='store_true',
                       help='Skip environment check')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip sample data download')
    
    args = parser.parse_args()
    
    print("\n" + "🚀"*30)
    print("VIETNAMESE TEXT SUMMARIZATION - QUICK START")
    print("🚀"*30)
    
    # Step 1: Check environment
    if not args.skip_check:
        check_environment()
    
    # Step 2: Get data path
    if args.data_path is None:
        if not args.skip_download:
            args.data_path = download_sample_data()
        else:
            print("\n❌ Error: No data path provided and skip-download is set")
            print("Usage: python quick_start.py --data-path /path/to/data.csv")
            sys.exit(1)
    
    # Check if data exists
    if not os.path.exists(args.data_path):
        print(f"\n❌ Error: Data file not found: {args.data_path}")
        sys.exit(1)
    
    # Step 3: Train
    try:
        quick_train(args.data_path, args.model, args.epochs)
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "🎉"*30)
    print("QUICK START COMPLETE!")
    print("🎉"*30)
    print("\nNext steps:")
    print("1. Review results in ./quick_train_output/")
    print("2. Try full training with: python vietnamese_text_summarization.py")
    print("3. For detailed guide, see: KAGGLE_SETUP_GUIDE.md")
    print("4. For advanced evaluation, check: evaluation_utils.py")


if __name__ == "__main__":
    main()
