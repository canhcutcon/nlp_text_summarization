#!/usr/bin/env python3
"""
Script to fix vietnamese_summarization_mt5_rtx_4070.ipynb
Adds diagnostic checks and fixes device detection for RTX 3090
"""

import json
import sys

# Additional diagnostic cell to insert before training
DIAGNOSTIC_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# 🔍 PRE-TRAINING DIAGNOSTICS (PREVENTS LOSS = 0 BUG)\n",
        "# ============================================================================\n",
        "print(\"\\n\" + \"=\" * 60)\n",
        "print(\"🔍 PRE-TRAINING DIAGNOSTICS\")\n",
        "print(\"=\" * 60)\n",
        "\n",
        "# Test 1: Check Labels\n",
        "print(\"\\n[Test 1] Checking labels...\")\n",
        "sample = tokenized_datasets[\"train\"][0]\n",
        "valid_labels = [l for l in sample['labels'] if l != -100]\n",
        "\n",
        "if len(valid_labels) == 0:\n",
        "    print(\"❌ CRITICAL: All labels are -100! Training will fail!\")\n",
        "    raise ValueError(\"All labels are -100\")\n",
        "else:\n",
        "    print(f\"✅ Valid labels: {len(valid_labels)} tokens\")\n",
        "    print(f\"   Sample: {tokenizer.decode(valid_labels[:20])}\")\n",
        "\n",
        "# Test 2: Forward Pass\n",
        "print(\"\\n[Test 2] Testing forward pass...\")\n",
        "from torch.utils.data import DataLoader\n",
        "\n",
        "test_loader = DataLoader(\n",
        "    tokenized_datasets[\"train\"].select(range(2)),\n",
        "    batch_size=2,\n",
        "    collate_fn=data_collator\n",
        ")\n",
        "\n",
        "batch = next(iter(test_loader))\n",
        "batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}\n",
        "\n",
        "model.eval()\n",
        "with torch.no_grad():\n",
        "    outputs = model(**batch_device)\n",
        "    test_loss = outputs.loss.item()\n",
        "\n",
        "print(f\"Test loss: {test_loss:.4f}\")\n",
        "\n",
        "if test_loss == 0.0:\n",
        "    print(\"❌ CRITICAL: Test loss is 0! Model cannot learn!\")\n",
        "    raise ValueError(\"Test loss is 0\")\n",
        "elif torch.isnan(torch.tensor(test_loss)):\n",
        "    print(\"❌ CRITICAL: Test loss is NaN! Numerical instability!\")\n",
        "    raise ValueError(\"Test loss is NaN\")\n",
        "else:\n",
        "    print(\"✅ Test loss looks normal!\")\n",
        "\n",
        "# Test 3: Gradients\n",
        "print(\"\\n[Test 3] Testing gradients...\")\n",
        "model.train()\n",
        "outputs = model(**batch_device)\n",
        "loss = outputs.loss\n",
        "loss.backward()\n",
        "\n",
        "has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 \n",
        "               for p in model.parameters() if p.requires_grad)\n",
        "\n",
        "if not has_grad:\n",
        "    print(\"❌ CRITICAL: No gradients computed! Model cannot learn!\")\n",
        "    raise ValueError(\"No gradients\")\n",
        "else:\n",
        "    print(\"✅ Gradients computed successfully!\")\n",
        "\n",
        "model.zero_grad()\n",
        "\n",
        "print(\"\\n\" + \"=\" * 60)\n",
        "print(\"✅ ALL DIAGNOSTIC TESTS PASSED!\")\n",
        "print(\"Proceeding with training...\")\n",
        "print(\"=\" * 60)"
    ]
}

# Fixed GPU detection cell
FIXED_GPU_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import torch\n",
        "import gc\n",
        "\n",
        "# ============================================================================\n",
        "# FIXED: Proper GPU Detection for RTX 3090\n",
        "# ============================================================================\n",
        "print(\"\\n\" + \"=\" * 60)\n",
        "print(\"🎯 GPU Setup (FIXED for RTX 3090)\")\n",
        "print(\"=\" * 60)\n",
        "\n",
        "if torch.cuda.is_available():\n",
        "    torch.cuda.empty_cache()\n",
        "    gc.collect()\n",
        "    device = torch.device(\"cuda\")\n",
        "    print(f\"✅ Using CUDA GPU: {torch.cuda.get_device_name(0)}\")\n",
        "    print(f\"   Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\")\n",
        "    USE_FP16 = True\n",
        "    USE_GRAD_CHECKPOINT = True\n",
        "else:\n",
        "    device = torch.device(\"cpu\")\n",
        "    print(\"⚠️  No CUDA GPU detected - using CPU (will be slow!)\")\n",
        "    USE_FP16 = False\n",
        "    USE_GRAD_CHECKPOINT = False"
    ]
}

def main():
    # Read the original notebook
    with open('vietnamese_summarization_mt5_rtx_4070.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Track changes
    changes_made = []
    
    # Find cells to modify
    gpu_cell_index = None
    model_load_index = None
    training_cell_index = None
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and 'source' in cell:
            source_text = ''.join(cell['source'])
            
            # Find GPU detection cell
            if 'torch.cuda.is_available' in source_text and ('Clear GPU Memory' in source_text or 'GPU memory cleared' in source_text):
                gpu_cell_index = i
                nb['cells'][i]['source'] = FIXED_GPU_CELL['source']
                changes_made.append(f"Fixed GPU detection cell at index {i}")
            
            # Find model loading cell
            if 'AutoModelForSeq2SeqLM.from_pretrained' in source_text and 'google/mt5-small' in source_text:
                model_load_index = i
            
            # Find training cell (contains trainer.train())
            if 'trainer.train()' in source_text and 'Starting training' in source_text:
                training_cell_index = i
    
    # Insert diagnostic cell before training
    if training_cell_index:
        nb['cells'].insert(training_cell_index, DIAGNOSTIC_CELL)
        changes_made.append(f"Inserted diagnostic cell before training at index {training_cell_index}")
    
    # Save the fixed notebook
    output_file = 'vietnamese_summarization_mt5_rtx_4070_FIXED.ipynb'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    
    print("✅ Fixed notebook created!")
    print(f"   Output: {output_file}")
    print("\nChanges made:")
    for change in changes_made:
        print(f"  - {change}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
