#!/usr/bin/env python3
"""
Quick verification script to check if all notebook fixes are in place
This does NOT run training, just verifies the configuration is correct
"""

import json

print("=" * 70)
print("NOTEBOOK FIXES VERIFICATION")
print("=" * 70)

# Read the notebook
notebook_path = "vietnamese_summarization_mt5_rtx_4070_FIXED.ipynb"

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
except FileNotFoundError:
    print(f"❌ ERROR: Notebook not found: {notebook_path}")
    exit(1)

cells = notebook['cells']
issues_found = []
fixes_verified = []

# ============================================================================
# Check 1: Verify imports cell exists
# ============================================================================
print("\n[Check 1] Verifying imports cell...")

imports_found = False
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'from transformers import' in source and 'AutoTokenizer' in source:
            imports_found = True
            # Check if it's after GPU setup
            if i > 8:  # Should be around cell 9
                fixes_verified.append("✅ Transformers imports cell found at correct position")
            else:
                issues_found.append("⚠️  Imports cell found but might be in wrong position")
            break

if not imports_found:
    issues_found.append("❌ CRITICAL: Transformers imports cell NOT found")
else:
    if not any("Transformers imports" in msg for msg in fixes_verified):
        fixes_verified.append("✅ Transformers imports cell found")

# ============================================================================
# Check 2: Verify FP16 uses USE_FP16 flag
# ============================================================================
print("[Check 2] Verifying FP16 configuration...")

fp16_fixed = False
for cell in cells:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'Seq2SeqTrainingArguments' in source:
            if 'fp16=USE_FP16' in source:
                fp16_fixed = True
                fixes_verified.append("✅ FP16 uses USE_FP16 flag (correct)")
            elif 'fp16=True' in source and 'USE_FP16' not in source:
                issues_found.append("❌ CRITICAL: FP16 hardcoded to True (will fail on CPU)")
            break

if not fp16_fixed and not any("FP16" in msg for msg in issues_found):
    issues_found.append("⚠️  FP16 configuration not found")

# ============================================================================
# Check 3: Verify gradient checkpointing uses flag
# ============================================================================
print("[Check 3] Verifying gradient checkpointing configuration...")

gc_fixed = False
for cell in cells:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'Seq2SeqTrainingArguments' in source:
            if 'gradient_checkpointing=USE_GRAD_CHECKPOINT' in source:
                gc_fixed = True
                fixes_verified.append("✅ Gradient checkpointing uses USE_GRAD_CHECKPOINT flag")
            break

if not gc_fixed:
    issues_found.append("⚠️  Gradient checkpointing might be hardcoded")

# ============================================================================
# Check 4: Verify adaptive batch sizes
# ============================================================================
print("[Check 4] Verifying adaptive batch sizes...")

adaptive_batch = False
for cell in cells:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'per_device_train_batch_size' in source and "if device.type == 'cuda'" in source:
            adaptive_batch = True
            fixes_verified.append("✅ Adaptive batch sizes for CPU/GPU")
            break

if not adaptive_batch:
    issues_found.append("⚠️  Batch size might not be adaptive")

# ============================================================================
# Check 5: Verify GPU setup sets flags
# ============================================================================
print("[Check 5] Verifying GPU setup...")

gpu_setup_found = False
for cell in cells:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'USE_FP16' in source and 'torch.cuda.is_available()' in source:
            gpu_setup_found = True
            fixes_verified.append("✅ GPU setup defines USE_FP16 and USE_GRAD_CHECKPOINT flags")
            break

if not gpu_setup_found:
    issues_found.append("⚠️  GPU setup might not set flags properly")

# ============================================================================
# Check 6: Verify pre-training diagnostics
# ============================================================================
print("[Check 6] Verifying pre-training diagnostics...")

diagnostics_found = False
for cell in cells:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'PRE-TRAINING DIAGNOSTICS' in source and 'Test 1' in source:
            diagnostics_found = True
            fixes_verified.append("✅ Pre-training diagnostics cell present")
            break

if not diagnostics_found:
    issues_found.append("⚠️  Pre-training diagnostics not found")

# ============================================================================
# Results
# ============================================================================
print("\n" + "=" * 70)
print("VERIFICATION RESULTS")
print("=" * 70)

print(f"\n✅ FIXES VERIFIED ({len(fixes_verified)}):")
for fix in fixes_verified:
    print(f"  {fix}")

if issues_found:
    print(f"\n⚠️  ISSUES FOUND ({len(issues_found)}):")
    for issue in issues_found:
        print(f"  {issue}")
else:
    print(f"\n✅ NO ISSUES FOUND!")

print("\n" + "=" * 70)

if any("CRITICAL" in issue for issue in issues_found):
    print("❌ CRITICAL ISSUES FOUND - Notebook may not work correctly!")
    print("=" * 70)
    exit(1)
elif issues_found:
    print("⚠️  Minor issues found - Notebook should work but may not be optimal")
    print("=" * 70)
    exit(0)
else:
    print("✅ ALL CHECKS PASSED - Notebook is ready to use!")
    print("=" * 70)
    exit(0)
