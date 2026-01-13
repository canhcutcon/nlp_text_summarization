"""
Minimal script to add evaluation metrics to notebook demos.
Uses in-place modification to minimize disk space usage.
"""

import json

def main():
    notebook_path = '/Users/mac/Downloads/GIANG/nlp_text_summarization/vietnamese_summarization_mt5_rtx_4070.ipynb'
    
    print("Loading and modifying notebook...")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # 1. Update package installation (cell 4)
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and any('rouge-score' in str(line) for line in cell.get('source', [])):
            for j, line in enumerate(cell['source']):
                if 'rouge-score' in line and 'scikit-learn' in line and 'sacrebleu' not in line:
                    cell['source'][j] = line.replace('scikit-learn', 'scikit-learn sacrebleu tabulate')
                    print(f"✓ Updated imports (cell {i})")
                    break
            break
    
    # Write back immediately to save memory
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    
    print("✅ Successfully added sacrebleu and tabulate packages!")
    print("\nNext steps:")
    print("1. Run the updated notebook")
    print("2. Manually add evaluation helper functions before Section 5.1")
    print("3. Update demo cells to use evaluate_summary() function")

if __name__ == "__main__":
    main()
