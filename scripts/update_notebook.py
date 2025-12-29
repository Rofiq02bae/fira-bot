
import json
import os
import re

notebook_path = r'd:\bot\New folder\training\bert_train.ipynb'
normalizer_path = r'd:\bot\New folder\core\processors\text_normalizer.py'

# 1. Read the project's TextNormalizer code
with open(normalizer_path, 'r', encoding='utf-8') as f:
    normalizer_content = f.read()

# Extract the class TextNormalizer from the file
# We'll use a regex to grab the class definition
match = re.search(r'class TextNormalizer:.*?(?=\n\n#|$)', normalizer_content, re.DOTALL)
if match:
    new_normalizer_code = match.group(0).strip()
    # Convert to list of lines with \n suffix as per notebook format
    new_normalizer_lines = [line + '\n' for line in new_normalizer_code.split('\n')]
else:
    print("Could not find TextNormalizer class in the project file.")
    exit(1)

# 2. Read and update the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update cell 3 preview (Download & Preview cell)
# We look for the cell that contains 'dataset_url' or 'urllib.request'
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and any('df.head' in line for line in cell['source']):
        for i, line in enumerate(cell['source']):
            if "df.head(3)[['intent', 'pattern']]" in line:
                cell['source'][i] = line.replace("['intent', 'pattern']", "['intent', 'pattern', 'is_master']")
                print(f"Updated preview in cell with 'df.head'")

# Update TextNormalizer in the training cell (index 3 usually)
# We look for 'class TextNormalizer' in cell source
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and any('class TextNormalizer' in line for line in cell['source']):
        # We need to find the range of lines that define the class TextNormalizer
        source = cell['source']
        start_idx = -1
        end_idx = -1
        
        for i, line in enumerate(source):
            if 'class TextNormalizer' in line:
                start_idx = i
                break
        
        if start_idx != -1:
            # Look for the end of the class (next top-level comment or global instance)
            for i in range(start_idx + 1, len(source)):
                if '# Global normalizer instance' in source[i] or '# Global instance' in source[i] or source[i].startswith('# ===================='):
                    end_idx = i
                    break
            
            if end_idx == -1: end_idx = len(source)
            
            # Replace the old class with the new one
            # Note: we need to keep the global instance part if it's there
            cell['source'] = source[:start_idx] + new_normalizer_lines + ['\n'] + source[end_idx:]
            print("Updated TextNormalizer class in training cell.")

# 3. Save the notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Notebook updated successfully.")
