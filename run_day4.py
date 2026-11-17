import json
import os

notebook_path = "Day4_LSTM_Attention_Demand_Forecasting.ipynb"

with open(notebook_path, 'r') as f:
    nb = json.load(f)

code_cells = [cell['source'] for cell in nb['cells'] if cell['cell_type'] == 'code']

full_code = ""
for cell_source in code_cells:
    if isinstance(cell_source, list):
        full_code += "".join(cell_source) + "\n\n"
    else:
        full_code += cell_source + "\n\n"

# Remove any magic commands like !pip or %matplotlib
cleaned_lines = []
for line in full_code.split("\n"):
    if line.startswith("!") or line.startswith("%"):
        continue
    # Fix display() function which doesn't exist in pure python
    if "display(" in line:
        line = line.replace("display(", "print(")
    # Fix matplotlib to use Agg
    if "import matplotlib.pyplot as plt" in line:
        cleaned_lines.append("import matplotlib")
        cleaned_lines.append("matplotlib.use('Agg')")
    cleaned_lines.append(line)

cleaned_code = "\n".join(cleaned_lines)

with open("day4_extracted.py", "w") as f:
    f.write(cleaned_code)

print("Extracted code to day4_extracted.py. Run 'python day4_extracted.py' to execute it.")
