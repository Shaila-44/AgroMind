import json

f = 'Day3_CNN_Classification.ipynb'
with open(f, 'r') as fp:
    d = json.load(fp)

# Fix Block 2 missing Conv2D
d['cells'][16]['source'][8] = "        Conv2D(64, (3, 3), activation='relu', padding='same'),\n"

# Fix literal backslash n in EPOCHS
d['cells'][18]['source'][8] = "EPOCHS = 2\n"

with open(f, 'w') as fp:
    json.dump(d, fp, indent=1)

print("Fixed Day 3 JSON!")
