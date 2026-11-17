import json

def fix_day4():
    f = 'Day4_LSTM_Attention_Demand_Forecasting.ipynb'
    with open(f, 'r') as fp:
        d = json.load(fp)
    
    # Simple search and replace for the dataset path string across all code cells
    for cell in d['cells']:
        if cell['cell_type'] == 'code':
            for i, line in enumerate(cell['source']):
                if "/Users/shailaneelofar/Desktop/Data Science/AgroMind/tomato_retail_sales_daily_2009_2022 (1).csv" in line:
                    cell['source'][i] = line.replace("/Users/shailaneelofar/Desktop/Data Science/AgroMind/tomato_retail_sales_daily_2009_2022 (1).csv", "tomato_retail_sales_daily_2009_2022 (1).csv")
                    
    with open(f, 'w') as fp:
        json.dump(d, fp, indent=1)
    print("Day 4 path fixed.")

def fix_day3():
    f = 'Day3_CNN_Classification.ipynb'
    with open(f, 'r') as fp:
        d = json.load(fp)
        
    for cell in d['cells']:
        if cell['cell_type'] == 'code':
            for i, line in enumerate(cell['source']):
                if "EPOCHS =" in line or "epochs=" in line:
                    # Let's just find the line declaring EPOCHS
                    if line.startswith("EPOCHS ="):
                        cell['source'][i] = "EPOCHS = 2\\n"
                        print("Day 3 Epochs reduced to 2.")
                        
    with open(f, 'w') as fp:
        json.dump(d, fp, indent=1)

fix_day4()
fix_day3()
