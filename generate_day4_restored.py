import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Cell 1: Title and Problem Statement
cells.append(nbf.v4.new_markdown_cell("""# Day 4: Time-Series Demand Forecasting using LSTM and Attention
## 01 Problem Statement

**Business Problem:**
While Days 1-3 focused on using Computer Vision to detect the ripeness of tomatoes for automated robotic harvesting, an integrated agricultural system also needs to manage the supply chain. Knowing how many tomatoes to harvest requires an understanding of future market demand.

**Goal:**
Use historical retail sales data to forecast the expected demand for tomatoes on the next day. This serves as a **supporting capability** to help farmers and automated systems plan their daily harvest quotas.

**Why LSTM and Attention?**
- **LSTM (Long Short-Term Memory):** Excellent at capturing temporal dependencies and sequences in time-series data.
- **Attention Mechanism:** Allows the model to explicitly learn which past days in the historical window are most important for predicting tomorrow's demand.
"""))

# Cell 2: Imports
cells.append(nbf.v4.new_markdown_cell("""## 02 Imports and Setup"""))
cells.append(nbf.v4.new_code_cell("""%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Deep Learning and ML
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Layer
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set random seeds for reproducibility
import random
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

print(f"TensorFlow Version: {tf.__version__}")
"""))

# Cell 3: Load Dataset
cells.append(nbf.v4.new_markdown_cell("""## 03 Load Dataset
The dataset consists of real daily retail sales data for tomatoes from 2009 to 2022."""))
cells.append(nbf.v4.new_code_cell("""# Load dataset
dataset_path = 'tomato_retail_sales_daily_2009_2022 (1).csv'
df = pd.read_csv(dataset_path)

# Parse dates and check date range
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

display(df.head())
print(f"Dataset Shape: {df.shape}")
print(f"Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
"""))

# Cell 4: EDA
cells.append(nbf.v4.new_markdown_cell("""## 04 Exploratory Data Analysis
Visualize the historical demand over time."""))
cells.append(nbf.v4.new_code_cell("""# Sort strictly by date
df = df.sort_values('Date').reset_index(drop=True)

# Plot historical demand over time
plt.figure(figsize=(15, 5))
plt.plot(df['Date'], df['Quantity Sold (kg)'], color='green', alpha=0.7)
plt.title('Daily Tomato Demand Over Time (2009 - 2022)')
plt.xlabel('Date')
plt.ylabel('Quantity Sold (kg)')
plt.grid(True, alpha=0.3)
plt.show()
"""))

# Cell 5: Chronological Split and Scaling
cells.append(nbf.v4.new_markdown_cell("""## 05 Data Splitting and Scaling
**Chronological Split:**
- **Train:** 2009–2018
- **Validation:** 2019–2020
- **Test:** 2021–2022

*Crucial:* We must fit the `MinMaxScaler` **only** on the training set to prevent future data leakage."""))
cells.append(nbf.v4.new_code_cell("""# --- STEP: DATA SCALING ---
# LSTMs are sensitive to the scale of input data. We scale demand to [0, 1].
target_col = 'Quantity Sold (kg)'
data = df[['Date', target_col]].copy()

# Chronological Split
train_df = data[(data['Date'].dt.year >= 2009) & (data['Date'].dt.year <= 2018)]
val_df   = data[(data['Date'].dt.year >= 2019) & (data['Date'].dt.year <= 2020)]
test_df  = data[(data['Date'].dt.year >= 2021) & (data['Date'].dt.year <= 2022)]

print(f"Train samples: {len(train_df)}")
print(f"Val samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit ONLY on training data to prevent leakage!
train_scaled = scaler.fit_transform(train_df[[target_col]])
val_scaled   = scaler.transform(val_df[[target_col]])
test_scaled  = scaler.transform(test_df[[target_col]])

# Extract raw targets for evaluation later
actual_test_demand = test_df[target_col].values
"""))

# Cell 6: Sequences
cells.append(nbf.v4.new_markdown_cell("""## 06 Sequence Generation
Convert the 1D time-series into 3D sequences `(Samples, Timesteps, Features)` required by LSTMs. We will use a **30-day lookback window**."""))
cells.append(nbf.v4.new_code_cell("""# --- STEP: SEQUENCE GENERATION ---
# We convert 1D time-series data into 3D sequences (Samples, Timesteps, Features) for the LSTM.
LOOKBACK = 30

def create_sequences(data_array, lookback):
    X, y = [], []
    for i in range(len(data_array) - lookback):
        X.append(data_array[i : i + lookback, 0])
        y.append(data_array[i + lookback, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, LOOKBACK)
X_val, y_val     = create_sequences(val_scaled, LOOKBACK)
X_test, y_test   = create_sequences(test_scaled, LOOKBACK)

# Reshape to 3D: (samples, timesteps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_val   = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
X_test  = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print(f"X_train shape: {X_train.shape} (Samples, Timesteps, Features)")
print(f"y_train shape: {y_train.shape}")
"""))

# Cell 7: Baselines
cells.append(nbf.v4.new_markdown_cell("""## 07 Baseline Models
Before using Deep Learning, we establish simple baselines on the Test set:
1. **Naive Forecast:** Predicts tomorrow's demand is exactly the same as today's.
2. **7-Day Moving Average:** Predicts tomorrow's demand is the average of the last 7 days."""))
cells.append(nbf.v4.new_code_cell("""# 1. Naive Forecast
# Shift test data by 1 day to simulate "predicting tomorrow using today"
naive_forecast = test_df[target_col].shift(1).copy()
naive_forecast.iloc[0] = val_df[target_col].iloc[-1]
naive_forecast = naive_forecast.values

# 2. 7-Day Moving Average
# Prepend the last 7 days of validation data to calculate the first 7 days of test data properly
history = pd.concat([val_df[target_col].iloc[-7:], test_df[target_col]])
moving_avg_forecast = history.rolling(window=7).mean().shift(1).dropna().values

# Align actuals and baselines (ignoring the first LOOKBACK days to match LSTM test sequences)
aligned_actuals = actual_test_demand[LOOKBACK:]
aligned_naive = naive_forecast[LOOKBACK:]
aligned_ma = moving_avg_forecast[LOOKBACK:]
"""))


# Cell 8: Build Models
cells.append(nbf.v4.new_markdown_cell("""## 08 Build Deep Learning Models
We build two models:
1. **Standard LSTM:** Processes the 30-day sequence.
2. **LSTM + Attention:** Uses a custom Attention layer to weigh the importance of specific days in the sequence."""))
cells.append(nbf.v4.new_code_cell("""# --- STEP: BUILD STANDARD LSTM ---
# This model processes the sequence sequentially without explicit attention weighting.
lstm_model = Sequential([
    Input(shape=(LOOKBACK, 1)),
    LSTM(64, activation='tanh'),
    Dense(32, activation='relu'),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])


# --- STEP: BUILD LSTM + ATTENTION ---
# The CustomAttention layer computes a weighted sum of the LSTM's hidden states,
# allowing the model to focus on the most important historical days.
class CustomAttention(Layer):
    def __init__(self, **kwargs):
        super(CustomAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', 
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(CustomAttention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

inputs = Input(shape=(LOOKBACK, 1))
x = LSTM(64, return_sequences=True, activation='tanh')(inputs)
x = CustomAttention()(x)
x = Dense(32, activation='relu')(x)
outputs = Dense(1)(x)

attention_model = Model(inputs=inputs, outputs=outputs)
attention_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("Models built successfully.")
"""))

# Cell 9: Train
cells.append(nbf.v4.new_markdown_cell("""## 09 Model Training"""))
cells.append(nbf.v4.new_code_cell("""# --- STEP: MODEL TRAINING ---
# We train both models using Early Stopping to halt training when validation loss stops improving.
EPOCHS = 30
BATCH_SIZE = 32
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)

print("Training Standard LSTM...")
history_lstm = lstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=0
)

print("Training LSTM + Attention...")
history_attn = attention_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=0
)

print("Training Complete.")
"""))

# Cell 10: Evaluation
cells.append(nbf.v4.new_markdown_cell("""## 10 Evaluation & Comparison
We evaluate all models on the Test set (2021-2022) using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)."""))
cells.append(nbf.v4.new_code_cell("""# --- STEP: EVALUATION ---
# 1. Make predictions
lstm_preds_scaled = lstm_model.predict(X_test, verbose=0)
attn_preds_scaled = attention_model.predict(X_test, verbose=0)

# 2. Convert predictions back to kg
lstm_preds = scaler.inverse_transform(lstm_preds_scaled).flatten()
attn_preds = scaler.inverse_transform(attn_preds_scaled).flatten()

# 3. Calculate metrics function
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

# 4. Compile Results
metrics = {
    "1. Naive Forecast": calculate_metrics(aligned_actuals, aligned_naive),
    "2. 7-Day Moving Avg": calculate_metrics(aligned_actuals, aligned_ma),
    "3. Standard LSTM": calculate_metrics(aligned_actuals, lstm_preds),
    "4. LSTM + Attention": calculate_metrics(aligned_actuals, attn_preds)
}

results_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['MAE (kg)', 'RMSE (kg)'])
display(results_df)
"""))

# Cell 11: Plots
cells.append(nbf.v4.new_markdown_cell("""## 11 Visualization"""))
cells.append(nbf.v4.new_code_cell("""# Plotting the first 100 days of the test set predictions for visual clarity
PLOT_DAYS = 100

plt.figure(figsize=(15, 6))
plt.plot(aligned_actuals[:PLOT_DAYS], label='Actual Demand (kg)', color='black', linewidth=2)
plt.plot(aligned_naive[:PLOT_DAYS], label='Naive Baseline', color='grey', linestyle='--', alpha=0.5)
plt.plot(lstm_preds[:PLOT_DAYS], label='Standard LSTM', color='blue', alpha=0.7)
plt.plot(attn_preds[:PLOT_DAYS], label='LSTM + Attention', color='red', alpha=0.9, linestyle='-.')

plt.title('Actual vs Predicted Tomato Demand (First 100 Days of Test Set)')
plt.xlabel('Days')
plt.ylabel('Quantity Sold (kg)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
"""))

# Cell 12: Save
cells.append(nbf.v4.new_markdown_cell("""## 12 Save the Best Model
We save the best performing model and the scaler so they can be loaded by `api.py` for inference."""))
cells.append(nbf.v4.new_code_cell("""import os
import joblib

os.makedirs('models', exist_ok=True)

# Select best model based on MAE
if results_df.loc["4. LSTM + Attention", "MAE (kg)"] < results_df.loc["3. Standard LSTM", "MAE (kg)"]:
    best_model = attention_model
    print("Saving Attention Model as best model.")
else:
    best_model = lstm_model
    print("Saving Standard LSTM as best model.")

best_model.save('models/best_lstm_attention_model.keras')
joblib.dump(scaler, 'models/demand_scaler.pkl')

print("Best model and scaler saved successfully to 'models/' directory.")
"""))

nb.cells = cells
with open('Day4_LSTM_Attention_Demand_Forecasting.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook Day 4 restored successfully!")
