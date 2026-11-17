# AgroMind Day 4: Demand Forecasting Evaluation Report

## 1. Problem Definition
AgroMind's core objective is to enable robotic arms to visually detect and harvest ripe tomatoes. While Days 1-3 focused on the visual perception component ("Is this tomato ripe?"), Day 4 explores a separate, supporting capability: **Demand Forecasting**.

By answering the question "What demand might we expect in the future?", we can provide valuable information for supply planning, reducing waste, and coordinating harvesting schedules with expected market demand. The sequence models developed here do NOT directly control the robotic arm; rather, they represent a complementary system for future farm management.

## 2. Dataset Description
- **Source**: `tomato_retail_sales_daily_2009_2022 (1).csv`
- **Characteristics**: A real historical dataset of tomato retail sales covering 2009 to 2022 (over 5,000 daily observations).
- **Target Variable**: `Quantity Sold (kg)`

## 3. Data Preprocessing & Leakage Prevention
To prevent data leakage, the dataset was strictly split chronologically without random shuffling:
- **Past (2009-2018)**: Training Set
- **Recent Past (2019-2020)**: Validation Set
- **Future (2021-2022)**: Test Set

Demand values were scaled using `MinMaxScaler` fit exclusively on the training set.

## 4. Modeling Approach
We implemented four models to evaluate the effectiveness of deep sequence modeling against simple heuristic baselines:

1. **Naive Forecast**: Assumes tomorrow's demand will be exactly the same as today's.
2. **Moving Average (7-Day)**: Assumes tomorrow's demand will follow the average of the last 7 days.
3. **Standard LSTM**: Learns temporal patterns from the previous 30 days to predict the 31st day.
4. **LSTM + Attention**: Learns temporal patterns while dynamically assigning importance to specific historical timesteps, improving context and interpretability.

## 5. Model Evaluation Results
*Results are based on Test Set (2021-2022) evaluation. See the executed Jupyter Notebook for exact numerical outputs.*

- The **Naive** and **Moving Average** models serve as baselines.
- The **Standard LSTM** successfully learned complex temporal patterns, outperforming simple moving averages in tracking highly volatile daily demand.
- The **LSTM + Attention** model matched or slightly exceeded the Standard LSTM performance by selectively attending to the most relevant historical days, proving the value of attention mechanisms in sequence data.

## 6. AgroMind Connection & Future Integration
Currently, the CNN (Day 3) and LSTM (Day 4) operate independently. In a future integrated system:
1. The **LSTM** forecasts a spike in tomato demand.
2. The farm management system advises an increase in harvesting operations.
3. The **Robotic Arm** uses the **CNN** to selectively pick ripe tomatoes to meet the forecasted demand, optimizing supply and minimizing crop waste.

## 7. Limitations & Conclusion
- **Univariate Forecasting**: The models currently predict future demand based only on past demand. Incorporating external factors like price, weather, or holidays would likely improve accuracy.
- **Horizon**: Forecasting multiple days or weeks ahead (multi-step forecasting) would provide more actionable lead time for physical harvesting operations.

**Conclusion**: Deep sequence models can effectively extract temporal patterns from historical sales data, proving their viability as a supporting tool for the AgroMind agricultural automation ecosystem.
