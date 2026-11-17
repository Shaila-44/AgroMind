# Smart Agriculture for Automated Crop Harvesting

## Overview
This project is part of a larger Smart Agriculture system designed to automate crop harvesting using computer vision. The core objective is to develop a machine learning pipeline that visually classifies tomatoes as either **Ripe** or **Unripe**. We have explored Traditional Machine Learning algorithms (Day 1 & 2) and Deep Learning using Convolutional Neural Networks (Day 3) for this core perception task. 

Additionally, we explored a supporting capability in Day 4: **Demand Forecasting**. Using LSTM and Attention sequence models on historical sales data, we demonstrate how demand forecasting can complement the robotic harvesting system by aiding in harvest scheduling and supply planning. 

In **Day 5 (Integration)**, we have deployed the best-performing CNN model using a robust backend (FastAPI) and an interactive web frontend (Streamlit). The application simulates the conceptual pipeline: predicting visual ripeness, making a harvest recommendation, and optionally displaying the day's forecasted demand.

Finally, in **Day 6 (NLP)**, we introduced a Natural Language Interface. Using a lightweight, rule-based NLP module, operators can issue plain-text commands (e.g., "Harvest ripe tomatoes in greenhouse 2") which are parsed into structured JSON intents. This serves as an interaction layer and does not replace the core computer vision system.

## Project Structure
```text
AgroMind/
├── app.py                 # Streamlit frontend application
├── api.py                 # FastAPI backend for model inference
├── utils.py               # Image preprocessing utilities
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
├── model/
│   └── best_model.h5      # Saved CNN model for inference
├── Dataset/
│   └── detection/images/  # Tomato dataset
├── uploads/               # Directory for uploaded images
├── nlp.py                 # Day 6 rule-based NLP command parser
├── evaluate_nlp.py        # Evaluation script for the NLP module
├── notebooks/             # Jupyter notebooks for model training and analysis
    ├── ML_Baseline.ipynb
    ├── Day2_Ensemble_Hyperparameter_Tuning.ipynb
    ├── Day3_CNN_Classification.ipynb
    └── Day4_LSTM_Attention_Demand_Forecasting.ipynb
├── Day4_Model_Evaluation_Report.md
├── Day5_Integration_Report.md
└── Day6_NLP_Report.md
```

## Installation
1. Clone this repository or open the project folder.
2. Ensure you have Python 3.8+ installed.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Deployment

### 1. Start the FastAPI Backend
Open a terminal and run the following command to start the FastAPI server on port 8000:
```bash
uvicorn api:app --reload
```
You can verify the API is running by visiting `http://localhost:8000/` in your browser. The API documentation is available at `http://localhost:8000/docs`.

### 2. Start the Streamlit Frontend
Open a new terminal window and run the Streamlit app:
```bash
streamlit run app.py
```
This will open the interactive web interface in your default browser (usually at `http://localhost:8501`).

## Usage
1. Open the Streamlit web interface.
2. Click on "Browse files" to upload an image of a tomato (JPG/PNG).
3. Click the **Predict Ripeness** button.
4. The system will send the image to the FastAPI backend, process it, and return the predicted class (Ripe or Unripe) along with the confidence score and a probability breakdown.

## Example API Request (Optional)
If you wish to test the API endpoint directly without the frontend:
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@path_to_your_tomato_image.jpg;type=image/jpeg'
```

## Future Work (Upcoming Phases)
- Transfer Learning (ResNet, EfficientNet, MobileNet) for performance enhancement.
- Object Detection (YOLO) for identifying and locating multiple tomatoes in complex greenhouse environments.
- Segmentation to identify the exact boundaries of tomatoes.
- Generative AI for synthetic data generation.
- Agentic AI integrations for Smart Agriculture automation.
