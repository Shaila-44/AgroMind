import streamlit as st
import requests
from PIL import Image
import matplotlib.pyplot as plt

# Configuration
FASTAPI_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="Tomato Ripeness Predictor",
    page_icon="🍅",
    layout="centered"
)

st.title("🍅 Smart Agriculture: Tomato Ripeness Predictor")
st.write("Upload an image of a tomato to determine if it is **Ripe** or **Unripe**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Process the image when the user clicks the button
    if st.button("Predict Ripeness", type="primary"):
        with st.spinner("Analyzing image..."):
            try:
                # Prepare the payload for FastAPI
                # Reset file pointer
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.read(), uploaded_file.type)}
                
                # Make the POST request to the API
                response = requests.post(FASTAPI_URL, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display the results
                    st.success("Prediction complete!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label="Predicted Class", value=result["predicted_class"])
                    with col2:
                        st.metric(label="Confidence", value=f"{result['confidence'] * 100:.2f}%")
                    
                    # Display probability bar chart
                    st.subheader("Class Probabilities")
                    probs = result["probabilities"]
                    
                    # Create a simple bar chart using matplotlib
                    fig, ax = plt.subplots(figsize=(6, 2.5))
                    classes = list(probs.keys())
                    probabilities = [probs[c] for c in classes]
                    colors = ['red' if c == 'Ripe' else 'green' for c in classes]
                    
                    bars = ax.barh(classes, probabilities, color=colors, alpha=0.8)
                    ax.set_xlim(0, 1)
                    ax.set_xlabel("Probability")
                    
                    # Remove borders and ticks for a cleaner look
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.xaxis.set_ticks_position('bottom')
                    ax.yaxis.set_ticks_position('none')
                    
                    # Annotate bars
                    for i, v in enumerate(probabilities):
                        ax.text(v + 0.02, i, f"{v*100:.1f}%", va='center', fontweight='bold')
                        
                    st.pyplot(fig)
                    
                    st.divider()
                    
                    # Day 5 Integration: Harvest Recommendation
                    st.subheader("Harvest Recommendation")
                    if result["predicted_class"] == "Ripe":
                        st.success("🍅 **Tomato is ready for harvesting.**")
                    else:
                        st.warning("🌱 **Tomato is not ready for harvesting.**")
                        
                    # Day 5 Integration: Supporting Demand Forecast (From Day 4)
                    st.divider()
                    st.subheader("Supporting Information (Day 4 Forecasting)")
                    st.info(f"📈 **Forecasted tomato demand:** {result.get('demand_forecast_kg', 'N/A')} kg")
                    st.caption("Note: Demand forecasting helps with supply and harvest planning. It does not control the robotic arm or affect ripeness predictions.")
                    
                else:
                    st.error(f"Error from API: {response.status_code} - {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("⚠️ Failed to connect to the backend API. Please ensure the FastAPI server is running on port 8000.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

# Day 6 Integration: Natural Language Interface
st.divider()
st.header("💬 AgroMind Natural Language Command")
st.write("Enter a command to test the NLP understanding module (e.g., *'Harvest the ripe tomatoes in greenhouse 2'*).")

nlp_command = st.text_input("Enter command:")
if st.button("Parse Command"):
    if nlp_command:
        # Import the NLP parser dynamically or at module level
        try:
            from nlp import parse_command
            parsed_result = parse_command(nlp_command)
            
            st.subheader("Interpreted Command")
            st.caption("This is a prototype interface. It does not physically control the robot.")
            
            # Display neatly
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Intent:**")
                st.info(parsed_result.get("intent", "UNKNOWN"))
            with col2:
                st.write("**Entities Extracted:**")
                for key, val in parsed_result.items():
                    if key != "intent":
                        st.write(f"- **{key.capitalize()}:** {val}")
                        
            st.write("**Structured JSON Output:**")
            st.json(parsed_result)
            
        except ImportError:
            st.error("NLP module not found. Ensure nlp.py exists in the project root.")
    else:
        st.warning("Please enter a command.")
