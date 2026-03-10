import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from groq import Groq

# --- Page Config ---
st.set_page_config(page_title="ParaDetect AI", page_icon="🔬")

# --- Initialize Groq (Optional Assistant) ---
# Set your API key in Streamlit secrets or environment variables
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def get_groq_assistant_response(prediction):
    prompt = f"The AI model diagnosed a cell image as {prediction} for malaria. Provide a brief, professional explanation of what this means and common next steps in a clinical setting."
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

# --- Model Loading ---
@st.cache_resource
def load_malaria_model():
    # Replace with the path to your actual trained .h5 file
    model = tf.keras.models.load_model('malaria_model.h5')
    return model

# --- Main UI ---
st.title("🔬 ParaDetect AI")
st.markdown("### Deep Learning-Based Malaria Diagnosis")
st.write("Upload a microscopic image of a blood cell to detect parasites.")

uploaded_file = st.file_uploader("Choose a cell image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Cell Image', use_container_width=True)
    
    with st.spinner('Analyzing...'):
        # Preprocessing (Adjust dimensions based on your specific model)
        img = image.resize((128, 128)) 
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Prediction
        model = load_malaria_model()
        prediction = model.predict(img_array)
        
        # Result Logic (Assuming 0 = Parasitized, 1 = Uninfected)
        result = "Parasitized" if prediction[0][0] < 0.5 else "Uninfected"
        confidence = (1 - prediction[0][0]) if result == "Parasitized" else prediction[0][0]
        
    # --- Display Results ---
    if result == "Parasitized":
        st.error(f"Prediction: {result}")
    else:
        st.success(f"Prediction: {result}")
        
    st.write(f"**Confidence Level:** {confidence:.2%}")

    # --- Groq Analysis ---
    if st.button("Get Detailed AI Analysis"):
        with st.spinner("Consulting Groq Medical Assistant..."):
            ai_analysis = get_groq_assistant_response(result)
            st.info(ai_analysis)

st.sidebar.info("Disclaimer: This tool is for educational purposes and should not replace professional medical advice.")