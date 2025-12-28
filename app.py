import streamlit as st
import joblib
import os
import pickle
import numpy as np

# --- CONFIGURATION ---
MODEL_DIR = "cefr_german_model"
MODEL_FILENAME = "model.pkl" 
VECTORIZER_FILENAME = "vectorizer.pkl"
LABELS_FILENAME = "labels.pkl"

# --- PAGE SETUP ---
st.set_page_config(page_title="German CEFR Predictor", page_icon="🇩🇪")

# --- LOADING ASSETS ---
@st.cache_resource
def load_assets():
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    vec_path = os.path.join(MODEL_DIR, VECTORIZER_FILENAME)
    labels_path = os.path.join(MODEL_DIR, LABELS_FILENAME)
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    
    with open(labels_path, 'rb') as f:
        labels = pickle.load(f)
        
    return model, vectorizer, labels

try:
    model, vectorizer, labels = load_assets()
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# --- USER INTERFACE ---
st.title("🇩🇪 German CEFR Classifier")

text_input = st.text_area("Enter German Text:", height=200, placeholder="Guten Tag!")

if st.button("Predict Level"):
    if text_input.strip():
        try:
            # 1. Vectorize text
            numerical_data = vectorizer.transform([text_input])
            
            # 2. Predict raw result
            prediction = model.predict(numerical_data)[0]
            
            # 3. Safe Mapping to Label
            # This handles 'numpy.int64' and string outputs correctly
            if isinstance(prediction, (np.integer, int)):
                result_label = labels[int(prediction)]
            else:
                result_label = str(prediction)

            # 4. Display Results
            st.divider()
            
            # Determine color based on string content
            if "A" in result_label:
                color = "#2ECC71" # Green
            elif "B" in result_label:
                color = "#F39C12" # Orange
            else:
                color = "#E74C3C" # Red
                
            st.markdown(f"Predicted Level: <h1 style='color:{color};'>{result_label}</h1>", unsafe_allow_html=True)
            
            descriptions = {
                "A1": "Beginner: Basic phrases and everyday expressions.",
                "A2": "Elementary: Frequently used expressions.",
                "B1": "Intermediate: Travel and work-related situations.",
                "B2": "Upper Intermediate: Degree of fluency in discussion.",
                "C1": "Advanced: Demanding, longer academic/scientific texts."
            }
            st.info(descriptions.get(result_label, "Level detected."))
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.warning("Please enter some text.")