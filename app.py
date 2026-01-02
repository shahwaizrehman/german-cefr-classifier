import streamlit as st
import joblib
import os
import pickle
import numpy as np
import pandas as pd

# --- CONFIGURATION ---
MODEL_DIR = "cefr_german_model"
MODEL_FILENAME = "model.pkl" 
VECTORIZER_FILENAME = "vectorizer.pkl"
LABELS_FILENAME = "labels.pkl"

# --- PAGE SETUP ---
st.set_page_config(page_title="German CEFR Predictor", page_icon="🇩🇪")

LEVEL_INFO = [
    {"Level": "A1", "Description": "Beginner: Basic phrases and everyday expressions."},
    {"Level": "A2", "Description": "Elementary: Frequently used expressions."},
    {"Level": "B1", "Description": "Intermediate: Travel and work-related situations."},
    {"Level": "B2", "Description": "Upper Intermediate: Fluent discussion."},
    {"Level": "C1", "Description": "Advanced: Demanding academic/scientific texts."},
]

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
if "history" not in st.session_state:
    st.session_state["history"] = []

st.caption("Predict CEFR levels for German text. Use the single-text form or upload a CSV for batch scoring.")
st.table(pd.DataFrame(LEVEL_INFO))

text_input = st.text_area("Enter German Text:", height=200, placeholder="Guten Tag!")

if st.button("Predict Level"):
    if text_input.strip():
        try:
            # 1. Vectorize text
            numerical_data = vectorizer.transform([text_input])
            
            # 2. Predict raw result
            prediction = model.predict(numerical_data)[0]
            
            # 3. Safe Mapping to Label
            if isinstance(prediction, (np.integer, int)):
                result_label = labels[int(prediction)]
            else:
                result_label = str(prediction)

            # 4. Optional confidence
            confidence = None
            if hasattr(model, "predict_proba"):
                try:
                    probas = model.predict_proba(numerical_data)[0]
                    confidence = float(np.max(probas))
                except Exception:
                    confidence = None

            # 5. Display Results
            st.divider()
            
            if "A" in result_label:
                color = "#2ECC71" # Green
            elif "B" in result_label:
                color = "#F39C12" # Orange
            else:
                color = "#E74C3C" # Red
                
            st.markdown(f"Predicted Level: <h1 style='color:{color};'>{result_label}</h1>", unsafe_allow_html=True)
            if confidence is not None:
                st.write(f"Confidence: **{confidence:.2f}**")
            
            descriptions = {
                "A1": "Beginner: Basic phrases and everyday expressions.",
                "A2": "Elementary: Frequently used expressions.",
                "B1": "Intermediate: Travel and work-related situations.",
                "B2": "Upper Intermediate: Degree of fluency in discussion.",
                "C1": "Advanced: Demanding, longer academic/scientific texts."
            }
            st.info(descriptions.get(result_label, "Level detected."))

            # 6. Log history for quick review
            st.session_state["history"].append(
                {
                    "Preview": text_input[:120] + ("…" if len(text_input) > 120 else ""),
                    "Predicted": result_label,
                    "Confidence": confidence,
                }
            )
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.warning("Please enter some text.")

if st.session_state["history"]:
    st.subheader("Recent predictions")
    st.dataframe(pd.DataFrame(st.session_state["history"]))

# --- DATASET & MODEL STATISTICS SECTION ---
st.divider()
st.subheader("📊 Dataset & Model Statistics")

@st.cache_data
def load_csv_stats():
    """Load CSV and calculate CEFR level distribution."""
    df = pd.read_csv("cefr_final_merged.csv")
    cefr_counts = df["CEFR"].value_counts().sort_index()
    total_rows = len(df)
    return cefr_counts, total_rows

try:
    cefr_counts, total_rows = load_csv_stats()
    
    # Create a displayable table with CEFR counts
    stats_data = pd.DataFrame({
        "CEFR Level": cefr_counts.index,
        "Number of Rows": cefr_counts.values
    }).reset_index(drop=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📈 Total Rows", total_rows)
    with col2:
        st.metric("🎯 Model Accuracy", "79.12%")
    with col3:
        st.metric("📚 CEFR Levels", len(cefr_counts))
    
    st.write("**CEFR Level Distribution:**")
    st.dataframe(stats_data, use_container_width=True, hide_index=True)
    
except Exception as e:
    st.error(f"Error loading dataset statistics: {e}")