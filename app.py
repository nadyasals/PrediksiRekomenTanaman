import streamlit as st
import numpy as np
import pandas as pd
import joblib, json, os

st.set_page_config(
    page_title="Rekomendasi Tanaman",
    page_icon="🌾",
    layout="wide"
)

# Path absolut agar tidak error di Streamlit Cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

@st.cache_resource
def load_artifacts():
    model   = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    scaler  = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    imputer = joblib.load(os.path.join(MODEL_DIR, "imputer.pkl"))
    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        meta = json.load(f)
    return model, scaler, imputer, meta

model, scaler, imputer, meta = load_artifacts()

st.title("🌾 Prediksi Rekomendasi Tanaman")
st.markdown("Sistem rekomendasi tanaman berbasis **Machine Learning** berdasarkan kondisi kesuburan tanah dan iklim.")
st.divider()

with st.sidebar:
    st.header("Informasi Model")
    st.metric("Model Terbaik",  meta["best_model_name"])
    st.metric("Test Accuracy",  str(round(meta["test_accuracy"]*100, 2)) + "%")
    st.metric("F1-Score Macro", str(round(meta["f1_macro"]*100, 2)) + "%")
    st.metric("Jumlah Kelas",   str(meta["n_classes"]) + " tanaman")
    st.divider()
    st.caption("Dataset: Crop Recommender - Kaggle")

tab1, tab2 = st.tabs(["Prediksi Manual", "Prediksi Upload CSV"])

with tab1:
    st.subheader("Masukkan Parameter Tanah & Iklim")
    feature_cols = meta["feature_columns"]
    cols = st.columns(3)
    input_vals = {}
    for i, feat in enumerate(feature_cols):
        with cols[i % 3]:
            input_vals[feat] = st.number_input(feat, value=0.0, format="%.4f", key=feat)
    st.divider()
    if st.button("Prediksi Tanaman", type="primary", use_container_width=True):
        input_df    = pd.DataFrame([input_vals])
        input_imp   = imputer.transform(input_df)
        input_final = scaler.transform(input_imp) if meta["best_model_scaled"] else input_imp
        pred        = model.predict(input_final)[0]
        proba       = model.predict_proba(input_final)[0]
        class_names = meta["class_names"]
        pred_label  = class_names[int(pred)]
        confidence  = float(np.max(proba)) * 100
        st.success("Tanaman yang direkomendasikan: **" + pred_label.upper() + "**")
        st.info("Confidence: " + str(round(confidence, 1)) + "%")
        st.markdown("**Probabilitas Top 5 Tanaman:**")
        top5_idx   = np.argsort(proba)[::-1][:5]
        top5_crops = [class_names[i] for i in top5_idx]
        top5_proba = [proba[i] for i in top5_idx]
        prob_df    = pd.DataFrame({"Tanaman": top5_crops, "Probabilitas": top5_proba})
        st.bar_chart(prob_df.set_index("Tanaman"))

with tab2:
    st.subheader("Upload File CSV")
    st.info("Kolom yang dibutuhkan: " + str(meta["
