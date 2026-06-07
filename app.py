import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import os

st.set_page_config(
    page_title="Rekomendasi Tanaman",
    page_icon="🌾",
    layout="wide"
)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    model  = joblib.load(os.path.join(BASE_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    with open(os.path.join(BASE_DIR, "metadata.json")) as f:
        meta = json.load(f)
    return model, scaler, meta

model, scaler, meta = load_artifacts()

st.title("🌾 Prediksi Rekomendasi Tanaman")
st.markdown("Sistem rekomendasi tanaman berbasis **Machine Learning** berdasarkan kondisi kesuburan tanah dan iklim.")
st.divider()

with st.sidebar:
    st.header("Informasi Model")
    st.metric("Model Terbaik",  meta["best_model_name"])
    st.metric("Test Accuracy",  str(round(meta["test_accuracy"] * 100, 2)) + "%")
    st.metric("F1-Score Macro", str(round(meta["f1_macro"] * 100, 2)) + "%")
    st.metric("Jumlah Kelas",   str(meta["n_classes"]) + " tanaman")
    st.divider()
    st.caption("Dataset: Crop Recommender - Kaggle")

tab1, tab2 = st.tabs(["Prediksi Manual", "Prediksi Upload CSV"])

with tab1:
    st.subheader("Masukkan Parameter Tanah dan Iklim")
    feature_cols = meta["feature_columns"]
    cols = st.columns(3)
    input_vals = {}
    for i, feat in enumerate(feature_cols):
        with cols[i % 3]:
            input_vals[feat] = st.number_input(
                label=feat,
                value=0.0,
                format="%.4f",
                key="input_" + feat
            )
    st.divider()
    if st.button("Prediksi Tanaman", type="primary", use_container_width=True):
        input_df    = pd.DataFrame([input_vals])
        input_final = scaler.transform(input_df) if meta["best_model_scaled"] else input_df.values
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
    feature_info = ", ".join(meta["feature_columns"])
    st.info("Kolom yang dibutuhkan: " + feature_info)
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.write("Data:", df_up.shape[0], "baris x", df_up.shape[1], "kolom")
        st.dataframe(df_up.head())
        if st.button("Prediksi Semua Baris", type="primary"):
            feat_cols = [c for c in meta["feature_columns"] if c in df_up.columns]
            X_up      = df_up[feat_cols]
            X_final   = scaler.transform(X_up) if meta["best_model_scaled"] else X_up.values
            preds     = model.predict(X_final)
            probas    = model.predict_proba(X_final)
            class_names = meta["class_names"]
            df_up["Rekomendasi"] = [class_names[int(p)] for p in preds]
            df_up["Confidence"]  = [str(round(float(np.max(pr)) * 100, 1)) + "%" for pr in probas]
            st.success("Prediksi selesai untuk " + str(len(df_up)) + " baris")
            st.dataframe(df_up)
            st.download_button(
                label="Download Hasil",
                data=df_up.to_csv(index=False),
                file_name="hasil_rekomendasi.csv",
                mime="text/csv"
            )

