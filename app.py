import streamlit as st
import pandas as pd
from admet_ai import ADMETModel

st.title("Chemical Property Predictor")

@st.cache_resource
def load_model():
    return ADMETModel()

model = load_model()

tab1, tab2 = st.tabs(["Single SMILES", "Upload CSV"])

with tab1:
    smiles = st.text_input("Enter SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")
    if st.button("Predict", key="single"):
        preds = model.predict(smiles=smiles)
        st.dataframe({"Property": list(preds.keys()), "Value": list(preds.values())})

with tab2:
    file = st.file_uploader("Upload a CSV file with a 'smiles' column", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        if "smiles" not in df.columns:
            st.error("Your CSV must have a column named 'smiles'.")
        elif len(df) > 200:
            st.error("Please upload at most 200 molecules at a time.")
        else:
            if st.button("Predict all", key="batch"):
                with st.spinner("Predicting..."):
                    preds = model.predict(smiles=df["smiles"].tolist())
                    result = pd.concat(
                        [df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1
                    )
                st.dataframe(result)
                st.download_button(
                    "Download results (CSV)",
                    result.to_csv(index=False),
                    "predictions.csv",
                    "text/csv",
                )
