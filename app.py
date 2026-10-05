import streamlit as st
from admet_ai import ADMETModel

st.title("Chemical Property Predictor")

@st.cache_resource
def load_model():
    return ADMETModel()

model = load_model()
smiles = st.text_input("Enter SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")

if st.button("Predict"):
    preds = model.predict(smiles=smiles)
    st.dataframe({"Property": list(preds.keys()), "Value": list(preds.values())})
