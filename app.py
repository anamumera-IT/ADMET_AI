import os
os.environ["TQDM_DISABLE"] = "1"

import io
import contextlib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from admet_ai import ADMETModel

st.title("Chemical Property Predictor")

SUFFIX = "_drugbank_approved_percentile"


@st.cache_resource
def load_model():
    m = ADMETModel()
    orig = m.predict

    def quiet(*args, **kwargs):
        with contextlib.redirect_stderr(io.StringIO()):
            return orig(*args, **kwargs)

    m.predict = quiet
    return m


model = load_model()

tab1, tab2 = st.tabs(["Single SMILES", "Upload CSV"])

with tab1:
    smiles = st.text_input("Enter SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")
    if st.button("Predict", key="single"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error("Invalid SMILES")
        else:
            preds = model.predict(smiles=smiles)

            axes = [
                ("BBB_Martins", "BBB", False),
                ("hERG", "hERG Safe", True),
                ("ClinTox", "Non-Toxic", True),
                ("Solubility_AqSolDB" + SUFFIX, "Soluble", False),
                ("Bioavailability_Ma", "Bioavailable", False),
            ]
            labels, vals = [], []
            for key, label, invert in axes:
                if key in preds:
                    v = float(preds[key]) * (1 if "percentile" in key else 100)
                    vals.append(100 - v if invert else v)
                    labels.append(label)

            if vals:
                angles = np.linspace(0, 2 * np.pi, len(vals), endpoint=False).tolist()
                fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
                ax.plot(angles + angles[:1], vals + vals[:1], color="red")
                ax.fill(angles + angles[:1], vals + vals[:1], color="red", alpha=0.25)
                ax.set_xticks(angles)
                ax.set_xticklabels(labels)
                ax.set_ylim(0, 100)

                c1, c2 = st.columns(2)
                c1.pyplot(fig)
                c2.image(Draw.MolToImage(mol, size=(400, 400)))
            else:
                st.image(Draw.MolToImage(mol, size=(400, 400)))

            rows = []
            for k, v in preds.items():
                if not k.endswith(SUFFIX):
                    rows.append({
                        "Property": k,
                        "Value": round(float(v), 3),
                        "DrugBank percentile (%)": round(
                            float(preds.get(k + SUFFIX, float("nan"))), 1
                        ),
                    })
            st.dataframe(pd.DataFrame(rows))

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
                    preds = preds.reset_index(drop=True)
                    old = df.reset_index(drop=True)
                    old = old[[c for c in old.columns if c not in preds.columns]]
                    result = pd.concat([old, preds], axis=1)
                st.dataframe(result)
                st.download_button(
                    "Download results (CSV)",
                    result.to_csv(index=False),
                    "predictions.csv",
                    "text/csv",
                )
