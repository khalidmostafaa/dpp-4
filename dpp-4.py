import streamlit as st
import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
import requests
import os

def download_from_google_drive(file_id, destination):
    """
    Download a large file from Google Drive by handling the confirmation token.
    """
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()
    
    response = session.get(url, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"confirm": token}
        response = session.get(url, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    """
    Extract confirmation token from the response cookies.
    """
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None

def save_response_content(response, destination):
    """
    Save the response content to a file.
    """
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

# File ID and destination path
file_id = "1cFh_gt5lsDoZWSAwRuxMeLMRkTtht3xk"
destination = "DPP4_model.pkl"

# Download the model if not already present
if not os.path.exists(destination):
    st.info("Downloading the model from Google Drive...")
    try:
        download_from_google_drive(file_id, destination)
        st.success("Model downloaded successfully!")
    except Exception as e:
        st.error(f"Error downloading the model: {e}")

# Load the model
if os.path.exists(destination):
    try:
        with open(destination, 'rb') as file:
            model = pickle.load(file)
        st.success("Model loaded successfully!")
    except pickle.UnpicklingError:
        st.error("The downloaded file is not a valid pickle file. Double-check the source file.")
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
else:
    st.error("Failed to download the model. Please check the file ID or link.")
    
# Function to generate PubChem-like fingerprints
def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return list(fingerprint)

# Set the overall style
st.markdown(
    """
    <style>
    .intro-text { 
        font-size: 14px; 
        color: #2c3e50;
        background-color: #ecf0f1; 
        padding: 15px;
        border-radius: 8px; 
    }
    .disclaimer {
        font-size: 14px;
        color: #e74c3c;
        margin-top: 20px;
        padding: 10px;
        border-top: 1px solid #bdc3c7;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app interface
st.title("DPP-4 Inhibitor Predictor")

# Intro section
st.markdown(
    """
    <div class="intro-text">
        DPP-4 inhibitors are a class of oral medications used to manage type 2 diabetes.
        This model predicts the potency (pIC50) of molecules against the DPP-4 enzyme, aiding in the discovery and development of novel drugs for type 2 diabetes.
    </div>
    """,
    unsafe_allow_html=True
)

# SMILES input
smiles_input = st.text_input("Enter the canonical SMILES:", placeholder="Example: CC[C@H](C)[C@H](N)C(=O)N1CCCC1")

if st.button("Predict"):
    if smiles_input:
        fingerprint = get_fingerprint(smiles_input)

        if fingerprint:
            # Convert fingerprint to DataFrame and ensure all columns are strings
            fingerprint_df = pd.DataFrame([fingerprint], columns=[f'PubchemFP{i}' for i in range(2048)])
            fingerprint_df.columns = fingerprint_df.columns.astype(str)

            # Prediction
            try:
                prediction = model.predict(fingerprint_df)[0]
                st.write(f"Predicted pIC50: {prediction:.2f}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.error("Invalid SMILES string. Please enter a valid SMILES.")
    else:
        st.warning("Please enter a SMILES string.")

# Disclaimer section
st.markdown(
    """
    <div class="disclaimer">
        <strong>Disclaimer:</strong> This model is intended for research purposes only and should not be used for medical treatments or diagnoses.
    </div>
    """,
    unsafe_allow_html=True
)
