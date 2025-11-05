# ESMloc
A lightweight, alignment-free, ESM-based protein localization predictor inspired by DeepLoc.
# ESMLoc

ESMLoc is a lightweight protein subcellular localization predictor that uses ESM-2 transformer embeddings and a simple neural network classifier. The model predicts multi-label localization output for 11 major cellular compartments.

This tool is inspired by the DeepLoc framework but does not require multiple sequence alignments or evolutionary features. Instead, it uses pre-trained protein language model representations (ESM2) for fast and alignment-free prediction.

## Features
- Alignment-free prediction (no BLAST, no MSAs)
- Uses ESM2 embeddings from the `esm2_t6_8M_UR50D` model
- Predicts multiple possible localization sites
- Works on CPU (embedding may take a few seconds)
- Deployable through Streamlit

## Input
A single protein sequence using standard amino acid codes.

## Output
- A list of predicted localizations above a confidence threshold
- Per-class probability scores

## Requirements
Python 3.8+
torch
esm
streamlit
numpy

Running Locally
arduino
Copy code
streamlit run app.py
Model Files
The following files must be in the application directory:

Copy code
deeploc_cnn.pt
label_columns.pkl
Citation
If using this tool for research, please cite the original DeepLoc publication:
Almagro Armenteros et al., "DeepLoc: prediction of protein subcellular localization using deep learning." Bioinformatics (2017).
