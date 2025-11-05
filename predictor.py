import torch
import pickle
import numpy as np
from model import ESMClassifier
from embedding import get_esm_embedding

def load_model():
    with open("label_columns.pkl", "rb") as f:
        labels = pickle.load(f)
    model = ESMClassifier(num_labels=len(labels))
    model.load_state_dict(torch.load("deeploc_cnn.pt", map_location="cpu"))
    model.eval()
    return model, labels

def predict_localization(model, labels, sequence, threshold=0.5):
    emb = get_esm_embedding(sequence)
    with torch.no_grad():
        logits = model(emb)
        probs = torch.sigmoid(logits).numpy().flatten()

    result = {label: float(p) for label, p in zip(labels, probs)}
    predicted = [l for l, p in result.items() if p >= threshold]
    return predicted, result
