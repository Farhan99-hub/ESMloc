import streamlit as st
from predictor import load_model, predict_localization

model, labels = load_model()

st.title("DeepLoc Subcellular Localization Predictor")

sequence = st.text_area("Enter protein sequence:")
threshold = st.slider("Prediction threshold", 0.1, 0.9, 0.5)

if st.button("Predict"):
    if sequence.strip():
        pred, scores = predict_localization(model, labels, sequence, threshold)
        st.write("Predicted locations:")
        st.write(pred if pred else "No class above threshold.")
        st.write("Class probabilities:")
        st.dataframe(
            [{"Localization": k, "Probability": round(v, 3)} for k, v in scores.items()]
        )
    else:
        st.write("Enter a valid sequence.")
