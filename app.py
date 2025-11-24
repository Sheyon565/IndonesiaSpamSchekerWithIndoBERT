import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer & model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("indoBERT-sms")
    model = AutoModelForSequenceClassification.from_pretrained("indoBERT-sms")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
model.to(device)

label_map = {
    0: "Private Message",
    1: "Spam",
    2: "Real Advertisement"
}

st.title("IndoBERT SMS Classifier")
st.write("A text classification demo using IndoBERT for Indonesian SMS.")

# Input text
user_input = st.text_area("Masukkan teks SMS untuk diklasifikasi:")


if st.button("Predict"):
    if len(user_input.strip()) == 0:
        st.warning("Input tidak boleh kosong.")
    else:
        # Tokenize
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs).item()

        st.subheader("Hasil Klasifikasi:")
        st.success(f"Label: **{label_map[pred]}**")

        # Show probabilities
        st.subheader("Probabilities:")
        st.json({
            label_map[i]: float(probs[0][i])
            for i in range(3)
        })
