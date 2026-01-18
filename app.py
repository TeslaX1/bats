import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="ML Model Demo",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Sentence Similarity App")
st.write("Compute cosine similarity between two sentences using a transformer model.")

# ----------------------------
# Load model (cached)
# ----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ----------------------------
# User inputs
# ----------------------------
sentence1 = st.text_area("Enter first sentence", height=100)
sentence2 = st.text_area("Enter second sentence", height=100)

# ----------------------------
# Button
# ----------------------------
if st.button("Compute Similarity"):
    if sentence1.strip() == "" or sentence2.strip() == "":
        st.warning("Please enter both sentences.")
    else:
        with st.spinner("Computing similarity..."):
            embeddings = model.encode(
                [sentence1, sentence2],
                convert_to_tensor=True
            )

            similarity = util.cos_sim(
                embeddings[0],
                embeddings[1]
            )

        st.success("Similarity computed!")
        st.metric(
            label="Cosine Similarity",
            value=f"{similarity.item():.4f}"
        )
