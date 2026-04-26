# =========================
# STREAMLIT APP
# =========================

import streamlit as st
from common import generate_summary, load_model_and_tokenizer


@st.cache_resource
def get_summarizer_components():
    return load_model_and_tokenizer()


tokenizer, model, device = get_summarizer_components()

# UI Title
st.title("📝 Text Summarizer (Transformer-Based)")

# Input box
text = st.text_area("Enter your article:")

# Button
if st.button("Summarize"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        summary = generate_summary(
            text=text,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )

        # Display result
        st.subheader("Summary:")
        st.write(summary)