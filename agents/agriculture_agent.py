import streamlit as st
from rag.agriculture_qa import ask_agriculture


def run_agriculture_agent():

    st.subheader("agriculture Assistant")
    question = st.text_input("Your question:")
    if question:
        with st.spinner("🔍 Searching agriculture knowledge…"):
            answer =ask_agriculture(question)
        st.markdown(f"**Answer:** {answer}")