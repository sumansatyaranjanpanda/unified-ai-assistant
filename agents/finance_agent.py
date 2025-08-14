import streamlit as st


def run_finance_agent():


    st.subheader("Finance Assistant")
    st.markdown("Ask me anything about personal finance, investments, or budgeting.")

    user_input = st.text_input("Ask your finance question:")

    if user_input:
        with st.spinner("Fetching answer..."):
            response = f"Processing your finance question: {user_input}"
        st.success(response)