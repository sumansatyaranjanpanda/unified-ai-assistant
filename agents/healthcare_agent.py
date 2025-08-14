import streamlit as st
import pandas as pd
import os
import pickle

from utils.healthcare_data import get_symptom_details, get_disease_precautions, get_description
from models.ml_models.healthcare_predictor import predict_disease
from utils.dl_predictor import predict_disease_dl
from rag.healthcare_qa import ask_healthcare

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load LLM and Prompt
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)

prompt = ChatPromptTemplate.from_template("""
    Provide a list of the 5 best doctors in {city} for treating {disease}. 
Include:
- Doctor Name
- Experience
- Consultation Fee (Approximate)
- Rating (from Practo or user reviews)
- Contact or Clinic Info (if known)
""")

llm_chain = prompt | llm | StrOutputParser()

# Load symptom list
SYMPTOMS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "symptom_list.pkl"))
with open(SYMPTOMS_PATH, "rb") as f:
    all_symptoms = pickle.load(f)

def run_healthcare_agent():
    st.set_page_config(page_title="Healthcare Assistant", page_icon="🏥")
    st.header("🏥 Healthcare Assistant")

    tab1, tab2 = st.tabs(["🔬 Symptom Predictor", "📚 Medical Q&A"])

    with tab1:
        mode = st.radio("Choose prediction mode:", ["ML (multiselect)", "DL (free-text)"])
        if mode == "ML (multiselect)":
            st.markdown("### Predict Disease from Symptoms (ML Model)")
            symptoms = st.multiselect("Select Symptoms", options=all_symptoms)

            if st.button("Predict Disease"):
                if not symptoms:
                    st.warning("Please select at least one symptom.")
                else:
                    disease = predict_disease(symptoms)
                    st.session_state["predicted_disease"] = disease
                    st.success(f"**Predicted Disease:** {disease}")

                    # Symptom details
                    st.markdown("#### Symptom Details")
                    details = get_symptom_details(symptoms)
                    st.table(pd.DataFrame(details))

                    # Description
                    st.markdown("#### Description of Disease")
                    desc_map = get_description()
                    res = desc_map.get(disease.strip().lower(), "No description available.")
                    st.write(f"**Description:** {res}")

                    # Precautions
                    st.markdown("#### Recommended Precautions")
                    for p in get_disease_precautions(disease):
                        st.write(f"- {p}")
            
         # Doctor suggestion (based on predicted disease)
            if "predicted_disease" in st.session_state: 
                disease = st.session_state["predicted_disease"]

                enter_city = st.text_input("Enter your city to find top doctors:")

                if enter_city:
                    with st.spinner("🔍 Finding top doctors in your city..."):
                        try:
                            doctors = llm_chain.invoke({"disease": disease, "city": enter_city})
                            st.markdown("### 🩺 Top 5 Doctors in Your City")
                            st.write(doctors)
                            st.session_state.clear()
                        except Exception as e:
                            st.error("Sorry, couldn't fetch doctor recommendations.")
                            st.exception(e)
            
            else:
                st.info("Predict a disease first to get doctor recommendations.")
            
        # If using DL model
        elif mode == "DL (free-text)":
            st.markdown("### Predict Disease from Symptoms (DL Model)")
           
            ## Voice input placeholder (will implement browser-based later)
            st.info("🎙️ Voice input temporarily disabled for deployment. Browser-based voice coming soon!")

            text_input = st.text_area("Type your symptoms here:")    

            if st.button("Predict Disease"):
                if not text_input:
                    st.warning("Please enter a symptom description.")
                else:
                    with st.spinner("🔍 Predicting disease..."):
                        predictions = predict_disease_dl(text_input)
                        if predictions:
                            st.session_state["predicted_disease_dl"] = predictions[0][0]
                            st.success(f"**Predicted Disease:** {predictions[0][0]}")

                            # Display top predictions
                            st.markdown("#### Top Predictions")
                            for disease, prob in predictions:
                                st.write(f"- {disease}: {prob * 100:.2f}%")
                        else:
                            st.error("No disease predicted. Please try again.")
            
            #Doctor suggestion (based on predicted disease)
            if "predicted_disease_dl" in st.session_state:
                disease = st.session_state["predicted_disease_dl"]

                enter_city = st.text_input("Enter your city to find top doctors:")

                if enter_city:
                    with st.spinner("🔍 Finding top doctors in your city..."):
                        try:
                            doctors = llm_chain.invoke({"disease": disease, "city": enter_city})
                            st.markdown("### 🩺 Top 5 Doctors in Your City")
                            st.write(doctors)
                            st.session_state.clear()
                            
                        except Exception as e:
                            st.error("Sorry, couldn't fetch doctor recommendations.")
                            st.exception(e)

    with tab2:
        st.markdown("### Ask anything about diseases, treatments, or symptoms")
        question = st.text_input("Your question:")
        if question:
            with st.spinner("🔍 Searching medical knowledge…"):
                answer = ask_healthcare(question)
            st.markdown(f"**Answer:** {answer}")