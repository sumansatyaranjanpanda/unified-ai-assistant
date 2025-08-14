import os, sys
# Ensure the repo root is on PYTHONPATH, so `import app…` works
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import streamlit as st

from app.router import router_to_agent

st.set_page_config(page_title="Unified AI Assistant", layout="wide")

st.title("Unified AI Assistant")    
st.markdown("Select a domain to get started.")

# Sidebar to select domain
domain = st.sidebar.selectbox("Choose Domain", ["Healthcare", "Agriculture", "Finance"])


# Route to appropriate agent
router_to_agent(domain)