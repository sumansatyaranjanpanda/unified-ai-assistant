# streamlit_app.py - Root level launcher for Streamlit Cloud
# This file just imports and runs your existing app/main.py

import os
import sys

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import and run your existing main app
try:
    ## Import your main app file
    import app.main
except ImportError as e:
    import streamlit as st
    st.error(f"Failed to import app.main: {e}")
    st.info("Make sure all required dependencies are installed and files are in correct locations.")