import streamlit as st

def inject_custom_css():
    st.markdown("""
        <style>
        /* === GLOBAL RESET === */
        .stApp, .stRadio, .stCheckbox, .stTextInput, .stTextArea {
            --primary-color: #7c3aed;
            --background-color: #0a041a;
            --secondary-background: #110a2e;
            --text-color: #e2e8f0;
            font-family: 'Segoe UI', sans-serif;
        }

        /* === MAIN BACKGROUND === */
        .stApp {
            background: linear-gradient(135deg, var(--background-color) 0%, #110a2e 50%, #1a103d 100%) !important;
            color: var(--text-color) !important;
        }

        /* === RADIO BUTTON FIX === */
        .stRadio [role="radiogroup"] {
            background: var(--secondary-background) !important;
            border: 1px solid #4c1d95 !important;
            border-radius: 8px !important;
            padding: 10px !important;
            gap: 12px !important;
        }

        .stRadio [role="radio"] {
            border: 2px solid #5b21b6 !important;
            background: var(--secondary-background) !important;
        }

        .stRadio [role="radio"][aria-checked="true"] {
            background: var(--primary-color) !important;
            border-color: var(--primary-color) !important;
        }

        .stRadio [role="radio"][aria-checked="true"]::after {
            background: white !important;
        }

        .stRadio [role="radio"] + span {
            color: var(--text-color) !important;
        }

        /* === SIDEBAR STYLING === */
        section[data-testid="stSidebar"] {
            background: linear-gradient(160deg, #0f0c29, #1a103d) !important;
            border-right: 1px solid #5b21b6 !important;
        }

        /* === INPUT FIELDS === */
        .stTextInput input, .stTextArea textarea {
            background: var(--secondary-background) !important;
            color: var(--text-color) !important;
            border: 1px solid #4c1d95 !important;
        }

        /* === BUTTONS === */
        button[kind="primary"] {
            background: var(--primary-color) !important;
            color: white !important;
        }

        /* === TEXT ELEMENTS === */
        h1, h2, h3, h4, h5, h6, p, div, span {
            color: var(--text-color) !important;
        }

        /* === WHITE BOX REMOVAL === */
        div[data-testid="stHorizontalBlock"],
        div[data-testid="stVerticalBlock"] {
            background: transparent !important;
        }
        </style>
    """, unsafe_allow_html=True)