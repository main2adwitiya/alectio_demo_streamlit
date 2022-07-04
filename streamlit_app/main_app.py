import streamlit as st

st.title('Image classification Streamlit ')
from modules import training
from modules import viz
from modules import inference
PAGES = {
    
    "Model Training": training,
    "Inference": inference,
    "Visualizations":viz
}
st.sidebar.title('Demo for Snowflake')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()