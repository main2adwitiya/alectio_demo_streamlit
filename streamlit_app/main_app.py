import streamlit as st

st.title('AL on Quickdraw Dataset')

from viz_modules import viz
import inference
PAGES = {
    
    "Visualizations":viz,
    "Inference": inference
    
}
st.sidebar.title('Alectio\'s Active Learning')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()



