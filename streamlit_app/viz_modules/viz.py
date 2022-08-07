
# viz.py
import streamlit as st

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from viz_modules.helper_viz import *
import os
st.set_option('deprecation.showPyplotGlobalUse', False)
import json
def app():
    
    st.header('Visualizations')
    st.markdown("***")

    comaprison_fig=comparison_function()
    st.plotly_chart(comaprison_fig, use_container_width=False, sharing="streamlit")
    label= 'Changing the loops'
    query_strategies = ('Confidence', 'Entropy', 'Margin', 'Random')
    st.markdown("***")
    st.header('Interactive viz for each QS')
    option = st.selectbox(
     'Select a Querying Srategy',
     query_strategies)
    loop_number=st.slider(label, min_value=0, max_value=9, value=None, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False)
    
    
    data = get_desired_data(option,loop_number)
    
    cf_fig=get_plot_confusion_matrix(data,loop_number)
    st.plotly_chart(cf_fig, use_container_width=False, sharing="streamlit")
    st.markdown("***")
    am=accuracy_metrics(data,loop_number)
    st.plotly_chart(am, use_container_width=False, sharing="streamlit")
    st.markdown("***") 
    acc_p=acc_per_class(data,loop_number)
    f1_p=f1_score_per_class(data,loop_number)
    ppc=precision_per_class(data,loop_number)
    
    st.plotly_chart(acc_p, use_container_width=False, sharing="streamlit")
    st.markdown("***")
    st.plotly_chart(f1_p, use_container_width=False, sharing="streamlit")
    st.markdown("***")
    st.plotly_chart(ppc, use_container_width=False, sharing="streamlit")
    
    

