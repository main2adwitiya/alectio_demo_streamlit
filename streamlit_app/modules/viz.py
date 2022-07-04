
# viz.py
import streamlit as st

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)
import json
def app():
    st.title('Dataset Information')
    st.header('Plant village dataset')
    st.write('In this data-set, 38 different classes of plant leaf and background images are available.  The data-set containing 61,486 images. We used six different augmentation techniques for increasing the data-set size. The techniques are image flipping, Gamma correction, noise injection, PCA color augmentation, rotation, and Scaling.')
    st.write('No of training images used for each class -800 that is 30400 images ')
    st.write('No of validation images used for each class -150 that is 5700 images')
    st.write('No of testing images used for each class -50 that is 1900 images')

    st.header('Models tried for training and testing')
    st.subheader(f'1. Resnet-50(Finetuned)')
    #st.subheader(f'2.Densenet-121(Finetuned)')
   # st.subheader(f'3.Custom PlDisease-NeT')

    st.write('The Resnet-50 model showed a training accuracy of 81% while testing accuracy of 75%')

    #st.write('The Densenet-121 model showed a training accuracy of 92% while testing accuracy of 60%, clearly overfitting on our dataset')
    #st.write('The PlDisease-NeT model showed a training accuracy of 99% while testing accuracy of 96.7%, showing one of the best results over existing models. Initially model was very unstable while training in first few epochs due to validation data being unbalanced interms of leaves but gradullay it got very stable which can be seen in the chart below.')

    

#    st.pyplot()


  #  st.header('The architecture of final model is below')
 #   st.image(image, caption='Custom cnn based model named PlDisease-NeT',clamp=True)