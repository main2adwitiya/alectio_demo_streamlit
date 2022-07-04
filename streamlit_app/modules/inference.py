
# Prediction app.py
import streamlit as st

import os
import pandas as pd 
from PIL import Image
import numpy as np
from skimage import transform
import os
import pandas as pd
import glob
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np

global model
#model=load_model('plant_main2.h5')
#model._name = "PlDisease-Net"

import json
# with open ('label_map.txt','r') as file:
#     f=file.read()
# f = f.replace("\'", "\"")
# label_map = json.loads(f)

def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (256, 256, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

def app():

    st.title('Prediction app')
    #st.title(f"Currently loaded model is {model._name }")
    #uploaded_file = st.file_uploader("Choose an image to upload...", type="jpg")
    # if uploaded_file is not None:
    # 	image = load(uploaded_file)
    # 	prob=model.predict(image)
    # 	prob1 = np.amax(prob)
    # 	prob1=prob1-0.04
    # 	#prob1=(format(prob1,".3f")
    # 	predictions=np.argmax(prob, axis=-1)
    # 	print(int(predictions))
    # 	predictions_id=[]
    # 	predictions_class=[]
    # 	for key, value in label_map.items():
    # 		if int(predictions) == value:
    # 			predictions_id.append(value)
    # 			predictions_class.append(key)
    # 			print(value,key)

    # 	#image = Image.open(uploaded_file)
    # 	class_str=str(predictions_class[0].replace('_',' '))
    # 	st.image(image, caption='Uploaded Image.', use_column_width=True)
    # 	st.write("")
    # 	st.write("Inference result")
    	
    # 	st.write(f'Probability is : {prob1}')
    # 	st.write(f'Prediction class is : {class_str}')
    # 	st.write(f'Prediction id is : {predictions_id[0]}')