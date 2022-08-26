
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
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
import torch
from torchvision.models import resnet18
import streamlit_drawable_canvas as st_canvas
from PIL import Image
device = "cuda" if torch.cuda.is_available() else "cpu"

device='cpu'
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

model = resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 8)






def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (256, 256, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]
)

def load_model(type_model):
    model = resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 8)
    
    model.load_state_dict(torch.load(f'data/{type_model}/last.pt',map_location=torch.device('cpu')))
    return model


def app():
    import streamlit as st
    from streamlit_drawable_canvas import st_canvas
    class_to_idx = {
    '0': 'book',
    '1': 'bicycle',
    '2': 'airplane',
    '3': 'banana',
    '4': 'The Eiffel Tower',
    '5': 'apple',
    '6': 'backpack',
    '7': 'bird',
}

    st.title('Prediction app')
    query_strategies = ('Confidence', 'Entropy', 'Margin', 'Random')
    option = st.selectbox(
     'Select a Querying Srategy',query_strategies)
    
    model = load_model(option)
    
    
    st.write(f'Currently loaded model is with QS {option}')
    print(model)


    
    
    uploaded_file = st.file_uploader("Choose an image to upload...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.')
        batch_t = torch.unsqueeze(transform(image), 0)
        model.eval()
        out = model(batch_t)
        prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
        max_prob=torch.max(prob)
        indices = torch.argmax(prob)
        
        idx=indices.item()
        metrics=f'{max_prob:.3f}'
        
        st.header(f'Ummm I guess it is a {class_to_idx[str(idx)]}')
        st.metric(label="Confidence", value=metrics)
        
        #_, indices = torch.sort(out, descending=True)
        #print([(prob[idx].item()) for idx in indices[0][:8]])
        #st.write(f'Probability is : {[(prob[idx].item()) for idx in indices[0][:8]]}')
        
    st.write('You can doodle as well, try drawing one of the following things below')
    class_list=['The Eiffel Tower','airplane','apple', 'backpack','banana','bicycle','bird','book']
    

    s = ''

    for i in class_list:
        s += "- " + i + "\n"

    st.markdown(s)
    canvas_result_1 = st_canvas(
      # Fixed fill color with some opacity
      fill_color="rgba(255, 165, 0, 0.3)",
      stroke_width=3,
      stroke_color="#000000",
      background_color="#FFFFFF",
      update_streamlit=True,
      height=500,
      width=500,
      drawing_mode="freedraw",
      key="canvas1",
    )
    
    if canvas_result_1.image_data is not None:
        im = Image.fromarray(canvas_result_1.image_data).convert('RGB')
        im.save('sample.jpg', "JPEG")
    
    
    if st.button('Predict'):
        st.write('Hold up')
        image=Image.open('sample.jpg')
        #st.image(image, caption='Uploaded Image.', use_column_width=True)
        batch_t = torch.unsqueeze(transform(image), 0)
        model.eval()
        out = model(batch_t)
        prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
        max_prob=torch.max(prob)
        indices = torch.argmax(prob)
        
        idx=indices.item()
        metrics=f'{max_prob:.3f}'
        
        st.header(f'Ummm I guess it is a {class_to_idx[str(idx)]}')
        st.metric(label="Confidence", value=metrics)
    
        
    
    
    
    
    
    
  