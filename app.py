import numpy as np
import streamlit as st
import pickle

pickle_in = open(r'efficientnet.pickle', 'rb') # download pickle file add path here
efficientnet = pickle.load(pickle_in) # add pickle file

import cv2
from PIL import Image, ImageOps

def import_and_predict(image_data,model):
    size =(128,128)
    image = ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction


def main():
    st.title("Dog Emotion Predictor")

    file = st.file_uploader("Please upload an image of a dog", type={'jpg','png','jpeg'})
    html_temp = """
    <div style="background-color:rgb(32,32,32);padding:10px">
    <h2 style="color:rgb(128,128,128);text-align:center;">Dog Emotion Predictor</h2>
    </div>
    """
    if file is None:
        st.text("please upload an image")
    else:
        image=Image.open(file)
        st.image(image, use_column_width=True)
        predictions = import_and_predict(image,efficientnet)
        class_names=['angry','sad','relaxed','happy']
        string="The dog is most likely "+class_names[np.argmax(predictions)]
        st.success(string)

if __name__=='__main__':
    main()
