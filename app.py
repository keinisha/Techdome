import numpy as np
import streamlit as st
import pickle
from prediction import predict



def main():
    st.title("Dog Emotion Predictor")

    file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png", "jpeg"])
    st.set_option('deprecation.showfileUploaderEncoding', False)
    html_temp = """
    <div style="background-color:rgb(32,32,32);padding:10px">
    <h2 style="color:rgb(128,128,128);text-align:center;">Dog Emotion Predictor</h2>
    </div>
    """

if __name__=='__main__':
    main()
