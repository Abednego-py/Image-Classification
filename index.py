import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from joblib import load
from PIL import Image, ImageOps
from image_classification import teachable_machine_classification
import streamlit as st


clf = load('image_class.joblib')
sc = StandardScaler()
class_names = ['aeroplane', 'car', 'bird']





st.title("Image Classification")
st.text("Upload an image to classify if it's an aeroplane, bird or car")


uploaded_file = st.file_uploader("Choose an image file ...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = teachable_machine_classification(image, clf)
    # if label == 0:
    #     st.write("The MRI scan has a brain tumor")
    # else:
    #     st.write("The MRI scan is healthy")
    st.write(label)

# def main():
#     pass
     
# if __name__=='__main__':
#     main()

