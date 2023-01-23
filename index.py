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


uploaded_file = st.file_uploader("Choose an image file ...", type=["jpg", "jpeg", "png", "webp"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = teachable_machine_classification(image, clf)
    if label == 0:
        st.write("The image is aeroplane")
    elif label == 1:
        st.write("The image is a car")
    elif label == 2:
        st.write("The image is a bird")
    else:
        st.write("The image cannot be classified")
    st.write(label)

# def main():
#     pass
     
# if __name__=='__main__':
#     main()

