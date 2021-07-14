from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import os
import cv2
import pydicom
from keras.models import Sequential
from keras.applications import Xception
from PIL import Image
from keras.models import Model, load_model
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
import streamlit as st
import matplotlib.pyplot as plt

showpred = 0
st.sidebar.title("About")

st.sidebar.info("Brain Hemorrhage Detector will detect the presence and type of hemorrhage in a CT scan using Deep Learning")

st.sidebar.title("Brain Hemorrhage Detector")

st.title("Hemorrhage Detector and Classifier")


def apply_window(image, center, width):
    image = image.copy()
    min_value = center - width // 2
    max_value = center + width // 2
    image[image < min_value] = min_value
    image[image > max_value] = max_value
    return image


def apply_window_policy(image):

    image1 = apply_window(image, 40, 80) # brain
    image2 = apply_window(image, 80, 200) # subdural
    image3 = apply_window(image, 40, 380) # bone
    image1 = (image1 - 0) / 80
    image2 = (image2 - (-20)) / 200
    image3 = (image3 - (-150)) / 380
    image = np.array([
        image1 - image1.mean(),
        image2 - image2.mean(),
        image3 - image3.mean(),
    ]).transpose(1,2,0)

    return image

def save_and_resize(dcm):
    image = get_pixels_hu(dcm)
    image = apply_window_policy(image[0])
    image -= image.min((0,1))
    image = (255*image).astype(np.uint8)
    image = cv2.resize(image, (299, 299)) #smaller
    #res = cv2.imwrite(image)
    return image

def get_pixels_hu(scan): 
    image = np.stack([scan.pixel_array])
    image = image.astype(np.int16) 
    
    image[image == -2000] = 0
    
    intercept = scan.RescaleIntercept
    slope = scan.RescaleSlope
    
    if slope != 1: 
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
    
    image += np.int16(intercept) 
    
    return np.array(image, dtype=np.int16)

def create_model():    
    base_model = Xception(weights = 'imagenet', include_top = False, input_shape = (299,299,3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.15)(x)
    y_pred = Dense(6, activation = 'sigmoid')(x)

    return Model(inputs = base_model.input, outputs = y_pred)

def get_model():
    model = create_model()
    model.load_weights("effnetb4.h5")
    return model

def prediction(image, model):
    #image = save_and_resize(image)
    image = np.reshape(image, [1, 299, 299, 3])
    prediction = model.predict(image)
    return prediction

uploaded_file = st.file_uploader("Choose an image...", type="dcm")
if uploaded_file is not None and st.sidebar.button('Diagnose'):
    model = get_model()
    ds = pydicom.dcmread(uploaded_file)
    image = Image.fromarray(ds.pixel_array)
    #st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    out_list = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
    image = save_and_resize(ds)
    st.image(image, caption='Preprocessed Scan', use_column_width=True, width=220)
    label = prediction(image, model)
    probability = np.amax(label)
    typ = np.argmax(label)
    st.write('Type : %s, Probability (%.2f%%)' % (out_list[typ], probability*100))