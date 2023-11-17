#tout les functions and class besoin pour l'app 
import os
import streamlit as st
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input
import numpy as np
#from keras.preprocessing.image import ImageDataGenerator
#list de function qui traite des objets python 

def treatment_premodel(image_path):
    img = image.load_img(image_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

#parameters of predict
#rescale= 1/255
#batch_size=

def model_prediction(treated_image):
    # Load model 
    model_path="model_20epoch_recall.h5"
    model = load_model(model_path)
    
    # Prediction
    prediction = model.predict(treated_image)
    
    # Get the index with the highest probability
    predicted_class_index = np.argmax(prediction)
    
    # Map string class indices to numeric class labels
    class_indices = {
        0: 'AnnualCrop', 1: 'Forest', 2: 'HerbaceousVegetation',
        3: 'Highway', 4: 'Industrial', 5: 'Pasture',
        6: 'PermanentCrop', 7: 'Residential', 8: 'River',
        9: 'SeaLake'
    }
    
    # Get the predicted class label
    predicted_class_label = class_indices[predicted_class_index]

    # Get the predicted probability for the predicted class
    predict_probability = prediction[0, predicted_class_index]
    
    # Print the results
    print(f"Predicted Class: {predicted_class_label}")
    print(f"Predicted Probability: {predict_probability*100:.2f}%") 
    return predict_probability, predicted_class_label