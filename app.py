import streamlit as st
from utils import *

#list de choses a faire  #task list
st.divider()
#header image
header_image="input_app_design/satellite_image.jpg"
st.image(header_image,caption='Image source=forplayday',use_column_width=True)
#header of app 
st.title("Satellite images classificator")
text = """
<div style="text-align: justify;">
    The satellite image classifier is inspired by the open access EuroSat dataset.
    <br>
    Composed with in total 27,000 labeled and geo-referenced images.
    <br>
    This challenge aims to classify 10 different classes of LuLc screening in the Sentinel -2 satellite images in RGB frequency bands encoding, applying a customized and pretrained RESNET50 model (code link below).
    <br>
    LuLc classes: {0: 'AnnualCrop', 1: 'Forest', 2: 'HerbaceousVegetation', 3: 'Highway',
    4: 'Industrial', 5: 'Pasture', 6: 'PermanentCrop', 7: 'Residential', 8: 'River', 9: 'SeaLake'}.
    <br>
    The metric used for performance evaluation was Recall, and the library to design the architecture of the model was Tensorflow and Keras.
</div>
"""
st.markdown(text, unsafe_allow_html=True)

#link to the model training code 
st.write("Resnet50 model training code [link](https://github.com/phelber/EuroSAT)")

st.divider()
st.subheader('Choose an image to test the pretrained RESNET50 model 🚀')
#1.choise image from folder_path and display
folder_path="test_images"
# Let the user choose an image from a list of the folder_path
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
choice_user = st.selectbox("Select an image to test the App:", image_files)
st.divider()

#display the chosen image in the app 
st.subheader('1.Display chosen Image')
st.subheader('Great!! this is how looks your image.. Lets see what land cover could be 🤔')
image_path = os.path.join(folder_path, choice_user)
st.image(image_path,use_column_width=True) #adjust width in case at the end, take to much space
#st.write(image_path)
st.divider()

#2.Traitment to chosen image to model.predict()
st.subheader('2.chosen image traitment to prediction')
treated_image=treatment_premodel(image_path)
#st.write(treated_image)

#3. prediction 
st.subheader('3. prediction')
st.spinner("the prediction is running...🏋️‍♀️")
predict_probability, predicted_class_label = model_prediction(treated_image)
st.write(f"the prediction to the chosen image is 😎:Predicted Class: {predicted_class_label},Predicted Probability: {predict_probability*100:.2f}%", unsafe_allow_html=True, 
         style={'border': '2px solid #4CAF50', 'padding': '10px', 'border-radius': '10px'})

st.divider()

#4. Model Performance 
st.subheader('4. Model Performance ')
st.write(f"The model performance was measure mainly by Recall, because of number of classes and also accurancy display below")
#confusion matrix 
st.subheader('Confusion Matrix')
cm_path="evaluation_graphs/Recall_eval/cm_recall_20epochs_final.png"
st.image(cm_path,use_column_width=True)
st.write('''The confusion matrix shows how in particular: Forest,Residential and Sea-Leak are the best detected and classified 
         by the pretrained and customized Resnet50 model''')

st.subheader('Accuracy plot')
permorfance_path="evaluation_graphs/accurancy_eval/performance_20epochs.png"
st.image(permorfance_path,use_column_width=True)
st.write('''The performance plot shows an optimal increment of the accurancy to test and train dataset ina range between 85-92%,respectively.
        On  the inverse tendence, the performance of Loss decreased with the sequence of epochs. expected behaviour to check it good performance
         in terms of accurancy''')

#5.Sources and Bibliography:
st.subheader('4.Sources and Bibliography')
st.write("1.Kaggle Competition ideas[link](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)")
st.write("2.Dataset EuroSAT [link](https://github.com/phelber/EuroSAT)")
st.divider()

#6. Acknowledgements
st.subheader('5. Acknowledgements')
st.write('''I would like to thank Region Ile de France and School of Data for your support. I would also appreciate all the support 
         and effort by professors and instructors. Lastly, but not, least important, I would like to thank my Husband Sammy and 
         my colleages and friends: ALice, Zara,Karim and ALi for the studentship and emotional 
         support that allowed me to conduct well this hurry and demanding challenge''')
st.divider()

#7. Logos 
logo_artefact="input_app_design/artefact.png"
logo_ilefrance="input_app_design/ilefrance.jpg"

# Display logos in the same row
st.image(logo_artefact, caption='Logo Artefact', use_column_width=True)
st.image(logo_ilefrance, caption='Logo Ile-de-France', use_column_width=True)