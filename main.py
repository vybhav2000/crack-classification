import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras import models
import warnings
import pandas as pd
from skimage import transform
import base64
warnings.filterwarnings("ignore")

# @st.cache
def load_model(model_path):
    model=models.load_model(model_path)
    return model

def transform_image(image_array,size):
    resized_img=transform.resize(img_array,size)
    expanded_img=np.expand_dims(resized_img,axis=0)
    return expanded_img

def predict_class(image,model):
    pred=model.predict(image)
    if pred>0.5:
        p1=1
    else:
        p1=0
    mapper={0:"No Crack present",1:"Crack present"}
    p=mapper[p1]
    return pred,p,p1

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    add_bg_from_local('photo.jpg')
    st.title("SURFACE CRACK DETECTION")
    ### Load the model
    model_path="test_model1.h5"
    with st.spinner('Model is being loaded..'):
        model=load_model(model_path)

    ### upload the image
    # st.subheader("Please upload an image in the specified format")
    image_file=st.file_uploader("",type=['jpg','png','jpeg'])
    if image_file is not None:
        model_path=""
        ### Load the uploaded image and convert to array

        image = Image.open(image_file).convert('RGB')
        img_array = np.array(image)

        st.subheader("The Uploaded Image")
        st.image(image.resize((300,300)))
        req_size=(120,120)

        ### Transformt the img to be fed to the model
        with st.spinner("Predicting the class for the image"):
            transformed_img=transform_image(img_array,req_size)

            ### Predict the classes
            p,prediction,p1=predict_class(transformed_img,model)
            st.subheader("Prediction")

            df=pd.DataFrame({"Prediction":[prediction],"Probability for crack":np.round_(p,2)[0]})
            st.dataframe(df)

    else:
        st.warning("Please upload an image to continue!")
        