import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import warnings
from skimage import transform
warnings.filterwarnings("ignore")

@st.cache
def load_model(model_path):
    model=load_model(model_path)
    return model

def transform_image(image_array,size):
    resized_img=transform.resize(img_array,size)
    expanded_img=np.expand_dims(resized_img,axis=0)
    return expanded_img

def predict_class(model,image):
    pred=model.predict_proba(image)
    p1=np.argmax(pred)
    mapper={0:"No Crack",1:"Crack present"}
    p=mapper[p1]
    return pred,p,p1



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    st.title("Crack Classification in concrete")
    ### Load the model
    # with st.spinner('Model is being loaded..'):
    #     model=load_model(model_path)

    ### upload the image
    image_file=st.file_uploader("Upload the image that you want to classify",type=['jpg','png','jpeg'])
    if image_file is not None:
        model_path=""
        ### Load the uploaded image and convert to array

        image = Image.open(image_file).convert('RGB')
        img_array = np.array(image)
        st.image(image_file,use_column_width=True)
        req_size=(120,120)

        ### Transformt the img to be fed to the model
        transformed_img=transform_image(img_array,req_size)
        st.write(transformed_img.shape)
        st.write(img_array.shape)

        ### Predict the classes
        # p,prediction,p1=predict_class(transformed_img,model)

    else:
        st.write("Please upload an image!")