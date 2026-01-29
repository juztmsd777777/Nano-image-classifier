import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image

def load_model():
    model = MobileNetV2(weights='imagenet')
    return model

def prepocess_image(image):
    img = np.array(image)
    img = cv2.resize(img,(224,224))
    img = preprocess_input(img)
    img = np.expand_dims(img,axis=0)
    return img

def image_prediction(model, image):
    try:
        img = prepocess_image(image)
        pred =model.predict(img)
        decode_prediction = decode_predictions(pred,top=3)[0]
        return decode_prediction
    except Exception as e:
        print(f"Error in image prediction:, str{e}")
        return None

def main():
    st.set_page_config(page_title="nano image classifier",page_icon="frame_with_picture",layout="centered")
    st.title("Nano Image Classifier")
    st.write("Upload an image and find out what it is!")
    
    @st.cache_resource
    def load_model_cache():
        model = load_model()
        return model
    
    model = load_model_cache()
    
    uploaded_file =st.file_uploader("Choose an image..",type=["jpg","png"])
    
    if uploaded_file is not None:
        st.image(
            uploaded_file,
            caption="Uploaded Image",
            use_container_width=True
        )   
        btn =st.button("Predict image")
        
        if btn:
            pill_image = Image.open(uploaded_file)
            with st.spinner("Classifying..."):
                predictions = image_prediction(model,pill_image)
                
            
            
            if predictions:
                st.subheader("Predictions:")
                for _,label,score in predictions:
                    st.write(f"**{label}** : {score:.2%}")
                    
if __name__ == "__main__":
    main()
                
            
    
    