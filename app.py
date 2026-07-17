
import streamlit as st 
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Defect Detection App",page_icon="⚙️",layout="wide",initial_sidebar_state="expanded")

with st.sidebar:
    option=option_menu(menu_title="Choisir un modèle",options=["CNN Model","EfficientNet B0 Model","Performances des modèles"],icons=["cpu","robot","clipboard-data"],menu_icon="cast",default_index=0,orientation="vertical")

img_size=224
mean=np.array([0.485, 0.456, 0.406], dtype=np.float32)
std=np.array([0.229, 0.224, 0.225], dtype=np.float32)


@st.cache_resource
def load_cnn_model():
    return load_model("models/model_cnn_aug.keras")
cnn_model =load_cnn_model()

@st.cache_resource
def load_efficientnet_model():
    return load_model("models/efficientnet_model.keras")
efficientnet_model =load_efficientnet_model()

def preprocess_image(image):
 # convertir l'image en RGB 
 image=image.convert("RGB")

 #redimensionnement
 image=image.resize((img_size, img_size))

 #conversion en matrice
 image=np.array(image).astype(np.float32)

 image=image/255

 #normalisation
 image=(image-mean)/std

 image= np.expand_dims(image, axis=0)

 return image

st.title("Détection de Défauts sur les Impellers par Deep Learning")
st.divider()

if option == "CNN Model":
 st.write("Importez une image d'un Impeller pour le classer: Normal Ou Défectueux ")
 img = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
 if img is not None:
     image = Image.open(img)
     c1,c2,c3=st.columns(3)
     with c2 :
      st.image(image, use_container_width=True)
    
     preprocessed_image = preprocess_image(image)
     prediction = cnn_model.predict(preprocessed_image, verbose=0)
     predicted_class = np.argmax(prediction)
     percent = prediction[0][predicted_class]

     if predicted_class == 1:
            st.success(f"Cette pièce est classée comme : Normale avec une probabilité de {percent:.2%}")
     else:
            st.error(f"Cette pièce est classée comme : Défectueuse avec une probabilité de {percent:.2%}")


elif option == "EfficientNet B0 Model":
         st.write("Importez une image d'un Impeller pour le classer: Normal Ou Défectueux ")
         img = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
         if img is not None:
             image = Image.open(img)
             c1,c2,c3=st.columns(3)
             with c2 :
                 st.image(image, use_container_width=True)
    
             preprocessed_image = preprocess_image(image)
             prediction = efficientnet_model.predict(preprocessed_image, verbose=0)
             score = prediction[0][0]
             if score > 0.5:
                 st.error(f"Cette pièce est classée comme : Défectueuse avec une probabilité de {score:.2%}")
             else:
                 st.success(f"Cette pièce est classée comme : Normale avec une probabilité de {(1 - score):.2%}")


elif  option == "Performances des modèles":
        st.subheader("🔹Courbes d'apprentissage du modèle CNN sans augmentation des données :")
        st.image("Courbes/CNN.png")

        st.divider()

        st.subheader("🔹Courbes d'apprentissage du modèle CNN (avec augmentation des données) :")
        st.image("Courbes/CNN augmenté.png")

        st.divider()

        st.subheader("🔹Courbes d'apprentissage du modèle EfficientNet B0 :")
        st.image("Courbes/EfficientNetB0.png")


        