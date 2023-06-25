import pickle
import tensorflow
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
import streamlit as st
import os
import glob
from streamlit_option_menu import option_menu

# DB Management
import sqlite3
conn = sqlite3.connect('data.db')
c = conn.cursor()

def create_producttable():
  c.execute('CREATE TABLE IF NOT EXISTS prostabletiga(productname TEXT,price TEXT,Lokasi TEXT,gambarproduct TEXT,gambarproductdua TEXT,gambarlokasi TEXT)')

def view_all_product():
  c.execute('SELECT * FROM prostabletiga')
  data = c.fetchall()
  return data

def searchproduct(gambarproduct):
  c.execute("SELECT * FROM prostabletiga WHERE gambarproduct='%s'" % (gambarproduct))
  data = c.fetchall()
  return data

with st.sidebar:
    selected = option_menu(
        menu_title="",
        options=["About", "Contact"]
    )

feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
 ])

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

st.title('Fashion Finder System')
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
      if save_uploaded_file(uploaded_file):
           # display the file
         #   display_image = Image.open(uploaded_file)
         #   st.image(display_image)
           # feature extract
           features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
         #   st.text(features)
           # recommendention
           indices = recommend(features,feature_list)
           search_result = searchproduct(filenames[indices[0][0]])
           for i in search_result:
            productnameone = i[0]
            priceone = i[1]
            lokasione = i[2]

            col1,col2,col3 = st.columns(3)
            with col1:
               st.image(filenames[indices[0][0]])
            with col2:
               search_result = searchproduct(filenames[indices[0][0]])

               for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[5]
               st.image(gambarlokasione)
            with col3:
               st.write("Product Name : ", productnameone)
               st.write("Price : ",priceone)
               st.write("Locate : ",lokasione)

st.subheader("List of Product")
image_files = glob.glob("images/*.jpg")
manuscripts = []

for image_file in image_files:
       image_file = image_file.replace("\\","/")
       parts = image_file.split("/")
       if parts[1] not in manuscripts:
          manuscripts.append(parts[1])

manuscripts.sort()
   # st.write(manuscripts)
view_manuscripts_images = manuscripts   
n = st.number_input("Select Grid Width", 1, 10, 5)
view_images = []

for image_file in image_files:
      if any(manuscripts in image_file for manuscripts in view_manuscripts_images):
         view_images.append(image_file)
         
groups = []

for i in range(0, len(view_images), n):
      groups.append(view_images[i:i+n])

for group in groups:
      cols = st.columns(n)
      for i, image_file in enumerate(group):
         cols[i].image(image_file)
         # cols[i].write(image_file)