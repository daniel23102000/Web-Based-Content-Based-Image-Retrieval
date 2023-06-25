import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import tensorflow
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
import os

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

uploaded_file = st.file_uploader("Choose an Image")
if uploaded_file is not None:
           if save_uploaded_file(uploaded_file):
           # display the file
            # display_image = Image.open(uploaded_file)
            # st.image(display_image)
            # feature extract
            features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
            # st.text(features)
            # recommendention
            indices = recommend(features,feature_list)
            search_result = searchproduct(filenames[indices[0][0]])
            for i in search_result:
               productnameone = i[0]

            search_result = searchproduct(filenames[indices[0][1]])
            for i in search_result:
               productnametwo = i[0]

            search_result = searchproduct(filenames[indices[0][2]])
            for i in search_result:
               productnamethree = i[0]
            
            search_result = searchproduct(filenames[indices[0][3]])
            for i in search_result:
               productnamefour = i[0]

            search_result = searchproduct(filenames[indices[0][4]])
            for i in search_result:
               productnamefive = i[0]
            task = st.selectbox("Task", ["Recommendation", productnameone, productnametwo, 
            productnamethree, productnamefour, productnamefive])

            if task == "Recommendation":
            # show
               col1,col2,col3,col4,col5 = st.columns(5)

               with col1:
                  st.image(filenames[indices[0][0]])
                  st.write(productnameone)
               with col2:
                  st.image(filenames[indices[0][1]])
                  st.write(productnametwo)
               with col3:
                  st.image(filenames[indices[0][2]])
                  st.write(productnamethree)
               with col4:
                  st.image(filenames[indices[0][3]])
                  st.write(productnamefour)
               with col5:
                  st.image(filenames[indices[0][4]])
                  st.write(productnamefive)
            
            elif task == productnameone:
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
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

            elif task == productnametwo:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenames[indices[0][1]])
               with col2:
                  search_result = searchproduct(filenames[indices[0][1]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[5]
                  st.image(gambarlokasione)
               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

            elif task == productnamethree:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenames[indices[0][2]])
               with col2:
                  search_result = searchproduct(filenames[indices[0][2]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[5]
                  st.image(gambarlokasione)
               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

            elif task == productnamefour:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenames[indices[0][3]])
               with col2:
                  search_result = searchproduct(filenames[indices[0][3]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[5]
                  st.image(gambarlokasione)
               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

            elif task == productnamefive:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenames[indices[0][4]])
               with col2:
                  search_result = searchproduct(filenames[indices[0][4]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[5]
                  st.image(gambarlokasione)
               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

else:
    st.header("Not Found")