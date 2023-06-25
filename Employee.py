import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
import streamlit as st
import glob
import pickle

# DB Management
import sqlite3
conn = sqlite3.connect('data.db')
c = conn.cursor()

def create_usertable():
  c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')

def login_user(username,password):
  c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
  data = c.fetchall()
  return data

def view_all_users():
  c.execute('SELECT * FROM userstable')
  data = c.fetchall()
  return data 

st.title("Fashion Finder System")

def main():
        st.subheader("Login Section")

        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password", type='password')

        if st.sidebar.checkbox("Login"):
            create_usertable()
            result = login_user(username,password)
            if result:

              st.success("Logged In as {}".format(username))

              task = st.selectbox("Task", ["Create New Image Storage", "Remove Image Storage", 
              "Reload Image Storage", "Remove Image Uploads"])
              if task == "Create New Image Storage":
                
                st.subheader("Create New Image Storage")
                image_files = glob.glob("images/*.jpg")
                manuscripts = []
                for image_file in image_files:
                    image_file = image_file.replace("\\","/")
                    parts = image_file.split("/")
                    if parts[1] not in manuscripts:
                       manuscripts.append(parts[1])
                manuscripts.sort()
                st.write(manuscripts)
                view_manuscripts_images = st.multiselect("Select Manuscripts(s)", manuscripts)
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
                    cols[i].write(image_file)
                    # st.write(i, image_file)
                uploaded_file = st.file_uploader("Choose an image")
                if uploaded_file is not None:
                  # display the file
                   open(os.path.join('images',uploaded_file.name),'wb').write(uploaded_file.getbuffer())

              #Remove Storage
              elif task == "Remove Image Storage":     
                st.subheader("Remove Image Storage")
                image_files = glob.glob("images\*.jpg")
                manuscripts = []

                for image_file in image_files:
                    image_file = image_file.replace("\\","/")
                    parts = image_file.split("/")
                    if parts[1] not in manuscripts:
                       manuscripts.append(parts[1])
                       
                manuscripts.sort()
                st.write(manuscripts)
                view_manuscripts_images = st.multiselect("Select Manuscripts(s)", manuscripts)
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
                    cols[i].write(image_file)
                    # st.write(i, image_file) 

                lokasione = st.text_input(" ", "siteplan/")
                
                if st.button("Lokasi"):
                  st.image(lokasione)
                  st.write(lokasione)

                file = st.text_input("Remove Filename Storage")
                
                if st.button("Remove Image Storage"):
                   path = os.path.join('images', file)
                   st.success("You have successfully")
                   os.remove(path)
              
              #Reload Image Storage
              elif task == "Reload Image Storage":
                st.subheader("Reload Image Storage")
                image_files = glob.glob("images/*.jpg")
                manuscripts = []
                for image_file in image_files:
                    image_file = image_file.replace("\\","/")
                    parts = image_file.split("/")
                    if parts[1] not in manuscripts:
                       manuscripts.append(parts[1])
                manuscripts.sort()
                st.write(manuscripts)
                view_manuscripts_images = st.multiselect("Select Manuscripts(s)", manuscripts)
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
                    cols[i].write(image_file)
                    # st.write(i, image_file)

                if st.button("Reload"):
                  model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
                  model.trainable = False

                  model = tensorflow.keras.Sequential([
                    model,
                    GlobalMaxPooling2D()
                  ])

                  def extract_features(img_path,model):
                      img = image.load_img(img_path,target_size=(224,224))
                      img_array = image.img_to_array(img)
                      expanded_img_array = np.expand_dims(img_array, axis=0)
                      preprocessed_img = preprocess_input(expanded_img_array)
                      result = model.predict(preprocessed_img).flatten()
                      normalized_result = result / norm(result)

                      return normalized_result

                  filenames = []

                  for file in os.listdir('images'):
                    filenames.append(os.path.join('images',file))

                    feature_list = []

                    for file in filenames:
                      feature_list.append(extract_features(file,model))
                      
                    pickle.dump(feature_list,open('embeddings.pkl','wb'))
                    pickle.dump(filenames,open('filenames.pkl','wb'))
                      # st.write(filenames)
                    st.write("Reload", file)
                  st.title("Reload Success")
                   
                #Remove Uploads
              elif task == "Remove Image Uploads":   
                st.subheader("Remove Image Uploads") 
                image_files = glob.glob("uploads/*.jpg")
                manuscripts = []

                for image_file in image_files:
                    image_file = image_file.replace("\\","/")
                    parts = image_file.split("/")
                    if parts[1] not in manuscripts:
                       manuscripts.append(parts[1])

                manuscripts.sort()
                st.write(manuscripts)
                view_manuscripts_uploads = st.multiselect("Select Manuscripts(s)", manuscripts)
                n = st.number_input("Select Grid Width", 1, 20, 10)
                view_images = []

                for image_file in image_files:
                  if any(manuscripts in image_file for manuscripts in view_manuscripts_uploads):
                    view_images.append(image_file)

                groups = []

                for i in range(0, len(view_images), n):
                  groups.append(view_images[i:i+n])

                for group in groups:
                  cols = st.columns(n)
                  for i, image_file in enumerate(group):
                    cols[i].image(image_file)
                    cols[i].write(image_file)
                    # st.write(i, image_file)

                fileuploads = st.text_input("remove filename uploads")

                if st.button("Remove Image Uploads"):
                   path = os.path.join('uploads',fileuploads)
                   st.success("You have successfully")
                   os.remove(path)

            else:
                st.warning("Incorrect Username/Password")       

if __name__ == '__main__':
    main()