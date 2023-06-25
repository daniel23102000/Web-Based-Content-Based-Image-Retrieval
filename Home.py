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

import glob

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

def searchshirt(gambarproductdua):
   c.execute("SELECT * FROM shirttable WHERE gambarproductdua='%s'" % (gambarproductdua))
   data = c.fetchall()
   return data

def searchtshirt(gambarproductdua):
   c.execute("SELECT * FROM tshirttable WHERE gambarproductdua='%s'" % (gambarproductdua))
   data = c.fetchall()
   return data

def searchmaleshoes(gambarproductdua):
   c.execute("SELECT * FROM maleshoestable WHERE gambarproductdua='%s'" % (gambarproductdua))
   data = c.fetchall()
   return data

def searchdress(gambarproductdua):
   c.execute("SELECT * FROM dresstable WHERE gambarproductdua='%s'" % (gambarproductdua))
   data = c.fetchall()
   return data

def searchwomentshirt(gambarproductdua):
   c.execute("SELECT * FROM womentshirttable WHERE gambarproductdua='%s'" % (gambarproductdua))
   data = c.fetchall()
   return data

def searchwomenshoes(gambarproductdua):
   c.execute("SELECT * FROM womenshoestable WHERE gambarproductdua='%s'" % (gambarproductdua))
   data = c.fetchall()
   return data

feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))



feature_listtshirt = np.array(pickle.load(open('embeddingstshirt.pkl','rb')))
feature_listmaleshoes = np.array(pickle.load(open('embeddingsmaleshoes.pkl','rb')))
feature_listdress = np.array(pickle.load(open('embeddingsdress.pkl','rb')))
feature_listwomentshirt = np.array(pickle.load(open('embeddingswomentshirt.pkl','rb')))
feature_listwomenshoes = np.array(pickle.load(open('embeddingswomenshoes.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))


filenamestshirt = pickle.load(open('filenamestshirt.pkl','rb'))
filenamesmaleshoes = pickle.load(open('filenamesmaleshoes.pkl','rb'))
filenamesdress = pickle.load(open('filenamesdress.pkl','rb'))
filenameswomentshirt = pickle.load(open('filenameswomentshirt.pkl','rb'))
filenameswomenshoes = pickle.load(open('filenameswomentshoes.pkl','rb'))

feature_listshirt = np.array(pickle.load(open('embeddingsshirt.pkl','rb')))
filenamesshirt = pickle.load(open('filenamesshirt.pkl','rb'))

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
    neighbors = NearestNeighbors()
   #  n_neighbors=6,algorithm='brute',metric='euclidean'
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

def recommendshirt(features,feature_listshirt):
    neighbors = NearestNeighbors()
    neighbors.fit(feature_listshirt)

    distances, indices = neighbors.kneighbors([features])

    return indices

def recommendtshirt(features,feature_listtshirt):
    neighbors = NearestNeighbors()
    neighbors.fit(feature_listtshirt)

    distances, indices = neighbors.kneighbors([features])

    return indices

def recommendmaleshoes(features,feature_listmaleshoes):
    neighbors = NearestNeighbors()
   #  n_neighbors=6
   #  algorithm='brute'
   #  metric='euclidean'
    neighbors.fit(feature_listmaleshoes)

    distances, indices = neighbors.kneighbors([features])

    return indices

def recommenddress(features,feature_listdress):
    neighbors = NearestNeighbors()
   #  n_neighbors=6
   #  algorithm='brute'
   #  metric='euclidean'
    neighbors.fit(feature_listdress)

    distances, indices = neighbors.kneighbors([features])

    return indices

def recommendwomentshirt(features,feature_listwomentshirt):
    neighbors = NearestNeighbors()
   #  n_neighbors=6
   #  algorithm='brute'
   #  metric='euclidean'
    neighbors.fit(feature_listwomentshirt)

    distances, indices = neighbors.kneighbors([features])

    return indices

def recommendwomenshoes(features,feature_listwomenshoes):
    neighbors = NearestNeighbors()
   #  n_neighbors=6
   #  algorithm='brute'
   #  metric='euclidean'
    neighbors.fit(feature_listwomenshoes)

    distances, indices = neighbors.kneighbors([features])

    return indices



page_bg_img = '''
<style>
# [data-testid="stSidebar"] {
# background-color: gray;
# background-size: cover;
# }

[data-testid="stAppViewContainer"] {
background-image: url("https://www.w3schools.com/w3images/jane.jpg");
background-size: cover;
}
<span class="css-10trblm e16nr0p30">Department Store</span>

</style>
<body>
</body>
'''

# selected = option_menu(
#         menu_title="",
#         options=["Home", "Product", "Camera", "Uploads", "About", "Contact"],
#         icons=["book","envelope"],
#         menu_icon="cast",
#         default_index=0,
#         orientation="horizontal",
#         key="hori",
#     )


with st.sidebar:
    selected = option_menu(
        menu_title="",
        options=["Home", "List of Product", "Camera Search Feature", "Uploads Search Feature", "About", "Contact"],
        icons=["house","card-list", "camera", "arrow-bar-up", "book", "envelope"],
        menu_icon="cast",
        default_index=0,
        key="side"
    )
st.markdown(page_bg_img, unsafe_allow_html=True)

if selected == "Home":
  st.title("Department Store")
  st.title("Collection 2023")

elif selected == "List of Product":
  menu = ["Men", "Women"]
  with st.sidebar:
   choice = st.selectbox("Menu", menu)

  if choice == "Men":
     with st.sidebar:
       selectedone = option_menu(
       menu_title="",
       options=["Shirts", "T-Shirts", "Shoes"],
       icons=["","", ""],
       menu_icon="cast",
       default_index=0,
       key="sideone"
     )

     if selectedone == "Shirts":
       st.title('Fashion Finder System')
       uploaded_file = st.file_uploader("Choose an image")
       if uploaded_file is not None:
        if save_uploaded_file(uploaded_file):
           features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
           indices = recommendshirt(features,feature_listshirt)
           search_result = searchshirt(filenamesshirt[indices[0][0]])
           for i in search_result:
            productnameone = i[0]
            priceone = i[1]
            lokasione = i[2]
            stockone = i[3]

            col1,col2,col3 = st.columns(3)
            with col1:
               st.image(filenamesshirt[indices[0][0]])
            with col2:
               search_result = searchshirt(filenamesshirt[indices[0][0]])

               for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    stockone = i[3]
                    gambarlokasione = i[7]
               st.image(gambarlokasione)
            with col3:
               st.write("Product Name : ", productnameone)
               st.write("Price : ",priceone)
               st.write("Locate : ",lokasione)
               st.write("Stock : ",stockone)

       st.subheader("List of Product")
       image_files = glob.glob("shirt/*.jpg")
       manuscripts = []
    
       for image_file in image_files:
        image_file = image_file.replace("\\","/")
        parts = image_file.split("/")
        if parts[1] not in manuscripts:
          manuscripts.append(parts[1])
    
       manuscripts.sort()
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

     elif selectedone == "T-Shirts":
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
           indices = recommendtshirt(features,feature_listtshirt)
           search_result = searchtshirt(filenamestshirt[indices[0][0]])
           for i in search_result:
            productnameone = i[0]
            priceone = i[1]
            lokasione = i[2]

            col1,col2,col3 = st.columns(3)
            with col1:
               st.image(filenamestshirt[indices[0][0]])
            with col2:
               search_result = searchtshirt(filenamestshirt[indices[0][0]])

               for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    stock = i[3]
                    gambarlokasione = i[7]
               st.image(gambarlokasione)
            with col3:
               st.write("Product Name : ", productnameone)
               st.write("Price : ",priceone)
               st.write("Locate : ",lokasione)
               st.write("Stock : ",stock)

       st.subheader("List of Product")
       image_files = glob.glob("tshirt/*.jpg")
       manuscripts = []
    
       for image_file in image_files:
        image_file = image_file.replace("\\","/")
        parts = image_file.split("/")
        if parts[1] not in manuscripts:
          manuscripts.append(parts[1])
    
       manuscripts.sort()
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

     elif selectedone == "Shoes":
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
           indices = recommendmaleshoes(features,feature_listmaleshoes)
           search_result = searchmaleshoes(filenamesmaleshoes[indices[0][0]])
           for i in search_result:
            productnameone = i[0]
            priceone = i[1]
            lokasione = i[2]

            col1,col2,col3 = st.columns(3)
            with col1:
               st.image(filenamesmaleshoes[indices[0][0]])
            with col2:
               search_result = searchmaleshoes(filenamesmaleshoes[indices[0][0]])

               for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    stock = i[3]
                    gambarlokasione = i[7]
               st.image(gambarlokasione)
            with col3:
               st.write("Product Name : ", productnameone)
               st.write("Price : ",priceone)
               st.write("Locate : ",lokasione)
               st.write("Stock : ",stock)

       st.subheader("List of Product")
       image_files = glob.glob("maleshoes/*.jpg")
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


  elif choice == "Women":
    with st.sidebar:
       selectedtwo = option_menu(
       menu_title="",
       options=["Dresses", "T-Shirts", "Shoes"],
       icons=["","", ""],
       menu_icon="cast",
       default_index=0,
       key="sidetwo"
     )

    if selectedtwo == "Dresses":
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
           indices = recommenddress(features,feature_listdress)
           search_result = searchdress(filenamesdress[indices[0][0]])
           for i in search_result:
            productnameone = i[0]
            priceone = i[1]
            lokasione = i[2]
            stock = i[3]

            col1,col2,col3 = st.columns(3)
            with col1:
               st.image(filenamesdress[indices[0][0]])
            with col2:
               search_result = searchdress(filenamesdress[indices[0][0]])

               for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    stock = i[3]
                    gambarlokasione = i[7]
               st.image(gambarlokasione)
            with col3:
               st.write("Product Name : ", productnameone)
               st.write("Price : ",priceone)
               st.write("Locate : ",lokasione)
               st.write("Stock : ",stock)

       st.subheader("List of Product")
       image_files = glob.glob("dress/*.jpg")
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

    elif selectedtwo == "T-Shirts":
       st.title('Fashion Finder System')
       uploaded_file = st.file_uploader("Choose an image")
       if uploaded_file is not None:
        if save_uploaded_file(uploaded_file):
           # display the file
         #   display_image = Image.open(uploaded_file)
         #   st.image(display_image)
           # feature extract
           features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
           # recommendention
           indices = recommendwomentshirt(features,feature_listwomentshirt)
           search_result = searchwomentshirt(filenameswomentshirt[indices[0][0]])
           for i in search_result:
            productnameone = i[0]
            priceone = i[1]
            lokasione = i[2]
            stock = i[3]

            col1,col2,col3 = st.columns(3)
            with col1:
               st.image(filenameswomentshirt[indices[0][0]])
            with col2:
               search_result = searchwomentshirt(filenameswomentshirt[indices[0][0]])

               for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    stock = i[3]
                    gambarlokasione = i[7]
               st.image(gambarlokasione)
            with col3:
               st.write("Product Name : ", productnameone)
               st.write("Price : ",priceone)
               st.write("Locate : ",lokasione)
               st.write("Stock: ",stock)

       st.subheader("List of Product")
       image_files = glob.glob("womentshirt/*.jpg")
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

    elif selectedtwo == "Shoes":
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
           indices = recommendwomenshoes(features,feature_listwomenshoes)
           search_result = searchwomenshoes(filenameswomenshoes[indices[0][0]])
           for i in search_result:
            productnameone = i[0]
            priceone = i[1]
            lokasione = i[2]
            stock = i [3]

            col1,col2,col3 = st.columns(3)
            with col1:
               st.image(filenameswomenshoes[indices[0][0]])
            with col2:
               search_result = searchwomenshoes(filenameswomenshoes[indices[0][0]])

               for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    stock = i[3]
                    gambarlokasione = i[7]
               st.image(gambarlokasione)
            with col3:
               st.write("Product Name : ", productnameone)
               st.write("Price : ",priceone)
               st.write("Locate : ",lokasione)
               st.write("Stock : ",stock)

       st.subheader("List of Product")
       image_files = glob.glob("womenshoes/*.jpg")
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

elif selected == "Camera Search Feature":
    menutwo = ["Men", "Women"]
    with st.sidebar:
      choice = st.selectbox("Menu", menutwo)

    if choice == "Men":
       with st.sidebar:
        selectedtwo = option_menu(
        menu_title="",
        options=["Shirts", "T-Shirts", "Shoes"],
        icons=["","", ""],
        menu_icon="cast",
        default_index=0,
        key="sidetwo"
       )

       if selectedtwo == "Shirts":
          uploaded_file = st.camera_input("Take a picture")
          if uploaded_file is not None:
             if save_uploaded_file(uploaded_file):
               features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
               indices = recommendshirt(features,feature_listshirt)
               search_result = searchshirt(filenamesshirt[indices[0][0]])
               for i in search_result:
                productnameone = i[0]

               search_result = searchshirt(filenamesshirt[indices[0][1]])
               for i in search_result:
                productnametwo = i[0]

               search_result = searchshirt(filenamesshirt[indices[0][2]])
               for i in search_result:
                 productnamethree = i[0]
            
               search_result = searchshirt(filenamesshirt[indices[0][3]])
               for i in search_result:
                 productnamefour = i[0]

               search_result = searchshirt(filenamesshirt[indices[0][4]])
               for i in search_result:
                 productnamefive = i[0]

               task = st.selectbox("Task", ["Recommendation", productnameone, productnametwo, 
               productnamethree, productnamefour, productnamefive])

               if task == "Recommendation":
                # show
                col1,col2,col3,col4,col5 = st.columns(5)

                with col1:
                  st.image(filenamesshirt[indices[0][0]])
                  st.write(productnameone)
                with col2:
                  st.image(filenamesshirt[indices[0][1]])
                  st.write(productnametwo)
                with col3:
                  st.image(filenamesshirt[indices[0][2]])
                  st.write(productnamethree)
                with col4:
                  st.image(filenamesshirt[indices[0][3]])
                  st.write(productnamefour)
                with col5:
                  st.image(filenamesshirt[indices[0][4]])
                  st.write(productnamefive)
            
               elif task == productnameone:
                col1,col2,col3 = st.columns(3)
                with col1:
                  st.image(filenamesshirt[indices[0][0]])
                with col2:
                  search_result = searchshirt(filenamesshirt[indices[0][0]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    stock = i[3]
                    gambarlokasione = i[6]

                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg')  

                with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Location : ",lokasione)
                  st.write("Stock : ",stock)

               elif task == productnametwo:
                 col1,col2,col3 = st.columns(3)
                 with col1:
                  st.image(filenamesshirt[indices[0][1]])
                 with col2:
                  search_result = searchshirt(filenamesshirt[indices[0][1]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    stock = i[3]
                    gambarlokasione = i[7]

                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

                 with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Location : ",lokasione)
                  st.write("Stock : ",stock)

               elif task == productnamethree:
                 col1,col2,col3 = st.columns(3)
                 with col1:
                  st.image(filenamesshirt[indices[0][2]])
                 with col2:
                  search_result = searchshirt(filenamesshirt[indices[0][2]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    stock = i[3]
                    gambarlokasione = i[7]

                 if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                 elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                 elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                 elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                 elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                 elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                 elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                 elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                 elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                 elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                 elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                 elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                 elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                 elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                 elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                 elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

                 with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Location : ",lokasione)
                  st.write("Stock : ",stock)

               elif task == productnamefour:
                 col1,col2,col3 = st.columns(3)
                 with col1:
                  st.image(filenamesshirt[indices[0][3]])
                 with col2:
                  search_result = searchshirt(filenamesshirt[indices[0][3]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    stock = i[3]
                    gambarlokasione = i[7]

                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

                 with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Location : ",lokasione)
                  st.write("Stock : ",stock)

               elif task == productnamefive:
                col1,col2,col3 = st.columns(3)
                with col1:
                  st.image(filenamesshirt[indices[0][4]])
                with col2:
                  search_result = searchshirt(filenamesshirt[indices[0][4]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    stock = i[3]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

                with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Location : ",lokasione)
                  st.write("Stock : ",stock)
          else:
              st.header("Not Found")

       elif selectedtwo == "T-Shirts":
         uploaded_file = st.camera_input("Take a picture")
         if uploaded_file is not None:
             if save_uploaded_file(uploaded_file):
              # display the file
              # display_image = Image.open(uploaded_file)
              # st.image(display_image)
              # feature extract
              features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
              # st.text(features)
              # recommendention
              indices = recommendtshirt(features,feature_listtshirt)
              search_result = searchtshirt(filenamestshirt[indices[0][0]])
              for i in search_result:
               productnameone = i[0]

             search_result = searchtshirt(filenamestshirt[indices[0][1]])
             for i in search_result:
               productnametwo = i[0]

             search_result = searchtshirt(filenamestshirt[indices[0][2]])
             for i in search_result:
               productnamethree = i[0]
            
             search_result = searchtshirt(filenamestshirt[indices[0][3]])
             for i in search_result:
               productnamefour = i[0]

             search_result = searchtshirt(filenamestshirt[indices[0][4]])
             for i in search_result:
               productnamefive = i[0]
             task = st.selectbox("Task", ["Recommendation", productnameone, productnametwo, 
             productnamethree, productnamefour, productnamefive])

             if task == "Recommendation":
             # show
               col1,col2,col3,col4,col5 = st.columns(5)

               with col1:
                  st.image(filenamestshirt[indices[0][0]])
                  st.write(productnameone)
               with col2:
                  st.image(filenamestshirt[indices[0][1]])
                  st.write(productnametwo)
               with col3:
                  st.image(filenamestshirt[indices[0][2]])
                  st.write(productnamethree)
               with col4:
                  st.image(filenamestshirt[indices[0][3]])
                  st.write(productnamefour)
               with col5:
                  st.image(filenamestshirt[indices[0][4]])
                  st.write(productnamefive)
            
             elif task == productnameone:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenamestshirt[indices[0][0]])
               with col2:
                  search_result = searchtshirt(filenamestshirt[indices[0][0]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]

                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg')

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

             elif task == productnametwo:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenamestshirt[indices[0][1]])
               with col2:
                  search_result = searchtshirt(filenamestshirt[indices[0][1]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]

                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg')

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

             elif task == productnamethree:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenamestshirt[indices[0][2]])
               with col2:
                  search_result = searchtshirt(filenamestshirt[indices[0][2]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

             elif task == productnamefour:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenamestshirt[indices[0][3]])
               with col2:
                  search_result = searchtshirt(filenamestshirt[indices[0][3]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]

                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

             elif task == productnamefive:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenamestshirt[indices[0][4]])
               with col2:
                  search_result = searchtshirt(filenamestshirt[indices[0][4]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]

                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)
         else:
            st.header("Not Found")
      
       elif selectedtwo == "Shoes":
          uploaded_file = st.camera_input("Take a picture")
          if uploaded_file is not None:
             if save_uploaded_file(uploaded_file):
              # display the file
              # display_image = Image.open(uploaded_file)
              # st.image(display_image)
              # feature extract
              features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
              # st.text(features)
              # recommendention
              indices = recommendmaleshoes(features,feature_listmaleshoes)
              search_result = searchmaleshoes(filenamesmaleshoes[indices[0][0]])
              for i in search_result:
               productnameone = i[0]

              search_result = searchmaleshoes(filenamesmaleshoes[indices[0][1]])
              for i in search_result:
               productnametwo = i[0]

              search_result = searchmaleshoes(filenamesmaleshoes[indices[0][2]])
              for i in search_result:
               productnamethree = i[0]
            
              search_result = searchmaleshoes(filenamesmaleshoes[indices[0][3]])
              for i in search_result:
               productnamefour = i[0]

              search_result = searchmaleshoes(filenamesmaleshoes[indices[0][4]])
              for i in search_result:
               productnamefive = i[0]
              task = st.selectbox("Task", ["Recommendation", productnameone, productnametwo, 
              productnamethree, productnamefour, productnamefive])

             if task == "Recommendation":
             # show
               col1,col2,col3,col4,col5 = st.columns(5)

               with col1:
                  st.image(filenamesmaleshoes[indices[0][0]])
                  st.write(productnameone)
               with col2:
                  st.image(filenamesmaleshoes[indices[0][1]])
                  st.write(productnametwo)
               with col3:
                  st.image(filenamesmaleshoes[indices[0][2]])
                  st.write(productnamethree)
               with col4:
                  st.image(filenamesmaleshoes[indices[0][3]])
                  st.write(productnamefour)
               with col5:
                  st.image(filenamesmaleshoes[indices[0][4]])
                  st.write(productnamefive)
            
             elif task == productnameone:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenamesmaleshoes[indices[0][0]])
               with col2:
                  search_result = searchmaleshoes(filenamesmaleshoes[indices[0][0]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

             elif task == productnametwo:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenamesmaleshoes[indices[0][1]])
               with col2:
                  search_result = searchmaleshoes(filenamesmaleshoes[indices[0][1]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

             elif task == productnamethree:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenamesmaleshoes[indices[0][2]])
               with col2:
                  search_result = searchmaleshoes(filenamesmaleshoes[indices[0][2]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

             elif task == productnamefour:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenamesmaleshoes[indices[0][3]])
               with col2:
                  search_result = searchmaleshoes(filenamesmaleshoes[indices[0][3]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]

                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

             elif task == productnamefive:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenamesmaleshoes[indices[0][4]])
               with col2:
                  search_result = searchmaleshoes(filenamesmaleshoes[indices[0][4]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]

                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)
          else:
            st.header("Not Found")

    elif choice == "Women":
      with st.sidebar:
        selectedthree = option_menu(
        menu_title="",
        options=["Dresses", "T-Shirts", "Shoes"],
        icons=["","", ""],
        menu_icon="cast",
        default_index=0,
        key="sidethree"
       )

      
      if selectedthree == "Dresses":
         uploaded_file = st.camera_input("Take a picture")
         if uploaded_file is not None:
            if save_uploaded_file(uploaded_file):
              # display the file
              # display_image = Image.open(uploaded_file)
              # st.image(display_image)
              # feature extract
              features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
              # st.text(features)
              # recommendention
              indices = recommenddress(features,feature_listdress)
              search_result = searchdress(filenamesdress[indices[0][0]])
              for i in search_result:
               productnameone = i[0]

              search_result = searchdress(filenamesdress[indices[0][1]])
              for i in search_result:
               productnametwo = i[0]

              search_result = searchdress(filenamesdress[indices[0][2]])
              for i in search_result:
               productnamethree = i[0]
            
              search_result = searchdress(filenamesdress[indices[0][3]])
              for i in search_result:
               productnamefour = i[0]

              search_result = searchdress(filenamesdress[indices[0][4]])
              for i in search_result:
               productnamefive = i[0]
              task = st.selectbox("Task", ["Recommendation", productnameone, productnametwo, 
              productnamethree, productnamefour, productnamefive])

              if task == "Recommendation":
              # show
               col1,col2,col3,col4,col5 = st.columns(5)

               with col1:
                  st.image(filenamesdress[indices[0][0]])
                  st.write(productnameone)
               with col2:
                  st.image(filenamesdress[indices[0][1]])
                  st.write(productnametwo)
               with col3:
                  st.image(filenamesdress[indices[0][2]])
                  st.write(productnamethree)
               with col4:
                  st.image(filenamesdress[indices[0][3]])
                  st.write(productnamefour)
               with col5:
                  st.image(filenamesdress[indices[0][4]])
                  st.write(productnamefive)
            
              elif task == productnameone:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenamesdress[indices[0][0]])
               with col2:
                  search_result = searchdress(filenamesdress[indices[0][0]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

              elif task == productnametwo:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenamesdress[indices[0][1]])
               with col2:
                  search_result = searchdress(filenamesdress[indices[0][1]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]

                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

              elif task == productnamethree:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenamesdress[indices[0][2]])
               with col2:
                  search_result = searchdress(filenamesdress[indices[0][2]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]

                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

              elif task == productnamefour:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenamesdress[indices[0][3]])
               with col2:
                  search_result = searchdress(filenamesdress[indices[0][3]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

              elif task == productnamefive:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenamesdress[indices[0][4]])
               with col2:
                  search_result = searchdress(filenamesdress[indices[0][4]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)
         else:
            st.header("Not Found")

      elif selectedthree == "T-Shirts":     
         uploaded_file = st.camera_input("Take a picture")
         if uploaded_file is not None:
            if save_uploaded_file(uploaded_file):
              # display the file
              # display_image = Image.open(uploaded_file)
               # st.image(display_image)
               # feature extract
              features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
              # st.text(features)
              # recommendention
              indices = recommendwomentshirt(features,feature_listwomentshirt)
              search_result = searchwomentshirt(filenameswomentshirt[indices[0][0]])
              for i in search_result:
               productnameone = i[0]

              search_result = searchwomentshirt(filenameswomentshirt[indices[0][1]])
              for i in search_result:
               productnametwo = i[0]

              search_result = searchwomentshirt(filenameswomentshirt[indices[0][2]])
              for i in search_result:
               productnamethree = i[0]
            
              search_result = searchwomentshirt(filenameswomentshirt[indices[0][3]])
              for i in search_result:
               productnamefour = i[0]

              search_result = searchwomentshirt(filenameswomentshirt[indices[0][4]])
              for i in search_result:
               productnamefive = i[0]
              task = st.selectbox("Task", ["Recommendation", productnameone, productnametwo, 
              productnamethree, productnamefour, productnamefive])

              if task == "Recommendation":
              # show
               col1,col2,col3,col4,col5 = st.columns(5)

               with col1:
                  st.image(filenameswomentshirt[indices[0][0]])
                  st.write(productnameone)
               with col2:
                  st.image(filenameswomentshirt[indices[0][1]])
                  st.write(productnametwo)
               with col3:
                  st.image(filenameswomentshirt[indices[0][2]])
                  st.write(productnamethree)
               with col4:
                  st.image(filenameswomentshirt[indices[0][3]])
                  st.write(productnamefour)
               with col5:
                  st.image(filenameswomentshirt[indices[0][4]])
                  st.write(productnamefive)
            
              elif task == productnameone:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenameswomentshirt[indices[0][0]])
               with col2:
                  search_result = searchwomentshirt(filenameswomentshirt[indices[0][0]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 


               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

              elif task == productnametwo:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenameswomentshirt[indices[0][1]])
               with col2:
                  search_result = searchwomentshirt(filenameswomentshirt[indices[0][1]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

              elif task == productnamethree:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenameswomentshirt[indices[0][2]])
               with col2:
                  search_result = searchwomentshirt(filenameswomentshirt[indices[0][2]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                    
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

              elif task == productnamefour:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenameswomentshirt[indices[0][3]])
               with col2:
                  search_result = searchwomentshirt(filenameswomentshirt[indices[0][3]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

              elif task == productnamefive:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenameswomentshirt[indices[0][4]])
               with col2:
                  search_result = searchwomentshirt(filenameswomentshirt[indices[0][4]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)
         else:
           st.header("Not Found")

      elif selectedthree == "Shoes":
         uploaded_file = st.camera_input("Take a picture")
         if uploaded_file is not None:
            if save_uploaded_file(uploaded_file):
              # display the file
              # display_image = Image.open(uploaded_file)
               # st.image(display_image)
               # feature extract
              features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
              # st.text(features)
              # recommendention
              indices = recommendwomenshoes(features,feature_listwomenshoes)
              search_result = searchwomenshoes(filenameswomenshoes[indices[0][0]])
              for i in search_result:
               productnameone = i[0]

              search_result = searchwomenshoes(filenameswomenshoes[indices[0][1]])
              for i in search_result:
               productnametwo = i[0]

              search_result = searchwomenshoes(filenameswomenshoes[indices[0][2]])
              for i in search_result:
               productnamethree = i[0]
            
              search_result = searchwomenshoes(filenameswomenshoes[indices[0][3]])
              for i in search_result:
               productnamefour = i[0]

              search_result = searchwomenshoes(filenameswomenshoes[indices[0][4]])
              for i in search_result:
               productnamefive = i[0]
              task = st.selectbox("Task", ["Recommendation", productnameone, productnametwo, 
              productnamethree, productnamefour, productnamefive])

              if task == "Recommendation":
              # show
               col1,col2,col3,col4,col5 = st.columns(5)

               with col1:
                  st.image(filenameswomenshoes[indices[0][0]])
                  st.write(productnameone)
               with col2:
                  st.image(filenameswomenshoes[indices[0][1]])
                  st.write(productnametwo)
               with col3:
                  st.image(filenameswomenshoes[indices[0][2]])
                  st.write(productnamethree)
               with col4:
                  st.image(filenameswomenshoes[indices[0][3]])
                  st.write(productnamefour)
               with col5:
                  st.image(filenameswomenshoes[indices[0][4]])
                  st.write(productnamefive)
            
              elif task == productnameone:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenameswomenshoes[indices[0][0]])
               with col2:
                  search_result = searchwomenshoes(filenameswomenshoes[indices[0][0]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

              elif task == productnametwo:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenameswomenshoes[indices[0][1]])
               with col2:
                  search_result = searchwomenshoes(filenameswomenshoes[indices[0][1]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

              elif task == productnamethree:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenameswomenshoes[indices[0][2]])
               with col2:
                  search_result = searchwomenshoes(filenameswomenshoes[indices[0][2]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

              elif task == productnamefour:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenameswomenshoes[indices[0][3]])
               with col2:
                  search_result = searchwomenshoes(filenameswomenshoes[indices[0][3]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

              elif task == productnamefive:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenameswomenshoes[indices[0][4]])
               with col2:
                  search_result = searchwomenshoes(filenameswomenshoes[indices[0][4]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)
         else:
           st.header("Not Found")


elif selected == "Uploads Search Feature":
    menutwo = ["Men", "Women"]
    with st.sidebar:
      choice = st.selectbox("Menu", menutwo)

    if choice == "Men":
       with st.sidebar:
        selectedtwo = option_menu(
        menu_title="",
        options=["Shirts", "T-Shirts", "Shoes"],
        icons=["","", ""],
        menu_icon="cast",
        default_index=0,
        key="sidetwo"
       )

       if selectedtwo == "Shirts":
          uploaded_file = st.file_uploader("Choose a Picture")
          if uploaded_file is not None:
             if save_uploaded_file(uploaded_file):
               features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
               indices = recommendshirt(features,feature_listshirt)
               search_result = searchshirt(filenamesshirt[indices[0][0]])
               for i in search_result:
                productnameone = i[0]

               search_result = searchshirt(filenamesshirt[indices[0][1]])
               for i in search_result:
                productnametwo = i[0]

               search_result = searchshirt(filenamesshirt[indices[0][2]])
               for i in search_result:
                 productnamethree = i[0]
            
               search_result = searchshirt(filenamesshirt[indices[0][3]])
               for i in search_result:
                 productnamefour = i[0]

               search_result = searchshirt(filenamesshirt[indices[0][4]])
               for i in search_result:
                 productnamefive = i[0]

               task = st.selectbox("Task", ["Recommendation", productnameone, productnametwo, 
               productnamethree, productnamefour, productnamefive])

               if task == "Recommendation":
                # show
                col1,col2,col3,col4,col5 = st.columns(5)

                with col1:
                  st.image(filenamesshirt[indices[0][0]])
                  st.write(productnameone)
                with col2:
                  st.image(filenamesshirt[indices[0][1]])
                  st.write(productnametwo)
                with col3:
                  st.image(filenamesshirt[indices[0][2]])
                  st.write(productnamethree)
                with col4:
                  st.image(filenamesshirt[indices[0][3]])
                  st.write(productnamefour)
                with col5:
                  st.image(filenamesshirt[indices[0][4]])
                  st.write(productnamefive)
            
               elif task == productnameone:
                col1,col2,col3 = st.columns(3)
                with col1:
                  st.image(filenamesshirt[indices[0][0]])
                with col2:
                  search_result = searchshirt(filenamesshirt[indices[0][0]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    stock = i[3]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

                with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Location : ",lokasione)
                  st.write("Stock : ",stock)

               elif task == productnametwo:
                 col1,col2,col3 = st.columns(3)
                 with col1:
                  st.image(filenamesshirt[indices[0][1]])
                 with col2:
                  search_result = searchshirt(filenamesshirt[indices[0][1]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    stock = i[3]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

                 with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)
                  st.write("Stock : ",stock)

               elif task == productnamethree:
                 col1,col2,col3 = st.columns(3)
                 with col1:
                  st.image(filenamesshirt[indices[0][2]])
                 with col2:
                  search_result = searchshirt(filenamesshirt[indices[0][2]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    stock = i[3]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

                 with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)
                  st.write("Stock : ",stock)

               elif task == productnamefour:
                 col1,col2,col3 = st.columns(3)
                 with col1:
                  st.image(filenamesshirt[indices[0][3]])
                 with col2:
                  search_result = searchshirt(filenamesshirt[indices[0][3]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    stock = i[3]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

                 with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)
                  st.write("Stock : ",stock)

               elif task == productnamefive:
                col1,col2,col3 = st.columns(3)
                with col1:
                  st.image(filenamesshirt[indices[0][4]])
                with col2:
                  search_result = searchshirt(filenamesshirt[indices[0][4]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    stock = i[3]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

                with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)
                  st.write("Stock : ",stock)
          else:
              st.header("Not Found")

       elif selectedtwo == "T-Shirts":
         uploaded_file = st.file_uploader("Choose a picture")
         if uploaded_file is not None:
             if save_uploaded_file(uploaded_file):
              # display the file
              # display_image = Image.open(uploaded_file)
              # st.image(display_image)
              # feature extract
              features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
              # st.text(features)
              # recommendention
              indices = recommendtshirt(features,feature_listtshirt)
              search_result = searchtshirt(filenamestshirt[indices[0][0]])
              for i in search_result:
               productnameone = i[0]

             search_result = searchtshirt(filenamestshirt[indices[0][1]])
             for i in search_result:
               productnametwo = i[0]

             search_result = searchtshirt(filenamestshirt[indices[0][2]])
             for i in search_result:
               productnamethree = i[0]
            
             search_result = searchtshirt(filenamestshirt[indices[0][3]])
             for i in search_result:
               productnamefour = i[0]

             search_result = searchtshirt(filenamestshirt[indices[0][4]])
             for i in search_result:
               productnamefive = i[0]
             task = st.selectbox("Task", ["Recommendation", productnameone, productnametwo, 
             productnamethree, productnamefour, productnamefive])

             if task == "Recommendation":
             # show
               col1,col2,col3,col4,col5 = st.columns(5)

               with col1:
                  st.image(filenamestshirt[indices[0][0]])
                  st.write(productnameone)
               with col2:
                  st.image(filenamestshirt[indices[0][1]])
                  st.write(productnametwo)
               with col3:
                  st.image(filenamestshirt[indices[0][2]])
                  st.write(productnamethree)
               with col4:
                  st.image(filenamestshirt[indices[0][3]])
                  st.write(productnamefour)
               with col5:
                  st.image(filenamestshirt[indices[0][4]])
                  st.write(productnamefive)
            
             elif task == productnameone:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenamestshirt[indices[0][0]])
               with col2:
                  search_result = searchtshirt(filenamestshirt[indices[0][0]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

             elif task == productnametwo:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenamestshirt[indices[0][1]])
               with col2:
                  search_result = searchtshirt(filenamestshirt[indices[0][1]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

             elif task == productnamethree:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenamestshirt[indices[0][2]])
               with col2:
                  search_result = searchtshirt(filenamestshirt[indices[0][2]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]

                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

             elif task == productnamefour:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenamestshirt[indices[0][3]])
               with col2:
                  search_result = searchtshirt(filenamestshirt[indices[0][3]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]

                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

             elif task == productnamefive:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenamestshirt[indices[0][4]])
               with col2:
                  search_result = searchtshirt(filenamestshirt[indices[0][4]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]

                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)
         else:
            st.header("Not Found")
      
       elif selectedtwo == "Shoes":
          uploaded_file = st.file_uploader("Choose a picture")
          if uploaded_file is not None:
             if save_uploaded_file(uploaded_file):
              # display the file
              # display_image = Image.open(uploaded_file)
              # st.image(display_image)
              # feature extract
              features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
              # st.text(features)
              # recommendention
              indices = recommendmaleshoes(features,feature_listmaleshoes)
              search_result = searchmaleshoes(filenamesmaleshoes[indices[0][0]])
              for i in search_result:
               productnameone = i[0]

              search_result = searchmaleshoes(filenamesmaleshoes[indices[0][1]])
              for i in search_result:
               productnametwo = i[0]

              search_result = searchmaleshoes(filenamesmaleshoes[indices[0][2]])
              for i in search_result:
               productnamethree = i[0]
            
              search_result = searchmaleshoes(filenamesmaleshoes[indices[0][3]])
              for i in search_result:
               productnamefour = i[0]

              search_result = searchmaleshoes(filenamesmaleshoes[indices[0][4]])
              for i in search_result:
               productnamefive = i[0]
              task = st.selectbox("Task", ["Recommendation", productnameone, productnametwo, 
              productnamethree, productnamefour, productnamefive])

             if task == "Recommendation":
             # show
               col1,col2,col3,col4,col5 = st.columns(5)

               with col1:
                  st.image(filenamesmaleshoes[indices[0][0]])
                  st.write(productnameone)
               with col2:
                  st.image(filenamesmaleshoes[indices[0][1]])
                  st.write(productnametwo)
               with col3:
                  st.image(filenamesmaleshoes[indices[0][2]])
                  st.write(productnamethree)
               with col4:
                  st.image(filenamesmaleshoes[indices[0][3]])
                  st.write(productnamefour)
               with col5:
                  st.image(filenamesmaleshoes[indices[0][4]])
                  st.write(productnamefive)
            
             elif task == productnameone:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenamesmaleshoes[indices[0][0]])
               
               with col2:
                  search_result = searchmaleshoes(filenamesmaleshoes[indices[0][0]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

             elif task == productnametwo:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenamesmaleshoes[indices[0][1]])

               with col2:
                  search_result = searchmaleshoes(filenamesmaleshoes[indices[0][1]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

             elif task == productnamethree:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenamesmaleshoes[indices[0][2]])
               with col2:
                  search_result = searchmaleshoes(filenamesmaleshoes[indices[0][2]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

             elif task == productnamefour:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenamesmaleshoes[indices[0][3]])
               with col2:
                  search_result = searchmaleshoes(filenamesmaleshoes[indices[0][3]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

             elif task == productnamefive:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenamesmaleshoes[indices[0][4]])
               with col2:
                  search_result = searchmaleshoes(filenamesmaleshoes[indices[0][4]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)
          else:
            st.header("Not Found")

    elif choice == "Women":
      with st.sidebar:
        selectedthree = option_menu(
        menu_title="",
        options=["Dresses", "T-Shirts", "Shoes"],
        icons=["","", ""],
        menu_icon="cast",
        default_index=0,
        key="sidethree"
       )

      
      if selectedthree == "Dresses":
         uploaded_file = st.file_uploader("Choose a picture")
         if uploaded_file is not None:
            if save_uploaded_file(uploaded_file):
              # display the file
              # display_image = Image.open(uploaded_file)
              # st.image(display_image)
              # feature extract
              features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
              # st.text(features)
              # recommendention
              indices = recommenddress(features,feature_listdress)
              
              search_result = searchdress(filenamesdress[indices[0][0]])
              for i in search_result:
               productnameone = i[0]

              search_result = searchdress(filenamesdress[indices[0][1]])
              for i in search_result:
               productnametwo = i[0]

              search_result = searchdress(filenamesdress[indices[0][2]])
              for i in search_result:
               productnamethree = i[0]
            
              search_result = searchdress(filenamesdress[indices[0][3]])
              for i in search_result:
               productnamefour = i[0]

              search_result = searchdress(filenamesdress[indices[0][4]])
              for i in search_result:
               productnamefive = i[0]

              task = st.selectbox("Task", ["Recommendation", productnameone, productnametwo, 
              productnamethree, productnamefour, productnamefive])

              if task == "Recommendation":
                  # show
                  col1,col2,col3,col4,col5 = st.columns(5)

                  with col1:
                     st.image(filenamesdress[indices[0][0]])
                     st.write(productnameone)

                  with col2:
                     st.image(filenamesdress[indices[0][1]])
                     st.write(productnametwo)

                  with col3:
                     st.image(filenamesdress[indices[0][2]])
                     st.write(productnamethree)

                  with col4:
                     st.image(filenamesdress[indices[0][3]])
                     st.write(productnamefour)

                  with col5:
                     st.image(filenamesdress[indices[0][4]])
                     st.write(productnamefive)
            
              elif task == productnameone:
                  col1,col2,col3 = st.columns(3)
                  with col1:
                     st.image(filenamesdress[indices[0][0]])
                  with col2:
                     search_result = searchdress(filenamesdress[indices[0][0]])

                     for i in search_result:
                        productnameoneone = i[0]
                        priceone = i[1]
                        lokasione = i[2]
                        gambarlokasione = i[7]

                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

                  with col3:
                     st.write("Product Name : ", productnameoneone)
                     st.write("Price : ",priceone)
                     st.write("Lokasi : ",lokasione)

              elif task == productnametwo:
                  col1,col2,col3 = st.columns(3)
                  with col1:
                     st.image(filenamesdress[indices[0][1]])
                  with col2:
                     search_result = searchdress(filenamesdress[indices[0][1]])

                     for i in search_result:
                        productnameoneone = i[0]
                        priceone = i[1]
                        lokasione = i[2]
                        gambarlokasione = i[7]
                     
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

                  with col3:
                     st.write("Product Name : ", productnameoneone)
                     st.write("Price : ",priceone)
                     st.write("Lokasi : ",lokasione)

              elif task == productnamethree:
                  col1,col2,col3 = st.columns(3)
                  with col1:
                     st.image(filenamesdress[indices[0][2]])
                  with col2:
                     search_result = searchdress(filenamesdress[indices[0][2]])

                     for i in search_result:
                        productnameoneone = i[0]
                        priceone = i[1]
                        lokasione = i[2]
                        gambarlokasione = i[7]
                     
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

                  with col3:
                     st.write("Product Name : ", productnameoneone)
                     st.write("Price : ",priceone)
                     st.write("Lokasi : ",lokasione)

              elif task == productnamefour:
                  col1,col2,col3 = st.columns(3)
                  with col1:
                     st.image(filenamesdress[indices[0][3]])
                  with col2:
                     search_result = searchdress(filenamesdress[indices[0][3]])

                     for i in search_result:
                        productnameoneone = i[0]
                        priceone = i[1]
                        lokasione = i[2]
                        gambarlokasione = i[7]
                     
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

                  with col3:
                     st.write("Product Name : ", productnameoneone)
                     st.write("Price : ",priceone)
                     st.write("Lokasi : ",lokasione)

              elif task == productnamefive:
                  col1,col2,col3 = st.columns(3)
                  with col1:
                     st.image(filenamesdress[indices[0][4]])
                  with col2:
                     search_result = searchdress(filenamesdress[indices[0][4]])

                     for i in search_result:
                        productnameoneone = i[0]
                        priceone = i[1]
                        lokasione = i[2]
                        gambarlokasione = i[7]
                     
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

                  with col3:
                     st.write("Product Name : ", productnameoneone)
                     st.write("Price : ",priceone)
                     st.write("Lokasi : ",lokasione)
         else:
            st.header("Not Found")

      elif selectedthree == "T-Shirts":     
         uploaded_file = st.file_uploader("Choose a picture")
         if uploaded_file is not None:
            if save_uploaded_file(uploaded_file):
              # display the file
              # display_image = Image.open(uploaded_file)
               # st.image(display_image)
               # feature extract
              features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
              # st.text(features)
              # recommendention
              indices = recommendwomentshirt(features,feature_listwomentshirt)
              search_result = searchwomentshirt(filenameswomentshirt[indices[0][0]])
              for i in search_result:
               productnameone = i[0]

              search_result = searchwomentshirt(filenameswomentshirt[indices[0][1]])
              for i in search_result:
               productnametwo = i[0]

              search_result = searchwomentshirt(filenameswomentshirt[indices[0][2]])
              for i in search_result:
               productnamethree = i[0]
            
              search_result = searchwomentshirt(filenameswomentshirt[indices[0][3]])
              for i in search_result:
               productnamefour = i[0]

              search_result = searchwomentshirt(filenameswomentshirt[indices[0][4]])
              for i in search_result:
               productnamefive = i[0]
              task = st.selectbox("Task", ["Recommendation", productnameone, productnametwo, 
              productnamethree, productnamefour, productnamefive])

              if task == "Recommendation":
              # show
               col1,col2,col3,col4,col5 = st.columns(5)

               with col1:
                  st.image(filenameswomentshirt[indices[0][0]])
                  st.write(productnameone)
               with col2:
                  st.image(filenameswomentshirt[indices[0][1]])
                  st.write(productnametwo)
               with col3:
                  st.image(filenameswomentshirt[indices[0][2]])
                  st.write(productnamethree)
               with col4:
                  st.image(filenameswomentshirt[indices[0][3]])
                  st.write(productnamefour)
               with col5:
                  st.image(filenameswomentshirt[indices[0][4]])
                  st.write(productnamefive)
            
              elif task == productnameone:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenameswomentshirt[indices[0][0]])
               with col2:
                  search_result = searchwomentshirt(filenameswomentshirt[indices[0][0]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg')

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

              elif task == productnametwo:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenameswomentshirt[indices[0][1]])
               with col2:
                  search_result = searchwomentshirt(filenameswomentshirt[indices[0][1]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

              elif task == productnamethree:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenameswomentshirt[indices[0][2]])
               with col2:
                  search_result = searchwomentshirt(filenameswomentshirt[indices[0][2]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

              elif task == productnamefour:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenameswomentshirt[indices[0][3]])
               with col2:
                  search_result = searchwomentshirt(filenameswomentshirt[indices[0][3]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

              elif task == productnamefive:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenameswomentshirt[indices[0][4]])
               with col2:
                  search_result = searchwomentshirt(filenameswomentshirt[indices[0][4]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)
         else:
           st.header("Not Found")

      elif selectedthree == "Shoes":
         uploaded_file = st.file_uploader("Choose a Picture")
         if uploaded_file is not None:
          if save_uploaded_file(uploaded_file):
              # display the file
              # display_image = Image.open(uploaded_file)
               # st.image(display_image)
               # feature extract
              features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
              # st.text(features)
              # recommendention
              indices = recommendwomenshoes(features,feature_listwomenshoes)
              search_result = searchwomenshoes(filenameswomenshoes[indices[0][0]])
              for i in search_result:
               productnameone = i[0]

              search_result = searchwomenshoes(filenameswomenshoes[indices[0][1]])
              for i in search_result:
               productnametwo = i[0]

              search_result = searchwomenshoes(filenameswomenshoes[indices[0][2]])
              for i in search_result:
               productnamethree = i[0]
            
              search_result = searchwomenshoes(filenameswomenshoes[indices[0][3]])
              for i in search_result:
               productnamefour = i[0]

              search_result = searchwomenshoes(filenameswomenshoes[indices[0][4]])
              for i in search_result:
               productnamefive = i[0]
              task = st.selectbox("Task", ["Recommendation", productnameone, productnametwo, 
              productnamethree, productnamefour, productnamefive])

              if task == "Recommendation":
              # show
               col1,col2,col3,col4,col5 = st.columns(5)

               with col1:
                  st.image(filenameswomenshoes[indices[0][0]])
                  st.write(productnameone)
               with col2:
                  st.image(filenameswomenshoes[indices[0][1]])
                  st.write(productnametwo)
               with col3:
                  st.image(filenameswomenshoes[indices[0][2]])
                  st.write(productnamethree)
               with col4:
                  st.image(filenameswomenshoes[indices[0][3]])
                  st.write(productnamefour)
               with col5:
                  st.image(filenameswomenshoes[indices[0][4]])
                  st.write(productnamefive)
            
              elif task == productnameone:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenameswomenshoes[indices[0][0]])
               with col2:
                  search_result = searchwomenshoes(filenameswomenshoes[indices[0][0]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

              elif task == productnametwo:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenameswomenshoes[indices[0][1]])
               with col2:
                  search_result = searchwomenshoes(filenameswomenshoes[indices[0][1]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

              elif task == productnamethree:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenameswomenshoes[indices[0][2]])
               with col2:
                  search_result = searchwomenshoes(filenameswomenshoes[indices[0][2]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

              elif task == productnamefour:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenameswomenshoes[indices[0][3]])
               with col2:
                  search_result = searchwomenshoes(filenameswomenshoes[indices[0][3]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 

               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)

              elif task == productnamefive:
               col1,col2,col3 = st.columns(3)
               with col1:
                  st.image(filenameswomenshoes[indices[0][4]])
               with col2:
                  search_result = searchwomenshoes(filenameswomenshoes[indices[0][4]])

                  for i in search_result:
                    productnameoneone = i[0]
                    priceone = i[1]
                    lokasione = i[2]
                    gambarlokasione = i[7]
                  
                  if lokasione == "Blok A":
                     st.image('siteplan\Blok A.jpg')
                  elif lokasione == "Blok B":
                     st.image('siteplan\Blok B.jpg')
                  elif lokasione == "Blok C":
                     st.image('siteplan\Blok C.jpg')
                  elif lokasione == "Blok D":
                     st.image('siteplan\Blok D.jpg')
                  elif lokasione == "Blok E":
                     st.image('siteplan\Blok E.jpg')
                  elif lokasione == "Blok F":
                     st.image('siteplan\Blok F.jpg')
                  elif lokasione == "Blok G":
                     st.image('siteplan\Blok G.jpg')
                  elif lokasione == "Blok H":
                     st.image('siteplan\Blok H.jpg')
                  elif lokasione == "Blok I":
                     st.image('siteplan\Blok I.jpg')
                  elif lokasione == "Blok J":
                     st.image('siteplan\Blok J.jpg')
                  elif lokasione == "Blok K":
                     st.image('siteplan\Blok K.jpg')
                  elif lokasione == "Blok L":
                     st.image('siteplan\Blok L.jpg')
                  elif lokasione == "Blok M":
                     st.image('siteplan\Blok M.jpg')  
                  elif lokasione == "Blok N":
                     st.image('siteplan\Blok N.jpg')
                  elif lokasione == "Blok O":
                     st.image('siteplan\Blok O.jpg')
                  elif lokasione == "Blok P":
                     st.image('siteplan\Blok P.jpg') 
                  
               with col3:
                  st.write("Product Name : ", productnameoneone)
                  st.write("Price : ",priceone)
                  st.write("Lokasi : ",lokasione)
         else:
           st.header("Not Found")


elif selected == "About":
    st.title("About Department Store")
    st.markdown("The Catering was founded in blabla by Mr. Smith in lorem ipsum dolor sit amet, consectetur adipiscing elit consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute iruredolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.We only use seasonal ingredients.")
    st.markdown("Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum consectetur adipiscing elit, sed do eiusmod temporincididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.")

elif selected == "Contact":
    st.title("Contact Us")
    st.markdown("We offer full-service catering for any event, large or small. We understand your needs and we will cater the food to satisfy the biggerst criteria of them all, both look and taste. Do not hesitate to contact us.")
    st.markdown("Catering Service, 42nd Living St, 43043 New York, NY")
    st.markdown("You can also contact us by phone 00553123-2323 or email danielraul4625@gmail.com")