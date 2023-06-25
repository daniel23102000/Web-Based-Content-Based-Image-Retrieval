import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import os
import glob
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
import pickle


# DB Management
import sqlite3
conn = sqlite3.connect('data.db')
c = conn.cursor()

def create_admintable():
  c.execute('CREATE TABLE IF NOT EXISTS adminstable(username TEXT,password TEXT)')  

def create_producttable():
  c.execute('CREATE TABLE IF NOT EXISTS prostabletiga(productname TEXT,price TEXT,Lokasi TEXT,gambarproduct TEXT,gambarproductdua TEXT,gambarlokasi TEXT)')

def create_shirttable():
  c.execute('CREATE TABLE IF NOT EXISTS shirttable(productname TEXT,price TEXT,Lokasi TEXT,stock TEXT,gambarproduct TEXT,gambarproductdua TEXT,gambarlokasi TEXT,gambarlokasidua TEXT)')

def create_shirttablereviseone():
  c.execute('CREATE TABLE IF NOT EXISTS shirttablerevisethree(productname TEXT,price TEXT,lokasi TEXT,stock TEXT,gambarproduct TEXT, gambarproductdua TEXT)')

def create_tshirttable():
  c.execute('CREATE TABLE IF NOT EXISTS tshirttable(productname TEXT,price TEXT,Lokasi TEXT,stock TEXT,gambarproduct TEXT,gambarproductdua TEXT,gambarlokasi TEXT,gambarlokasidua TEXT)')

def create_maleshoestable():
  c.execute('CREATE TABLE IF NOT EXISTS maleshoestable(productname TEXT,price TEXT,Lokasi TEXT,stock TEXT,gambarproduct TEXT,gambarproductdua TEXT,gambarlokasi TEXT,gambarlokasidua TEXT)')

def create_dresstable():
  c.execute('CREATE TABLE IF NOT EXISTS dresstable(productname TEXT,price TEXT,Lokasi TEXT,stock TEXT,gambarproduct TEXT,gambarproductdua TEXT,gambarlokasi TEXT,gambarlokasidua TEXT)')

def create_womentshirttable():
  c.execute('CREATE TABLE IF NOT EXISTS womentshirttable(productname TEXT,price TEXT,Lokasi TEXT,stock TEXT,gambarproduct TEXT,gambarproductdua TEXT,gambarlokasi TEXT,gambarlokasidua TEXT)')

def create_womenshoestable():
  c.execute('CREATE TABLE IF NOT EXISTS womenshoestable(productname TEXT,price TEXT,Lokasi TEXT,stock TEXT,gambarproduct TEXT,gambarproductdua TEXT,gambarlokasi TEXT,gambarlokasidua TEXT)')  

def add_admindata(username,password):
  c.execute('INSERT INTO adminstable(username,password) VALUES (?,?)',(username,password))
  conn.commit()

def add_userdata(username,password):
  c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
  conn.commit()

def add_prodata(productname,price,lokasi,gambarproduct,gambarproductdua,gambarlokasi):
  c.execute('INSERT INTO prostabletiga(productname,price,lokasi,gambarproduct,gambarproductdua,gambarlokasi) VALUES (?,?,?,?,?,?)',(productname,price,lokasi,gambarproduct,gambarproductdua,gambarlokasi))
  conn.commit()

def add_shirtdata(productname,price,lokasi,stock,gambarproduct,gambarproductdua,gambarlokasi,gambarlokasidua):
  c.execute('INSERT INTO shirttable(productname,price,lokasi,stock,gambarproduct,gambarproductdua,gambarlokasi,gambarlokasidua) VALUES (?,?,?,?,?,?,?,?)',(productname,price,lokasi,stock,gambarproduct,gambarproductdua,gambarlokasi,gambarlokasidua))
  conn.commit()

def add_shirtdatareviseone(productname,price,lokasi,stock,gambarproduct,gambarproductdua):
  c.execute('INSERT INTO shirttablerevisethree(productname,price,lokasi,stock,gambarproduct,gambarproductdua) VALUES (?,?,?,?,?,?)',(productname,price,lokasi,stock,gambarproduct,gambarproductdua))
  conn.commit()

def add_tshirtdata(productname,price,lokasi,stock,gambarproduct,gambarproductdua,gambarlokasi,gambarlokasidua):
  c.execute('INSERT INTO tshirttable(productname,price,lokasi,stock,gambarproduct,gambarproductdua,gambarlokasi,gambarlokasidua) VALUES (?,?,?,?,?,?,?,?)',(productname,price,lokasi,stock,gambarproduct,gambarproductdua,gambarlokasi,gambarlokasidua))
  conn.commit()

def add_maleshoesdata(productname,price,lokasi,stock,gambarproduct,gambarproductdua,gambarlokasi,gambarlokasidua):
  c.execute('INSERT INTO maleshoestable(productname,price,lokasi,stock,gambarproduct,gambarproductdua,gambarlokasi,gambarlokasidua) VALUES (?,?,?,?,?,?,?,?)',(productname,price,lokasi,stock,gambarproduct,gambarproductdua,gambarlokasi,gambarlokasidua))
  conn.commit()

def add_dressdata(productname,price,lokasi,stock,gambarproduct,gambarproductdua,gambarlokasi,gambarlokasidua):
  c.execute('INSERT INTO dresstable(productname,price,lokasi,stock,gambarproduct,gambarproductdua,gambarlokasi,gambarlokasidua) VALUES (?,?,?,?,?,?,?,?)',(productname,price,lokasi,stock,gambarproduct,gambarproductdua,gambarlokasi,gambarlokasidua))
  conn.commit()

def add_womentshirtdata(productname,price,lokasi,stock,gambarproduct,gambarproductdua,gambarlokasi,gambarlokasidua):
  c.execute('INSERT INTO womentshirttable(productname,price,lokasi,stock,gambarproduct,gambarproductdua,gambarlokasi,gambarlokasidua) VALUES (?,?,?,?,?,?,?,?)',(productname,price,lokasi,stock,gambarproduct,gambarproductdua,gambarlokasi,gambarlokasidua))
  conn.commit()

def add_womenshoesdata(productname,price,lokasi,stock,gambarproduct,gambarproductdua,gambarlokasi,gambarlokasidua):
  c.execute('INSERT INTO womenshoestable(productname,price,lokasi,stock,gambarproduct,gambarproductdua,gambarlokasi,gambarlokasidua) VALUES (?,?,?,?,?,?,?,?)',(productname,price,lokasi,stock,gambarproduct,gambarproductdua,gambarlokasi,gambarlokasidua))
  conn.commit()

def login_admin(username,password):
  c.execute('SELECT * FROM adminstable WHERE username =? AND password = ?',(username,password))
  data = c.fetchall()
  return data

def get_shirt(productname):
  c.execute('SELECT * FROM shirttable WHERE productname="{}"'.format(productname))

def view_all_admins():
  c.execute('SELECT * FROM adminstable')
  data = c.fetchall()
  return data

def view_all_users():
  c.execute('SELECT * FROM userstable')
  data = c.fetchall()
  return data  

def view_all_product():
  c.execute('SELECT * FROM prostabletiga')
  data = c.fetchall()
  return data

def view_all_shirt():
  c.execute('SELECT * FROM shirttable')
  data = c.fetchall()
  return data

def view_all_shirtrevise():
  c.execute('SELECT * FROM shirttablerevisethree')
  data = c.fetchall()
  return data

def view_all_tshirt():
  c.execute('SELECT * FROM tshirttable')
  data = c.fetchall()
  return data

def view_all_maleshoes():
  c.execute('SELECT * FROM maleshoestable')
  data = c.fetchall()
  return data

def view_all_dress():
  c.execute('SELECT * FROM dresstable')
  data = c.fetchall()
  return data

def view_all_womentshirt():
  c.execute('SELECT * FROM womentshirttable')
  data = c.fetchall()
  return data

def view_all_womenshoes():
  c.execute('SELECT * FROM womenshoestable')
  data = c.fetchall()
  return data

def edit_shirts(update_stok,stok):
  c.execute("UPDATE shirttable SET stock=? WHERE stock=?",(update_stok,stok))
  conn.commit()
  data = c.fetchall()
  return data

def edit_tshirts(update_lokasi,update_stok,lokasi,stok):
  c.execute("UPDATE tshirttable SET lokasi=?, stock=? WHERE lokasi=? and stock=?",(update_lokasi,update_stok,lokasi,stok))
  conn.commit()
  data = c.fetchall()
  return data

def edit_maleshoes(update_stok,stok):
  c.execute("UPDATE maleshoestable SET stock=? WHERE stock=?",(update_stok,stok))
  conn.commit()
  data = c.fetchall()
  return data

def edit_dress(update_stok,stok):
  c.execute("UPDATE dresstable SET stock=? WHERE stock=?",(update_stok,stok))
  conn.commit()
  data = c.fetchall()
  return data

def edit_womentshirt(update_stok,stok):
  c.execute("UPDATE womentshirttable SET stock=? WHERE stock=?",(update_stok,stok))
  conn.commit()
  data = c.fetchall()
  return data

def edit_womenshoes(update_stok,stok):
  c.execute("UPDATE womenshoestable SET stock=? WHERE stock=?",(update_stok,stok))
  conn.commit()
  data = c.fetchall()
  return data
  
def delete_employees(username):
  c.execute('DELETE FROM userstable WHERE username="{}"'.format(username))
  conn.commit()

def delete_admins(username):
  c.execute('DELETE FROM adminstable WHERE username="{}"'.format(username)) 
  conn.commit() 

def delete_products(gambarproductdua):
  c.execute('DELETE FROM prostabletiga WHERE gambarproductdua="{}"'.format(gambarproductdua))
  conn.commit()

def delete_shirts(gambarproduct):
  c.execute('DELETE FROM shirttable WHERE gambarproduct="{}"'.format(gambarproduct))
  conn.commit()

def delete_tshirts(gambarproduct):
  c.execute('DELETE FROM tshirttable WHERE gambarproduct="{}"'.format(gambarproduct))
  conn.commit()

def delete_maleshoes(gambarproduct):
   c.execute('DELETE FROM maleshoestable WHERE gambarproduct="{}"'.format(gambarproduct))
   conn.commit()

def delete_dresses(gambarproduct):
   c.execute('DELETE FROM dresstable WHERE gambarproduct="{}"'.format(gambarproduct))
   conn.commit()

def delete_womentshirts(gambarproduct):
   c.execute('DELETE FROM womentshirttable WHERE gambarproduct="{}"'.format(gambarproduct))
   conn.commit()

def delete_womenshoes(gambarproduct):
   c.execute('DELETE FROM womenshoestable WHERE gambarproduct="{}"'.format(gambarproduct))
   conn.commit()

def search(username):
  c.execute("SELECT * FROM userstable WHERE username='%s'" % (username))
  data = c.fetchall()
  return data

def searchproduct(gambarproduct):
  c.execute("SELECT * FROM prostabletiga WHERE gambarproduct='%s'" % (gambarproduct))
  data = c.fetchall()
  return data

def searchshirt(productname):
  c.execute("SELECT * FROM shirttable WHERE productname='%s'" % (productname))
  data = c.fetchall()
  return data

def searchtshirt(productname):
  c.execute("SELECT * FROM tshirttable WHERE productname='%s'" % (productname))
  data = c.fetchall()
  return data

def searchmaleshoes(productname):
  c.execute("SELECT * FROM maleshoestable WHERE productname='%s'" % (productname))
  data = c.fetchall()
  return data

def searchdress(productname):
  c.execute("SELECT * FROM dresstable WHERE productname='%s'" % (productname))
  data = c.fetchall()
  return data

def searchwomentshirt(productname):
  c.execute("SELECT * FROM womentshirttable WHERE productname='%s'" % (productname))
  data = c.fetchall()
  return data

def searchwomenshoes(productname):
  c.execute("SELECT * FROM womenshoestable WHERE productname='%s'" % (productname))
  data = c.fetchall()
  return data

st.title("Fashion Finder System")

def main():

    menu = ["Login", "SignUp"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Login":
        st.subheader("Login Section")

        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password", type='password')
        if st.sidebar.checkbox("Login"):
            create_admintable()
            result = login_admin(username,password)
            if result:

              st.success("Logged In as {}".format(username))

              with st.sidebar:
                selectedone = option_menu(
                menu_title="",
                options=["Create", "Update","Delete", "Reload"],
                icons=["","", ""],
                menu_icon="cast",
                default_index=0,
                key="sideone"
              )

              if selectedone == "Create":

                task = st.selectbox("Task", ["Create New Image Shirt Storage", 
                "Create New Image T-Shirt Storage", "Create New Image Male Shoes Storage", 
                "Create New Image Dress Storage", "Create New Image Women T-Shirt Storage",
                "Create New Image Women Shoes Storage"])

                if task == "Create New Image Shirt Storage":
                  # admin_result = view_all_shirtrevise()
                  # if admin_result is not None:
                  #   clean_db = pd.DataFrame(admin_result)
                  #   st.table(clean_db)
                  # else:
                  #   st.write("")

                  new_productname = st.text_input("Product Name")
                  new_price = st.text_input("Price")
                  new_lokasi = st.selectbox("Location", ["", "Blok A", "Blok B", "Blok C", 
                  "Blok D", "Blok E", "Blok F", "Blok G", "Blok H", "Blok I","Blok J", "Blok K", 
                  "Blok L", "Blok M", "Blok N", "Blok O", "Blok P"])
                  new_stock = st.text_input("Stock")
                  uploaded_filesatu = st.file_uploader("Choose an image Product",key="file")
                  if uploaded_filesatu is not None:
                    new_gambarproduct = st.text_input("Product Images",(str(uploaded_filesatu.name)))
                    new_gambarproductfolder = st.text_input("Paste File Name Here", "shirt\\")
                  else:
                    st.write("")

                  if st.button("Input"):
                    open(os.path.join('shirt',uploaded_filesatu.name),'wb').write(uploaded_filesatu.getbuffer())
                    create_shirttablereviseone()
                    add_shirtdatareviseone(new_productname,new_price,new_lokasi,new_stock,new_gambarproduct,new_gambarproductfolder)
                    st.success("You have successfully created an New Product")
                    
                elif task == "Create New Image T-Shirt Storage":
                  admin_result = view_all_tshirt()
                  if admin_result is not None:
                    clean_db = pd.DataFrame(admin_result)
                    st.table(clean_db)

                  new_productname = st.text_input("Product Name")
                  new_price = st.text_input("Price")
                  new_lokasi = st.text_input("Location")
                  new_stock = st.text_input("Stock")
                  uploaded_filesatu = st.file_uploader("Choose an image Product",key="file")
                  if uploaded_filesatu is not None:
                    new_gambarproduct = st.text_input("Product Images",(str(uploaded_filesatu.name)))
                  else:
                    st.write("No Data")
                  new_gambarproductdua = st.text_input("Name of file name Product","tshirt\\")
                  uploaded_filedua = st.file_uploader("Choose an image Location",key="filedua")
                  if uploaded_filedua is not None:
                    new_gambarlokasi = st.text_input("Siteplan Images",(str(uploaded_filedua.name)))
                  else:
                    st.write("No Data")
                  new_gambarlokasidua = st.text_input("Name of file name Location","siteplan\\")

                  if st.button("Input"):
                    open(os.path.join('tshirt',uploaded_filesatu.name),'wb').write(uploaded_filesatu.getbuffer())
                    open(os.path.join('siteplan',uploaded_filedua.name),'wb').write(uploaded_filedua.getbuffer())
                    create_tshirttable()
                    add_tshirtdata(new_productname,new_price,new_lokasi,new_stock,new_gambarproduct,new_gambarproductdua,new_gambarlokasi,new_gambarlokasidua)
                    st.success("You have successfully created an New Product")

                elif task == "Create New Image Male Shoes Storage":
                  admin_result = view_all_maleshoes()
                  if admin_result is not None:
                    clean_db = pd.DataFrame(admin_result)
                    st.table(clean_db)

                  new_productname = st.text_input("Product Name")
                  new_price = st.text_input("Price")
                  new_lokasi = st.text_input("Location")
                  new_stock = st.text_input("Stock")
                  uploaded_filesatu = st.file_uploader("Choose an image",key="file")
                  if uploaded_filesatu is not None:
                    new_gambarproduct = st.text_input("Product Images",(str(uploaded_filesatu.name)))
                  else:
                    st.write("No Data")
                  new_gambarproductdua = st.text_input("Name of file image product","maleshoes\\")
                  uploaded_filedua = st.file_uploader("Choose an image",key="filedua")
                  if uploaded_filedua is not None:
                    new_gambarlokasi = st.text_input("Siteplan Images",(str(uploaded_filedua.name)))
                  else:
                    st.write("No Data")
                  new_gambarlokasidua = st.text_input("Name of file image Location","siteplan\\")

                  if st.button("Input"):
                    open(os.path.join('maleshoes',uploaded_filesatu.name),'wb').write(uploaded_filesatu.getbuffer())
                    open(os.path.join('siteplan',uploaded_filedua.name),'wb').write(uploaded_filedua.getbuffer())
                    create_maleshoestable()
                    add_maleshoesdata(new_productname,new_price,new_lokasi,new_stock,new_gambarproduct,new_gambarproductdua,new_gambarlokasi,new_gambarlokasidua)
                    st.success("You have successfully created an New Product")

                elif task == "Create New Image Dress Storage":
                  admin_result = view_all_dress()
                  if admin_result is not None:
                    clean_db = pd.DataFrame(admin_result)
                    st.table(clean_db)

                  new_productname = st.text_input("Product Name")
                  new_price = st.text_input("Price")
                  new_lokasi = st.text_input("Location")
                  new_stock = st.text_input("Stock")
                  uploaded_filesatu = st.file_uploader("Choose an image Product",key="file")
                  if uploaded_filesatu is not None:
                    new_gambarproduct = st.text_input("Product Images",(str(uploaded_filesatu.name)))
                  else:
                    st.write("No Data")
                  new_gambarproductdua = st.text_input("Name of file image Product","dress\\")
                  uploaded_filedua = st.file_uploader("Choose an image Location",key="filedua")
                  if uploaded_filedua is not None:
                    new_gambarlokasi = st.text_input("Siteplan Images",(str(uploaded_filedua.name)))
                  else:
                    st.write("No Data")
                  new_gambarlokasidua = st.text_input("Name of file image Location","siteplan\\")

                  if st.button("Input"):
                    open(os.path.join('dress',uploaded_filesatu.name),'wb').write(uploaded_filesatu.getbuffer())
                    open(os.path.join('siteplan',uploaded_filedua.name),'wb').write(uploaded_filedua.getbuffer())
                    create_dresstable()
                    add_dressdata(new_productname,new_price,new_lokasi,new_stock,new_gambarproduct,new_gambarproductdua,new_gambarlokasi,new_gambarlokasidua)
                    st.success("You have successfully created an New Product")

                elif task == "Create New Image Women T-Shirt Storage":
                  admin_result = view_all_womentshirt()
                  if admin_result is not None:
                    clean_db = pd.DataFrame(admin_result)
                    st.table(clean_db)

                  new_productname = st.text_input("Product Name")
                  new_price = st.text_input("Price")
                  new_lokasi = st.text_input("Location")
                  new_stock = st.text_input("Stock")
                  uploaded_filesatu = st.file_uploader("Choose an image product",key="file")
                  if uploaded_filesatu is not None:
                    new_gambarproduct = st.text_input("Images",(str(uploaded_filesatu.name)))
                  else:
                    st.write("No Data")
                  new_gambarproductdua = st.text_input("Name of file image product","womentshirt\\")
                  uploaded_filedua = st.file_uploader("Choose an image location",key="filedua")
                  if uploaded_filedua is not None:
                    new_gambarlokasi = st.text_input("Siteplan Images",(str(uploaded_filedua.name)))
                  else:
                    st.write("No Data")
                  new_gambarlokasidua = st.text_input("Name of file image location","siteplan\\")

                  if st.button("Input"):
                    open(os.path.join('womentshirt',uploaded_filesatu.name),'wb').write(uploaded_filesatu.getbuffer())
                    open(os.path.join('siteplan',uploaded_filedua.name),'wb').write(uploaded_filedua.getbuffer())
                    create_womentshirttable()
                    add_womentshirtdata(new_productname,new_price,new_lokasi,new_stock,new_gambarproduct,new_gambarproductdua,new_gambarlokasi,new_gambarlokasidua)
                    st.success("You have successfully created an New Product")
              
                elif task == "Create New Image Women Shoes Storage":
                  admin_result = view_all_womenshoes()
                  if admin_result is not None:
                    clean_db = pd.DataFrame(admin_result)
                    st.table(clean_db)


                  new_productname = st.text_input("Product Name")
                  new_price = st.text_input("Price")
                  new_lokasi = st.text_input("Location")
                  new_stock = st.text_input("Stock")
                  uploaded_filesatu = st.file_uploader("Choose an image product",key="file")
                  if uploaded_filesatu is not None:
                    new_gambarproduct = st.text_input("Images",(str(uploaded_filesatu.name)))
                  else:
                    st.write("No Data")
                  new_gambarproductdua = st.text_input("Name of file image product","womenshoes\\")
                  uploaded_filedua = st.file_uploader("Choose an image location",key="filedua")
                  if uploaded_filedua is not None:
                    new_gambarlokasi = st.text_input("Siteplan Images",(str(uploaded_filedua.name)))
                  else:
                    st.write("No Data")
                  new_gambarlokasidua = st.text_input("Name of file image location","siteplan\\")

                  if st.button("Input"):
                    open(os.path.join('womenshoes',uploaded_filesatu.name),'wb').write(uploaded_filesatu.getbuffer())
                    open(os.path.join('siteplan',uploaded_filedua.name),'wb').write(uploaded_filedua.getbuffer())
                    create_womenshoestable()
                    add_womenshoesdata(new_productname,new_price,new_lokasi,new_stock,new_gambarproduct,new_gambarproductdua,new_gambarlokasi,new_gambarlokasidua)
                    st.success("You have successfully created an New Product")


                elif task == "Create New Image Storage":
        
                  st.subheader("Create New Image Storage")
                  image_files = glob.glob("images/*.jpg")
                  manuscripts = []
                  for image_file in image_files:
                    image_file = image_file.replace("\\","/")
                    parts = image_file.split("/")
                    if parts[1] not in manuscripts:
                       manuscripts.append(parts[1])
                  manuscripts.sort()
                  # st.write(manuscripts)
                  view_manuscripts_images = st.multiselect("Select Manuscripts(s)", manuscripts)
                  n = st.number_input("Select Grid Width", 1, 10, 5)
                  view_images = []
                  for image_file in image_files:
                    if any(manuscripts in image_file for manuscripts in view_manuscripts_images):
                      view_images.append(image_file)
                  groups = []
                  for i in range(0, len(view_images), n):
                    groups.append(view_images[i:i+n])
                  # tes = view_all_product()  
                  # for i in tes:
                  #   productname = i[0]
                  #   price = i[1]
                  #   locate = i[2]
                  #   gambarproduct = i[3]
                  #   gambarproductdua = i[4]
                  #   gambarlokasi = i[5]
                  #   st.write(productname)

                  for group in groups:
                    cols = st.columns(n)
                    for i, image_file in enumerate(group):
                      cols[i].image(image_file)
                      button = cols[i].button("Click",key=image_file)
                      if button:
                        productnamena = cols[i].write(image_file)
                      # search_result = searchproduct(productnamena)
                      # for i in search_result:
                      #   productname = i[0]
                      #   price = i[1]
                      #   locate = i[2]
                      #   gambarproduct = i[3]
                      #   gambarproductdua = i[4]
                      #   gambarlokasi = i[5]
                      # st.write(productname,gambarproductdua)

                  
                    # cols[i].write(image_file)

                    # search_result = searchproduct(image_file)
                    # for i in search_result:
                    #   global productname
                    #   productname = i[0]
                    #   price = i[1]
                    #   locate = i[2]
                    #   gambarproduct = i[3]
                    #   gambarproductdua = i[4]
                    #   gambarlokasi = i[5]
                    
                    # cols[i].write(productname)
                    # st.write(i, image_file)

                  st.subheader("Create New Product")

                  admin_result = view_all_product()
                  if admin_result is not None:
                    clean_db = pd.DataFrame(admin_result,columns=["Product Name","Price","Lokasi","Gambar Product","Gambar Product (1)","Gambar Lokasi"])
                    st.table(clean_db)

                  else:
                    st.header("Not found") 

                # product_result = view_all_product()
                # st.write(product_result)
                # for i in product_result:
                #   product = i[0]
                #   price = i[1]
                #   lokasi = i[2]
                #   gambarproduk = i[3]
                #   gambarprodukdua = i[4]
                #   gambarlokasi = i[5]
                #   st.write(product,price,lokasi,gambarproduk,gambarprodukdua,gambarlokasi)

                  new_productname = st.text_input("Product Name")
                  new_price = st.text_input("Price")
                  new_lokasi = st.text_input("Location")
                  uploaded_filesatu = st.file_uploader("Choose an image")
                  if uploaded_filesatu is not None:
                    new_gambarproductdua = st.text_input("Images",(str(uploaded_filesatu.name)))
                  else:
                    st.write("No Data")
                  new_gambarproduct = st.text_input("","images\\")
                  new_gambarlokasi = st.text_input("","siteplan\\")
                # new_gambarproductproduct = st.text_input("","images\\",new_gambarproduct)
                # new_gambarproductduadua = st.text_input(new_gambarproductdua)
                # new_gambarlokasilokasi = st.text_input("","siteplan\\",new_gambarlokasi)

                  if st.button("Input"):
                    open(os.path.join('images',uploaded_filesatu.name),'wb').write(uploaded_filesatu.getbuffer())
                    create_producttable()
                    add_prodata(new_productname,new_price,new_lokasi,new_gambarproduct,new_gambarproductdua,new_gambarlokasi)
                    st.success("You have successfully created an New Product")

                # uploaded_file = st.file_uploader("Choose an image")
                # if uploaded_file is not None:
                #   # display the file
                #    open(os.path.join('images',uploaded_file.name),'wb').write(uploaded_file.getbuffer())

              elif selectedone == "Update":
                taskupdate = st.selectbox("TaskUpdate",["Update Shirt","Update T-Shirt",
                "Update Shoes","Update Dress","Update Women T-Shirt","Update Women Shoes"])

                if taskupdate == "Update Shirt":
                  list_of_shirt = [i[0] for i in view_all_shirt()]
                  selected_task = st.selectbox("Choose",list_of_shirt)

                  selected_result = searchshirt(selected_task)
                  st.write(selected_result)
                  if selected_result:
                    stok = selected_result[0][3]

                    update_stok = st.text_input("Update Stok",stok)

                    if st.button("Update Shirt"):
                      edit_shirts(update_stok,stok)
                      st.success("Successfully Added Data")

                elif taskupdate == "Update T-Shirt":
                  list_of_tshirt = [i[0] for i in view_all_tshirt()]
                  selected_task = st.selectbox("Choose",list_of_tshirt)

                  selected_result = searchtshirt(selected_task)
                  st.write(selected_result)
                  if selected_result:
                    stok = selected_result[0][3]
                    lokasi = selected_result[0][2]
                    
                    update_lokasi = st.selectbox("Location", [lokasi, "Blok A", "Blok B", "Blok C", 
                  "Blok D", "Blok E", "Blok F", "Blok G", "Blok H", "Blok I","Blok J", "Blok K", 
                  "Blok L", "Blok M", "Blok N", "Blok O", "Blok P"])
                    # update_lokasi = st.text_input("Update Lokasi",lokasi)
                    update_stok = st.text_input("Update Stok",stok)

                    if st.button("Update Shirt"):
                      edit_tshirts(update_lokasi,update_stok,lokasi,stok)
                      st.success("Successfully Added Data")

                elif taskupdate == "Update Shoes":
                  list_of_maleshoes = [i[0] for i in view_all_maleshoes()]
                  selected_task = st.selectbox("Choose",list_of_maleshoes)

                  selected_result = searchmaleshoes(selected_task)
                  st.write(selected_result)
                  if selected_result:
                    stok = selected_result[0][3]

                    update_stok = st.text_input("Update Stok",stok)

                    if st.button("Update Shoes"):
                      edit_maleshoes(update_stok,stok)
                      st.success("Successfully Added Data")

                elif taskupdate == "Update Dress":
                  list_of_dress = [i[0] for i in view_all_dress()]
                  selected_task = st.selectbox("Choose",list_of_dress)

                  selected_result = searchdress(selected_task)
                  st.write(selected_result)
                  if selected_result:
                    stok = selected_result[0][3]

                    update_stok = st.text_input("Update Stok",stok)

                    if st.button("Update Dress"):
                      edit_dress(update_stok,stok)
                      st.success("Successfully Added Data")

                elif taskupdate == "Update Women T-Shirt":
                  list_of_womentshirt = [i[0] for i in view_all_womentshirt()]
                  selected_task = st.selectbox("Choose",list_of_womentshirt)

                  selected_result = searchwomentshirt(selected_task)
                  st.write(selected_result)
                  if selected_result:
                    stok = selected_result[0][3]

                    update_stok = st.text_input("Update Stok",stok)

                    if st.button("Update Women T-Shirt"):
                      edit_womentshirt(update_stok,stok)
                      st.success("Successfully Added Data")

                elif taskupdate == "Update Women Shoes":
                  list_of_womenshoes = [i[0] for i in view_all_womenshoes()]
                  selected_task = st.selectbox("Choose",list_of_womenshoes)

                  selected_result = searchwomenshoes(selected_task)
                  st.write(selected_result)
                  if selected_result:
                    stok = selected_result[0][3]

                    update_stok = st.text_input("Update Stok",stok)

                    if st.button("Update Women Shoes"):
                      edit_womenshoes(update_stok,stok)
                      st.success("Successfully Added Data")

              elif selectedone == "Delete":
                task = st.selectbox("Task", ["Remove Shirt Image Storage",
                "Remove T-Shirt Image Storage","Remove Shoes Image Storage",
                "Remove Dress Image Storage","Remove Women T-Shirt Image Storage",
                "Remove Women Shoes Image Storage","Remove Image Uploads Storage", 
                "Remove Admin Account"])

                if task == "Remove Shirt Image Storage":     
                  #Remove Storage 
                  st.subheader("Remove Shirt Image Storage")
                  view_product = view_all_shirt()    
                  if view_product is not None:
                    clean_db = pd.DataFrame(view_product,columns=["Product Name","Price",
                    "Location","stok","Gambar Produk","gambar produk dua","gambar lokasi",
                    "gambar lokasi dua"])
                    st.table(clean_db)

                  else:
                    st.header("Not found")

                  productdelete = st.text_input("Remove Filename Storage",".jpg")
                
                  if st.button("Remove Shirt Image Storage"):
                    path = os.path.join('shirt', productdelete)
                    pathtwo = os.path.join('siteplan',productdelete)
                    delete_shirts(productdelete)
                    st.warning("Delete: '{}'".format(productdelete))
                    st.success("You have successfully")
                    os.remove(path)
                    os.remove(pathtwo)

                elif task == "Remove T-Shirt Image Storage":
                  st.subheader("Remove T-Shirt Image Storage")
                  view_product = view_all_tshirt()    
                  if view_product is not None:
                    clean_db = pd.DataFrame(view_product,columns=["Product Name","Price",
                    "Location","stok","Gambar Produk","gambar produk dua","gambar lokasi",
                    "gambar lokasi dua"])
                    st.table(clean_db)

                  else:
                    st.header("Not found")

                  productdelete = st.text_input("Remove Filename Storage",".jpg")
                
                  if st.button("Remove T-Shirt Image Storage"):
                    path = os.path.join('tshirt', productdelete)
                    pathtwo = os.path.join('siteplan', productdelete)
                    delete_tshirts(productdelete)
                    st.warning("Delete: '{}'".format(productdelete))
                    st.success("You have successfully")
                    os.remove(path)
                    os.remove(pathtwo)

                elif task == "Remove Shoes Image Storage":
                  st.subheader("Remove Shoes Image Storage")
                  view_product = view_all_maleshoes()    
                  if view_product is not None:
                    clean_db = pd.DataFrame(view_product,columns=["Product Name","Price",
                    "Location","stok","Gambar Produk","gambar produk dua","gambar lokasi",
                    "gambar lokasi dua"])
                    st.table(clean_db)

                  else:
                    st.header("Not found")

                  productdelete = st.text_input("Remove Filename Storage",".jpg")
                
                  if st.button("Remove Shoes Image Storage"):
                    path = os.path.join('maleshoes', productdelete)
                    pathtwo = os.path.join('siteplan',productdelete)
                    delete_maleshoes(productdelete)
                    st.warning("Delete: '{}'".format(productdelete))
                    st.success("You have successfully")
                    os.remove(path)
                    os.remove(pathtwo)

                elif task == "Remove Dress Image Storage":
                  st.subheader("Remove Dress Image Storage")
                  view_product = view_all_dress()    
                  if view_product is not None:
                    clean_db = pd.DataFrame(view_product,columns=["Product Name","Price",
                    "Location","stok","Gambar Produk","gambar produk dua","gambar lokasi",
                    "gambar lokasi dua"])
                    st.table(clean_db)

                  else:
                    st.header("Not found")

                  productdelete = st.text_input("Remove Filename Storage",".jpg")
                
                  if st.button("Remove Dress Image Storage"):
                    path = os.path.join('dress', productdelete)
                    pathtwo = os.path.join('siteplan', productdelete)
                    delete_dresses(productdelete)
                    st.warning("Delete: '{}'".format(productdelete))
                    st.success("You have successfully")
                    os.remove(path)
                    os.remove(pathtwo)

                elif task == "Remove Women T-Shirt Image Storage":
                  st.subheader("Remove Women T-Shirt Image Storage")
                  view_product = view_all_womentshirt()    
                  if view_product is not None:
                    clean_db = pd.DataFrame(view_product,columns=["Product Name","Price",
                    "Location","stok","Gambar Produk","gambar produk dua","gambar lokasi",
                    "gambar lokasi dua"])
                    st.table(clean_db)

                  else:
                    st.header("Not found")

                  productdelete = st.text_input("Remove Filename Storage",".jpg")
                
                  if st.button("Remove Image Storage"):
                    path = os.path.join('womentshirt', productdelete)
                    pathtwo = os.path.join('siteplan', productdelete)
                    delete_womentshirts(productdelete)
                    st.warning("Delete: '{}'".format(productdelete))
                    st.success("You have successfully")
                    os.remove(path)
                    os.remove(pathtwo)

                elif task == "Remove Women Shoes Image Storage":
                  st.subheader("Remove Women Shoes Image Storage")
                  view_product = view_all_womenshoes()    
                  if view_product is not None:
                    clean_db = pd.DataFrame(view_product,columns=["Product Name","Price",
                    "Location","stok","Gambar Produk","gambar produk dua","gambar lokasi",
                    "gambar lokasi dua"])
                    st.table(clean_db)

                  else:
                    st.header("Not found")

                  productdelete = st.text_input("Remove Filename Storage",".jpg")
                
                  if st.button("Remove Image Storage"):
                    path = os.path.join('womenshoes', productdelete)
                    pathtwo = os.path.join('siteplan', productdelete)
                    delete_womenshoes(productdelete)
                    st.warning("Delete: '{}'".format(productdelete))
                    st.success("You have successfully")
                    os.remove(path)
                    os.remove(pathtwo)
 
                elif task == "Remove Image Uploads Storage":   
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

                  if st.button("Remove All Image Uploads"):
                    path = os.path.join('uploads',manuscripts[0])
                    st.success("You have successfully")
                    os.remove(path)
                
                  #Remove Admin Account    
                elif task == "Remove Admin Account":
                  st.subheader("Remove Admin Account")
                  admindelete = st.text_input("Admin Username")
                  if st.button("Remove Admin Account"):
                    delete_admins(admindelete)
                    st.warning("Delete: '{}'".format(admindelete)) 

                  admin_result = view_all_admins()
                  clean_db = pd.DataFrame(admin_result,columns=["Username","Password"])
                  st.table(clean_db)    
              
              elif selectedone == "Reload":
                task = st.selectbox("Task", ["Reload Shirts Image Storage", 
                "Reload T-Shirts Image Storage", "Reload Male Shoes Image Storage", 
                "Reload Dress Image Storage", "Reload Women T-Shirts Image Storage", 
                "Reload Women Shoes Image Storage"])

                  
                if task == "Reload Shirts Image Storage":
                  st.subheader("Reload Shirts Image Storage")
                  image_files = glob.glob("shirt/*.jpg")
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
                     for file in os.listdir('shirt'):
                       filenames.append(os.path.join('shirt',file))
                     feature_list = []
                     for file in filenames:
                       feature_list.append(extract_features(file,model))
                       pickle.dump(feature_list,open('embeddingsshirt.pkl','wb'))
                       pickle.dump(filenames,open('filenamesshirt.pkl','wb'))
                      
                       st.write("Reload", file)
                     st.write("Reload Success") 

                elif task == "Reload T-Shirts Image Storage":
                  st.subheader("Reload T-Shirts Image Storage")
                  image_files = glob.glob("tshirt/*.jpg")
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
                     for file in os.listdir('tshirt'):
                     # for file in os.listdir('images'):
                       filenames.append(os.path.join('tshirt',file))
                     # filenames.append(os.path.join('images',file))
                     feature_list = []
                     for file in tqdm(filenames):
                       feature_list.append(extract_features(file,model))
                       pickle.dump(feature_list,open('embeddingstshirt.pkl','wb'))
                      # pickle.dump(feature_list,open('embeddings.pkl','wb'))
                       pickle.dump(filenames,open('filenamestshirt.pkl','wb'))
                      # pickle.dump(filenames,open('filenames.pkl','wb'))
                      
                       st.write("Reload", file)
                     st.write("Reload Success")
                
                elif task == "Reload Male Shoes Image Storage":
                  st.subheader("Reload Male Shoes Image Storage")
                  image_files = glob.glob("maleshoes/*.jpg")
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
                     for file in os.listdir('maleshoes'):
                     # for file in os.listdir('images'):
                       filenames.append(os.path.join('maleshoes',file))
                     # filenames.append(os.path.join('images',file))
                     feature_list = []
                     for file in tqdm(filenames):
                       feature_list.append(extract_features(file,model))
                       pickle.dump(feature_list,open('embeddingsmaleshoes.pkl','wb'))
                      # pickle.dump(feature_list,open('embeddings.pkl','wb'))
                       pickle.dump(filenames,open('filenamesmaleshoes.pkl','wb'))
                      # pickle.dump(filenames,open('filenames.pkl','wb'))
                      
                       st.write("Reload", file)
                     st.write("Reload Success")

                elif task == "Reload Dress Image Storage":
                  st.subheader("Reload Dress Image Storage")
                  image_files = glob.glob("dress/*.jpg")
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
                     for file in os.listdir('dress'):
                     # for file in os.listdir('images'):
                       filenames.append(os.path.join('dress',file))
                     # filenames.append(os.path.join('images',file))
                     feature_list = []
                     for file in tqdm(filenames):
                       feature_list.append(extract_features(file,model))
                       pickle.dump(feature_list,open('embeddingsdress.pkl','wb'))
                      # pickle.dump(feature_list,open('embeddings.pkl','wb'))
                       pickle.dump(filenames,open('filenamesdress.pkl','wb'))
                      # pickle.dump(filenames,open('filenames.pkl','wb'))
                      
                       st.write("Reload", file)
                     st.write("Reload Success")

                elif task == "Reload Women T-Shirts Image Storage":
                  st.subheader("Reload Women T-Shirts Image Storage")
                  image_files = glob.glob("womentshirt/*.jpg")
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
                     for file in os.listdir('womentshirt'):
                     # for file in os.listdir('images'):
                       filenames.append(os.path.join('womentshirt',file))
                     # filenames.append(os.path.join('images',file))
                     feature_list = []
                     for file in tqdm(filenames):
                       feature_list.append(extract_features(file,model))
                       pickle.dump(feature_list,open('embeddingswomentshirt.pkl','wb'))
                      # pickle.dump(feature_list,open('embeddings.pkl','wb'))
                       pickle.dump(filenames,open('filenameswomentshirt.pkl','wb'))
                      # pickle.dump(filenames,open('filenames.pkl','wb'))
                      
                       st.write("Reload", file)
                     st.write("Reload Success")

                elif task == "Reload Women Shoes Image Storage":
                   st.subheader("Reload Women Shoes Image Storage")
                   image_files = glob.glob("womenshoes/*.jpg")
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
                     for file in os.listdir('womenshoes'):
                     # for file in os.listdir('images'):
                       filenames.append(os.path.join('womenshoes',file))
                     # filenames.append(os.path.join('images',file))
                     feature_list = []
                     for file in tqdm(filenames):
                       feature_list.append(extract_features(file,model))
                       pickle.dump(feature_list,open('embeddingswomenshoes.pkl','wb'))
                      # pickle.dump(feature_list,open('embeddings.pkl','wb'))
                       pickle.dump(filenames,open('filenameswomenshoes.pkl','wb'))
                      # pickle.dump(filenames,open('filenames.pkl','wb'))
                      
                       st.write("Reload", file)
                     st.write("Reload Success")
                  
            else:
                st.warning("Incorrect Username/Password")

    elif choice == "SignUp":
        st.subheader("Create New Account")
        new_user = st.text_input("Username")
        new_password = st.text_input("Password",type='password')

        if st.button("Signup"):
            create_admintable()
            add_admindata(new_user,new_password)
            st.success("You have successfully created an valid Account")
            st.info("Go to Login Menu to login")            


if __name__ == '__main__':
    main()