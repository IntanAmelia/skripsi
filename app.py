# app.py
#import library

import pickle
import streamlit as st
import pandas as pd
import numpy as np
import keras
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import math
import joblib

from keras.optimizers import Adam
from keras.layers import Dense, LSTM, Dropout
from keras.models import load_model

def main():
    st.set_page_config(
    page_title="PREDIKSI CURAH HUJAN MENGGUNAKAN LSTM DAN K-NN DALAM IMPUTASI MISSING VALUE"
)
    st.title('PREDIKSI CURAH HUJAN MENGGUNAKAN LSTM DAN K-NN DALAM IMPUTASI MISSING VALUE')
    
    tab1, tab2, tab3, tab4 = st.tabs(["Data Understanding", "Imputasi Missing Value Menggunakan KNN", "Hapus Data yang terdapat Missing Value", "Prediksi Selanjutnya"])

    with tab1:
        st.write("""
        <h5>Data Understanding</h5>
        <br>
        """, unsafe_allow_html=True)

        st.markdown("""
        Link Dataset:
        https://dataonline.bmkg.go.id
        """, unsafe_allow_html=True)


        st.write('Dataset ini berisi tentang curah hujan')
        missing_values = ['8888']
        df = pd.read_excel('Dataset_Curah_Hujan.xlsx', na_values = missing_values)
        st.write("Dataset Curah Hujan : ")
        st.write(df)
        
    with tab2:
        st.write("""
        <h5>Imputasi Missing Value Menggunakan KNN</h5>
        <br>
        """, unsafe_allow_html=True)
        st.write("""
        Pada skenario ini akan dibagi menjadi beberapa parameter, yakni sebagai berikut : 
        <ol>
        <li> K = 3; batch size = 32; hidden layer = 100; learning rate = 0.01; epoch = 12; time step = 25 </li>
        <li> K = 4; batch size = 32; hidden layer = 100; learning rate = 0.001; epoch = 25; time step = 50 </li>
        <li> K = 5; batch size = 32; hidden layer = 100; learning rate = 0.0001; epoch = 50; time step = 75 </li>
        </ol>
        """,unsafe_allow_html=True)

        preprocessing = st.radio(
        "Preprocessing Data",
        ('K = 3; batch size = 32; hidden layer = 100; learning rate = 0.01; epoch = 12; time step = 25',
         'K = 4; batch size = 32; hidden layer = 100; learning rate = 0.001; epoch = 25; time step = 50',
         'K = 5; batch size = 32; hidden layer = 100; learning rate = 0.0001; epoch = 50; time step = 75'))
        if preprocessing == 'K = 3; batch size = 32; hidden layer = 100; learning rate = 0.01; epoch = 12; time step = 25':
            # Load model
            # model = joblib.load('model_lstm_knn_s1_j.pkl')
            # model = load_model('model_lstm_knn_s1.h5')
            # model = keras.models.load_model('model_lstm_knn_s1.keras')
            model = pickle.load(open('thyroidmodel (2).sav', 'rb'))
            st.write(model.summary())
            
            # # Memuat data testing (x_test)
            # x_test = pd.read_csv('x_test_knn_s2.csv')
            
            # # Melakukan prediksi
            # predictions = model.predict(x_test)
            
            # # Menampilkan hasil prediksi
            # st.write("Hasil Prediksi:")
            # st.write(predictions)
            
        # elif preprocessing == 'K = 4; batch size = 32; hidden layer = 100; learning rate = 0.001; epoch = 25; time step = 50':
            
        # elif preprocessing == 'K = 5; batch size = 32; hidden layer = 100; learning rate = 0.0001; epoch = 50; time step = 75':
            
    # with tab3:
    #     st.write("""
    #     <h5>Modelling</h5>
    #     <br>
    #     """, unsafe_allow_html=True)
    #     df5=pd.read_csv('dataset_imputasi_finish_encode_1.csv')
    #     df_copy = df5.copy()
    #     df_copy = df_copy[['genre_encode','type_encoder','rating']]

    #     # Memilih kolom yang akan digunakan untuk clustering
    #     # Gantilah ['fitur_1', 'fitur_2', 'fitur_3'] dengan nama fitur yang sesuai dalam dataset Anda
    #     X = df_copy[['genre_encode','type_encoder','rating']].values

    #     # Membagi data menjadi data pelatihan dan data pengujian
    #     X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    #     # Membuat objek DBSCAN
    #     # Sesuaikan nilai epsilon (eps) dan min_samples sesuai kebutuhan Anda
    #     dbscan = DBSCAN(eps=0.9, min_samples=3)

    #     # Melakukan clustering pada data pelatihan
    #     labels_train = dbscan.fit_predict(X_train)

    #     # Melakukan clustering pada data pengujian
    #     labels_test = dbscan.fit_predict(X_test)

    #     # Menambahkan kolom label ke data pelatihan dan pengujian
    #     data_train = pd.DataFrame(X_train, columns=['genre_encode','type_encoder','rating'])
    #     data_train['Cluster'] = labels_train

    #     data_test = pd.DataFrame(X_test, columns=['genre_encode','type_encoder','rating'])
    #     data_test['Cluster'] = labels_test


    #     # Menampilkan hasil  cluster dan nilai silhouette score
    #     col1, col2 =st.columns(2)

    #     with col1:
    #         st.write("Hasil clustering dengan data pelatihan:")
    #         st.write(data_train)

    #         score_train = silhouette_score(data_train, data_train['Cluster'])
    #         st.write("Silhouette score train : {}".format(score_train))

    #     with col2:
    #         st.write("Hasil clustering dengan data pengujian:")
    #         st.write(data_test)

    #         score_test= silhouette_score(data_test, data_test['Cluster'])
    #         st.write("Silhouette score test : {}".format(score_test))

    # with tab4:

    #     # Menampilkan hasil clustering pada data pelatihan
    #     fig_train = plt.figure(figsize=(10, 8))
    #     ax_train = fig_train.add_subplot(111, projection='3d')

    #     ax_train.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=labels_train, cmap='viridis', marker='o', s=50)

    #     ax_train.set_title('DBSCAN Clustering - Data Pelatihan')
    #     ax_train.set_xlabel('Fitur 1')
    #     ax_train.set_ylabel('Fitur 2')
    #     ax_train.set_zlabel('Fitur 3')


    #     st.pyplot(fig_train)

    #     # Menampilkan hasil clustering pada data pelatihan
    #     fig_test = plt.figure(figsize=(10, 8))
    #     ax_test = fig_test.add_subplot(111, projection='3d')

    #     ax_test.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=labels_test, cmap='viridis', marker='o', s=50)

    #     ax_test.set_title('DBSCAN Clustering - Data Pelatihan')
    #     ax_test.set_xlabel('Fitur 1')
    #     ax_test.set_ylabel('Fitur 2')
    #     ax_test.set_zlabel('Fitur 3')


    #     st.pyplot(fig_test)
    
        
if __name__ == "__main__":
    main()
