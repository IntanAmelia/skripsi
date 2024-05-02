import numpy as np
from sklearn.cluster import DBSCAN
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

df5=pd.read_csv('dataset_imputasi_finish_encode_1.csv')

df_copy = df5.copy()
df_copy = df_copy[['genre_encode','type_encoder','rating']]

# Memilih kolom yang akan digunakan untuk clustering
# Gantilah ['fitur_1', 'fitur_2', 'fitur_3'] dengan nama fitur yang sesuai dalam dataset Anda
X = df_copy[['genre_encode','type_encoder','rating']].values

# Membagi data menjadi data pelatihan dan data pengujian
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Membuat objek DBSCAN
# Sesuaikan nilai epsilon (eps) dan min_samples sesuai kebutuhan Anda
dbscan = DBSCAN(eps=0.9, min_samples=3)

# Melakukan clustering pada data pelatihan
labels_train = dbscan.fit_predict(X_train)

# Melakukan clustering pada data pengujian
labels_test = dbscan.fit_predict(X_test)

# Menambahkan kolom label ke data pelatihan dan pengujian
data_train = pd.DataFrame(X_train, columns=['genre_encode','type_encoder','rating'])
data_train['Cluster'] = labels_train

data_test = pd.DataFrame(X_test, columns=['genre_encode','type_encoder','rating'])
data_test['Cluster'] = labels_test

# Simpan model ke dalam file
filename = 'dbscan_model.pkl'
pickle.dump(dbscan, open(filename,'wb'))

