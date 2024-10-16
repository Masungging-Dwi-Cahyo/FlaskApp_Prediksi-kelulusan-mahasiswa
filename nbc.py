import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load dataset dari file Excel
dataset = pd.read_excel("DATASET.xlsx", sheet_name='DATASHEET')

# Menghapus kolom yang tidak diperlukan dari dataset
df = dataset.drop(['NO', 'NIM', 'NAMA', 'PROGRAM STUDI', 'IPS_5', 'IPS_6', 'IPK LULUS', 'NILAI KARYA TULIS'], axis=1)

# Memisahkan fitur (x) dan label (y) dari dataset
x = df.iloc[:, :-1].values  # Matriks fitur (semua kolom kecuali kolom terakhir)
y = df['LABEL KELAS']       # Label (kolom 'LABEL KELAS')

# Inisialisasi objek Standard Scaler untuk normalisasi fitur
scaler = StandardScaler()

# Menerapkan standar scaler pada fitur
x_scaled = scaler.fit_transform(x)

# Mengganti nilai NaN dengan rata-rata kolom
imputer = SimpleImputer(strategy='mean')
x_imputed = imputer.fit_transform(x_scaled)

# Mengaplikasikan PCA dengan 5 komponen utama
pca = PCA(n_components=5)
pca_dataset = pca.fit_transform(x_imputed)

# Membagi dataset menjadi data pelatihan 80% dan data pengujian 20%
x_train, x_test, y_train, y_test = train_test_split(pca_dataset, y, test_size=0.20, random_state=42)

# Membuat objek Gaussian Naive Bayes (NBC)
nbc = GaussianNB()

# Melatih model NBC dengan data pelatihan
model = nbc.fit(x_train, y_train)

def predict(data_test):
    # Normalisasi dan menerapkan PCA pada data uji
    pca_test = pca.transform(scaler.transform(data_test))

    # Membuat prediksi menggunakan model yang sudah dilatih
    y_pred = model.predict(pca_test)

    # Mengembalikan hasil prediksi dalam bentuk deskriptif
    if y_pred == 1:
        return "Mahasiswa lulus dalam 3 tahun dengan IPK ≥ 3,00"
    elif y_pred == 2:
        return "Mahasiswa lulus dalam 3 tahun dengan IPK < 3,00"
    elif y_pred == 3:
        return "Mahasiswa lulus lebih dari 3 tahun dengan IPK ≥ 3,00"
    else:
        return "Mahasiswa lulus lebih dari 3 tahun dengan IPK < 3,00"

def get_accuracy():
    # Membuat prediksi pada data pengujian
    y_prediction = model.predict(x_test)

    # Menghitung dan mengembalikan akurasi model
    return accuracy_score(y_test, y_prediction)

def get_y_true():
    # Mengembalikan nilai sebenarnya dari data pengujian
    return y_test

def get_y_pred():
    # Membuat prediksi pada data pengujian
    return model.predict(x_test)
