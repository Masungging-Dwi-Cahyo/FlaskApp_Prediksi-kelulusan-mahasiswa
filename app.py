from flask import Flask, render_template, request, redirect, url_for
from flask_mysqldb import MySQL
from nbc import predict as predict_nbc, get_accuracy as get_accuracy_nbc, get_y_true as get_y_true_nbc, get_y_pred as get_y_pred_nbc
from sklearn.metrics import confusion_matrix
import numpy as np

app = Flask(__name__)

# Konfigurasi database
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'db_users'

mysql = MySQL(app)

def ensure_confusion_matrix_size(cm, num_classes=4):
    if cm.shape[0] < num_classes:
        new_cm = np.zeros((num_classes, num_classes), dtype=int)
        new_cm[:cm.shape[0], :cm.shape[1]] = cm
        return new_cm
    return cm

# Routes
@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
        user = cur.fetchone()
        cur.close()

        if user:
            return redirect(url_for('home'))  # Mengarahkan pengguna ke halaman home setelah login sukses
        else:
            error = "Username atau password salah !"

    return render_template("login.html", error=error)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        ulangpass = request.form["ulangpass"]
        email = request.form["email"]
        nama = request.form["nama"]
        jekel = request.form["jekel"]

        if password != ulangpass:
            error = "Password dan Ulang Password tidak cocok !"
            return render_template("register.html", error=error)

        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users (username, password, email, nama, jekel) VALUES (%s, %s, %s, %s, %s)",
                    (username, password, email, nama, jekel))
        mysql.connection.commit()
        cur.close()

        return redirect(url_for('login'))  # Mengarahkan pengguna ke halaman login setelah registrasi sukses
    
    return render_template("register.html")

@app.route("/home")
def home():
    return render_template("index.html")

# Route untuk melakukan prediksi dengan metode nbc
@app.route('/predict_nbc', methods=['POST'])
def prediction_nbc():
    # Ambil data dari form input
    JENIS_KELAMIN = int(request.form['JENIS KELAMIN'])
    ASAL_DAERAH = int(request.form['ASAL DAERAH'])
    IPS_1 = float(request.form['IPS_1'])
    IPS_3 = float(request.form['IPS_3'])
    IPS_2 = float(request.form['IPS_2'])
    IPS_4 = float(request.form['IPS_4'])
    SEROLOGI_GOLONGAN_DARAH_I = int(request.form['SEROLOGI GOLONGAN DARAH I'])
    SEROLOGI_GOLONGAN_DARAH_III = int(request.form['SEROLOGI GOLONGAN DARAH III'])
    SEROLOGI_GOLONGAN_DARAH_II = int(request.form['SEROLOGI GOLONGAN DARAH II'])
    SEROLOGI_GOLONGAN_DARAH_IV = int(request.form['SEROLOGI GOLONGAN DARAH IV'])
    INFEKSI_MENULAR_LEWAT_TRANSFUSI_DARAH_I = int(request.form['INFEKSI MENULAR LEWAT TRANSFUSI DARAH I'])
    INFEKSI_MENULAR_LEWAT_TRANSFUSI_DARAH_III = int(request.form['INFEKSI MENULAR LEWAT TRANSFUSI DARAH III'])
    INFEKSI_MENULAR_LEWAT_TRANSFUSI_DARAH_II = int(request.form['INFEKSI MENULAR LEWAT TRANSFUSI DARAH II'])
    INFEKSI_MENULAR_LEWAT_TRANSFUSI_DARAH_IV = int(request.form['INFEKSI MENULAR LEWAT TRANSFUSI DARAH IV'])
    PENYADAPAN_DARAH = int(request.form['PENYADAPAN DARAH'])
    KOMPONEN_DARAH = int(request.form['KOMPONEN DARAH'])

    # Buat data test dari input form
    data_test = [[JENIS_KELAMIN, ASAL_DAERAH, IPS_1, IPS_3, IPS_2, IPS_4, SEROLOGI_GOLONGAN_DARAH_I, SEROLOGI_GOLONGAN_DARAH_III,
                  SEROLOGI_GOLONGAN_DARAH_II, SEROLOGI_GOLONGAN_DARAH_IV, INFEKSI_MENULAR_LEWAT_TRANSFUSI_DARAH_I, INFEKSI_MENULAR_LEWAT_TRANSFUSI_DARAH_III,
                   INFEKSI_MENULAR_LEWAT_TRANSFUSI_DARAH_II, INFEKSI_MENULAR_LEWAT_TRANSFUSI_DARAH_IV, PENYADAPAN_DARAH, KOMPONEN_DARAH]]

    # Dapatkan prediksi dan akurasi menggunakan NBC
    result = predict_nbc(data_test)
    accuracy = get_accuracy_nbc()

    # Format akurasi dengan dua digit di belakang koma
    formatted_accuracy = f"{accuracy:.2f}"

    # Render hasil prediksi dan akurasi ke hasil.html
    return render_template('hasil.html', hasil=result, akurasi=formatted_accuracy, matrix_route=url_for('confusion_matrix_nbc'))

# Route untuk menampilkan confusion matrix NBC
@app.route('/confusion_matrix_nbc')
def confusion_matrix_nbc():
    # Mendapatkan true labels dan predicted labels dari model NBC
    y_true = get_y_true_nbc()
    y_pred = get_y_pred_nbc()

    # Menghitung confusion matrix menggunakan sklearn
    cm = confusion_matrix(y_true, y_pred)
    cm = ensure_confusion_matrix_size(cm, num_classes=4)

    # Render template confusion_matrix.html dengan confusion matrix yang telah dihitung
    return render_template('confusion_matrix.html', confusion_matrix=cm)

if __name__ == '__main__':
    app.run(debug=True)
