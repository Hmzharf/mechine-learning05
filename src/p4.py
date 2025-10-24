import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import os

# Cek apakah file dataset.csv ada
if not os.path.exists(r"D:\Hamzah\mechine-learning\.venv\Scripts\src\dataset.csv"):
    print("File 'dataset.csv' tidak ditemukan. Pastikan file tersebut berada di direktori yang benar.")
else:
    try:
        # Mencoba membaca dataset dengan opsi untuk menangani file yang tidak lengkap
        df = pd.read_csv("dataset.csv", skip_blank_lines=True)
        
        # Cek apakah dataset kosong
        if df.empty:
            print("Dataset kosong! Harap periksa file CSV Anda.")
        else:
            # Informasi dasar tentang dataset
            print(df.info())
            print(df.head())

            # Cek jumlah missing values
            print("\nJumlah Missing Values:")
            print(df.isnull().sum())

            # Menghapus duplikat
            df = df.drop_duplicates()

            # Visualisasi boxplot IPK
            sns.boxplot(x=df['IPK'])

            # Statistik deskriptif
            print("\nStatistik Deskriptif:")
            print(df.describe())

            # Visualisasi histogram dan scatter plot
            sns.histplot(df['IPK'], bins=10, kde=True)
            sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus')
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm")

            # Membuat fitur baru
            df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14
            df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']

            # Menyimpan dataset yang sudah diproses
            df.to_csv("processed_kelulusan.csv", index=False)
            print("\nData yang telah diproses telah disimpan ke 'processed_kelulusan.csv'.")
            print(df.head())

            # Menentukan fitur dan target
            X = df.drop(columns=['Lulus'])
            y = df['Lulus']

            # Memeriksa distribusi kelas
            print("\nDistribusi kelas 'Lulus':")
            print(y.value_counts())

            # Mengatasi kelas yang hanya memiliki 1 sampel
            if y.value_counts().min() <= 1:
                print("\nKelas dengan hanya satu sampel ditemukan. Menghapus data yang tidak seimbang...")
                # Menghapus kelas yang memiliki hanya satu sampel
                df = df[df['Lulus'].map(df['Lulus'].value_counts()) > 1]
                X = df.drop(columns=['Lulus'])
                y = df['Lulus']

            # Membagi dataset
            try:
                # Menggunakan stratified splitting jika distribusi kelas sudah baik
                X_train, X_temp, y_train, y_temp = train_test_split(
                    X, y, test_size=0.3, stratify=y, random_state=42)
                
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
            except ValueError:
                # Jika masih ada kelas dengan satu sampel setelah filter, menggunakan split acak
                print("\nMenggunakan split acak karena masih ada kelas dengan satu sampel.")
                X_train, X_temp, y_train, y_temp = train_test_split(
                    X, y, test_size=0.3, random_state=42)
                
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=0.5, random_state=42)

            # Menampilkan shape dan distribusi kelas setelah pembagian
            print("\nShapes (train/val/test):", X_train.shape, X_val.shape, X_test.shape)
            print("\nClass counts:")
            print("train:\n", y_train.value_counts())
            print("val:\n",   y_val.value_counts())
            print("test:\n",  y_test.value_counts())

    except pd.errors.EmptyDataError:
        print("File CSV kosong atau tidak dapat dibaca. Harap periksa file CSV Anda.")
    except FileNotFoundError:
        print("File 'dataset.csv' tidak ditemukan.")
    except Exception as e:
        print(f"Terjadi kesalahan saat membaca file: {e}")
