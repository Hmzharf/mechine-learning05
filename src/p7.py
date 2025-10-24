import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras import layers, models
from keras import regularizers
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Langkah 1 — Siapkan Data
df = pd.read_csv("processed_kelulusan.csv")
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# Cek distribusi kelas dalam target 'y'
print("Distribusi Kelas: ")
print(y.value_counts())

# Menghapus kelas dengan hanya 1 sampel (jika ada)
kelas_dengan_1_sampel = y.value_counts()[y.value_counts() == 1].index.tolist()
if kelas_dengan_1_sampel:
    print(f"Menghapus kelas dengan hanya 1 sampel: {kelas_dengan_1_sampel}")
    df = df[~df["Lulus"].isin(kelas_dengan_1_sampel)]

# Cek ulang distribusi setelah penghapusan kelas dengan 1 sampel
print("Distribusi Kelas Setelah Penghapusan:")
print(df["Lulus"].value_counts())

# Standarisasi data
sc = StandardScaler()
Xs = sc.fit_transform(df.drop("Lulus", axis=1))
y = df["Lulus"]

# Pembagian data menjadi training, validation, dan test tanpa stratifikasi
X_train, X_temp, y_train, y_temp = train_test_split(Xs, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Training Set Shape:", X_train.shape)
print("Validation Set Shape:", X_val.shape)
print("Test Set Shape:", X_test.shape)

# Langkah 2 — Bangun Model ANN
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")  # Klasifikasi biner
])

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy", "AUC"])
model.summary()

# Langkah 3 — Training dengan Early Stopping
es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100, batch_size=32,
    callbacks=[es], verbose=1
)

# Langkah 4 — Evaluasi di Test Set
loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy:", acc, "AUC:", auc)

y_proba = model.predict(X_test).ravel()
y_pred = (y_proba >= 0.5).astype(int)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=3))

# Langkah 5 — Visualisasi Learning Curve
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Learning Curve")
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=120)

# Langkah 6 — Eksperimen

# Fungsi untuk membangun model dengan variasi arsitektur
def build_model(neurons, optimizer, regularizer=None):
    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(neurons, activation="relu", kernel_regularizer=regularizer),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy", "AUC"])
    return model

# Eksperimen 1: Coba model dengan 64 neuron
model_64 = build_model(64, keras.optimizers.Adam(1e-3))

# Eksperimen 2: Model dengan SGD optimizer
model_sgd = build_model(32, keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9))

# Eksperimen 3: Model dengan L2 Regularization
model_l2 = build_model(32, keras.optimizers.Adam(1e-3), regularizers.l2(0.01))

# Daftar model untuk evaluasi
models = [model_64, model_sgd, model_l2]
names = ["Model_64", "Model_SGD", "Model_L2"]

# Evaluasi model yang telah dikembangkan
for model_variant, name in zip(models, names):
    print(f"\nEvaluating {name}:")
    history_variant = model_variant.fit(X_train, y_train,
                                        validation_data=(X_val, y_val),
                                        epochs=100, batch_size=32,
                                        callbacks=[es], verbose=0)

    loss, acc, auc = model_variant.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy ({name}):", acc, "AUC:", auc)

    y_proba_variant = model_variant.predict(X_test).ravel()
    y_pred_variant = (y_proba_variant >= 0.5).astype(int)

    print(confusion_matrix(y_test, y_pred_variant))
    print(classification_report(y_test, y_pred_variant, digits=3))

    # Visualisasi learning curve untuk setiap model
    plt.plot(history_variant.history["loss"], label=f"Train Loss ({name})")
    plt.plot(history_variant.history["val_loss"], label=f"Val Loss ({name})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Learning Curve for {name}")
    plt.tight_layout()
    plt.savefig(f"learning_curve_{name}.png", dpi=120)
