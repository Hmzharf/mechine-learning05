import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================
# Memuat Data dan Menangani Kelas Jarang
# ======================================
df = pd.read_csv("processed_kelulusan.csv")
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# Cek distribusi kelas
print("Distribusi Kelas:\n", y.value_counts())

# Menghapus kelas yang hanya memiliki satu data
y_counts = y.value_counts()
rare_classes = y_counts[y_counts <= 1].index
df_clean = df[~df["Lulus"].isin(rare_classes)]
X_clean = df_clean.drop("Lulus", axis=1)
y_clean = df_clean["Lulus"]

# Membagi data tanpa stratifikasi untuk menghindari error
X_train, X_temp, y_train, y_temp = train_test_split(X_clean, y_clean, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("\nShapes (train/val/test):", X_train.shape, X_val.shape, X_test.shape)
print("\nClass counts:")
print("train:\n", y_train.value_counts())
print("val:\n", y_val.value_counts())
print("test:\n", y_test.value_counts())

# ======================================
# Membangun Model Baseline (Logistic Regression)
# ======================================
num_cols = X_train.select_dtypes(include="number").columns

pre = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                          ("sc", StandardScaler())]), num_cols),
], remainder="drop")

logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
pipe_lr = Pipeline([("pre", pre), ("clf", logreg)])

pipe_lr.fit(X_train, y_train)
y_val_pred = pipe_lr.predict(X_val)
print("\nBaseline (LogReg) F1(val):", f1_score(y_val, y_val_pred, average="macro"))
print(classification_report(y_val, y_val_pred, digits=3))

# ======================================
# Membangun Model Alternatif (Random Forest)
# ======================================
rf = RandomForestClassifier(n_estimators=300, max_features="sqrt", class_weight="balanced", random_state=42)
pipe_rf = Pipeline([("pre", pre), ("clf", rf)])

pipe_rf.fit(X_train, y_train)
y_val_rf = pipe_rf.predict(X_val)
print("\nRandomForest F1(val):", f1_score(y_val, y_val_rf, average="macro"))

# ======================================
# Validasi Silang & Tuning Model dengan GridSearchCV
# ======================================
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

param = {
  "clf__max_depth": [None, 12, 20, 30],
  "clf__min_samples_split": [2, 5, 10]
}

gs = GridSearchCV(pipe_rf, param_grid=param, cv=sss, scoring="f1_macro", n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)
print("\nBest params:", gs.best_params_)
print("Best CV F1:", gs.best_score_)

best_rf = gs.best_estimator_
y_val_best = best_rf.predict(X_val)
print("\nBest RF F1(val):", f1_score(y_val, y_val_best, average="macro"))

# ======================================
# Evaluasi Akhir pada Test Set
# ======================================
final_model = best_rf  # atau pipe_lr jika baseline lebih baik
y_test_pred = final_model.predict(X_test)

print("\nF1(test):", f1_score(y_test, y_test_pred, average="macro"))
print(classification_report(y_test, y_test_pred, digits=3))
print("Confusion matrix (test):")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

# Plot Confusion Matrix menggunakan Seaborn untuk visualisasi
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Tidak Lulus", "Lulus"], yticklabels=["Tidak Lulus", "Lulus"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()  # Memperbaiki layout agar tidak terpotong
plt.savefig("confusion_matrix.png", dpi=120)  # Menyimpan confusion matrix sebagai file gambar
print("Confusion Matrix telah disimpan sebagai 'confusion_matrix.png'.")

# ROC-AUC (jika ada predict_proba)
if hasattr(final_model, "predict_proba"):
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    try:
        print("ROC-AUC(test):", roc_auc_score(y_test, y_test_proba))
    except:
        pass
    
    # Plot ROC curve dan simpan ke file
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc_score(y_test, y_test_proba))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal line (random model)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    
    # Menyimpan grafik ke file tanpa menampilkan
    plt.tight_layout()  # Memperbaiki layout agar tidak terpotong
    plt.savefig("roc_test.png", dpi=120)  # Menyimpan grafik sebagai file
    print("Grafik ROC telah disimpan sebagai 'roc_test.png'.")
else:
    print("\nModel tidak mendukung prediksi probabilitas (predict_proba).")
