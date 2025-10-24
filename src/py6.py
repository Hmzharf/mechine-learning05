import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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
X_train, X_temp, y_train, y_temp = train_test_split(X_clean, y_clean, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

print("\nShapes (train/val/test):", X_train.shape, X_val.shape, X_test.shape)
print("\nClass counts:")
print("train:\n", y_train.value_counts())
print("val:\n", y_val.value_counts())
print("test:\n", y_test.value_counts())

# ======================================
# Membangun Model Baseline (Random Forest)
# ======================================
num_cols = X_train.select_dtypes(include="number").columns

# Preprocessing step: handle missing values and scale numerical features
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
], remainder="drop")

# Random Forest Classifier with balanced class weights
rf = RandomForestClassifier(
    n_estimators=300, max_features="sqrt", class_weight="balanced", random_state=42
)

# Full pipeline: preprocessing and model
pipe = Pipeline([("pre", pre), ("clf", rf)])

# Melatih model dengan data training
pipe.fit(X_train, y_train)

# Prediksi menggunakan validation set
y_val_pred = pipe.predict(X_val)
print("Baseline RF — F1(val):", f1_score(y_val, y_val_pred, average="macro"))
print(classification_report(y_val, y_val_pred, digits=3))

# ======================================
# Validasi Silang (Cross-validation)
# ======================================
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

# Gunakan StratifiedShuffleSplit yang lebih fleksibel
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

# Cross-validation dengan StratifiedShuffleSplit
scores = cross_val_score(pipe, X_train, y_train, cv=sss, scoring="f1_macro", n_jobs=-1)

print("CV F1-macro (train):", scores.mean(), "±", scores.std())

# ======================================
# Tuning Ringkas dengan GridSearch
# ======================================
from sklearn.model_selection import GridSearchCV

# Parameter untuk grid search
param = {
  "clf__max_depth": [None, 12, 20, 30],
  "clf__min_samples_split": [2, 5, 10]
}

# Tuning model menggunakan GridSearchCV
gs = GridSearchCV(pipe, param_grid=param, cv=sss, scoring="f1_macro", n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)

print("Best params:", gs.best_params_)
best_model = gs.best_estimator_

# Evaluasi model terbaik di validation set
y_val_best = best_model.predict(X_val)
print("Best RF — F1(val):", f1_score(y_val, y_val_best, average="macro"))

# ======================================
# Evaluasi Akhir (Test Set)
# ======================================
final_model = best_model  # Atau gunakan pipe_lr jika baseline lebih baik
y_test_pred = final_model.predict(X_test)

# F1 score untuk test set
f1_test = f1_score(y_test, y_test_pred, average="macro")
print("\nF1(test):", f1_test)
print(classification_report(y_test, y_test_pred, digits=3))
print("Confusion Matrix (test):")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

# Visualisasi Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Tidak Lulus", "Lulus"], yticklabels=["Tidak Lulus", "Lulus"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120)  # Menyimpan confusion matrix

# ROC-AUC (jika ada predict_proba)
if hasattr(final_model, "predict_proba"):
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    try:
        print("ROC-AUC(test):", roc_auc_score(y_test, y_test_proba))
    except:
        pass
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc_score(y_test, y_test_proba))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal line (random model)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_test.png", dpi=120)  # Menyimpan ROC curve

    # Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, y_test_proba)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve (test)")
    plt.tight_layout()
    plt.savefig("pr_test.png", dpi=120)  # Menyimpan Precision-Recall curve
else:
    print("\nModel tidak mendukung prediksi probabilitas (predict_proba).")

# ======================================
# Simpan Model
# ======================================
import joblib

# Simpan model terbaik
joblib.dump(final_model, "rf_model.pkl")
print("Model disimpan sebagai 'rf_model.pkl'")

# ======================================
# Cek Inference Lokal
# ======================================
mdl = joblib.load("rf_model.pkl")

# Contoh input fiktif untuk prediksi
sample = pd.DataFrame([{
  "IPK": 3.4,
  "Jumlah_Absensi": 4,
  "Waktu_Belajar_Jam": 7,
  "Rasio_Absensi": 4/14,
  "IPK_x_Study": 3.4*7
}])

# Prediksi
print("Prediksi:", int(mdl.predict(sample)[0]))
