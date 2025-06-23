import os
import cv2
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils.glcm_features import extract_glcm_features

# Lokasi folder dataset
DATASET_PATH = 'dataset_augmented'
LABELS = {'asli': 0, 'palsu': 1}

# Menyimpan fitur dan label
X = []
y = []

# Loop setiap folder (asli dan palsu)
for label_name in LABELS:
    label_folder = os.path.join(DATASET_PATH, label_name)
    label_value = LABELS[label_name]

    for filename in os.listdir(label_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(label_folder, filename)
            img = cv2.imread(filepath)
            if img is not None:
                features = extract_glcm_features(img)
                X.append(features)
                y.append(label_value)

# Konversi ke array
X = np.array(X)
y = np.array(y)

# 💡 Bagi data menjadi training dan testing (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 🧠 Buat dan latih model KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 🔍 Uji model
y_pred = knn.predict(X_test)

# 📊 Evaluasi hasil
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['Asli', 'Palsu']))

print(f"\nAkurasi: {accuracy_score(y_test, y_pred):.2f}")

# 💾 Simpan model ke file
os.makedirs('model', exist_ok=True)
with open('model/knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)

print("\n✅ Model KNN berhasil dilatih, diuji, dan disimpan di 'model/knn_model.pkl'")
