from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import joblib
from utils.glcm_features import extract_glcm_features
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = joblib.load('model/knn_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    features = None
    original_path = None
    green_path = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Baca dan resize citra
            img = cv2.imread(filepath)
            img_resized = cv2.resize(img, (800, 400))
            cv2.imwrite(filepath, img_resized)  # Simpan ulang versi resize

            # Ekstrak green channel
            green_channel = img_resized[:, :, 1]
            green_filename = 'green_' + filename
            green_path = os.path.join(UPLOAD_FOLDER, green_filename)
            cv2.imwrite(green_path, green_channel)

            # Ekstraksi fitur
            feature_values = extract_glcm_features(img_resized)
            features = [round(f, 4) for f in feature_values]
            prediction = model.predict(np.array(feature_values).reshape(1, -1))[0]
            result = "Asli" if prediction == 0 else "Palsu"

            original_path = filepath

    return render_template('index.html',
                           result=result,
                           features=features,
                           original=original_path,
                           green=green_path)

if __name__ == '__main__':
    app.run(debug=True)
