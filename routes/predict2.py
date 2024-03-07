import os

import cv2
from flask import Blueprint, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

predict_bp2 = Blueprint("predict2", __name__)

# Direktori untuk menyimpan gambar yang diterima
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@predict_bp2.route("/predict2", methods=["POST"])
def predict():
    if "file" in request.files:
        try:
            # Simpan gambar yang diterima
            photo = request.files["file"]
            filename = os.path.join(UPLOAD_FOLDER, photo.filename)
            photo.save(filename)

            result = predict_class(filename, 'https://storage.googleapis.com/default0987/best_model.h5')  # Sesuaikan path model dengan lokasi dan nama sebenarnya

            return jsonify({"result": result})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "No photo provided"}), 400


# Function to load and preprocess an image
def load_and_preprocess_image(image_path):
    # Load and resize the image
    img = image.load_img(image_path, target_size=(224, 224))
    # Convert the image to a NumPy array
    img_array = image.img_to_array(img)
    # Expand the dimensions to create a batch of size 1
    img_array = tf.expand_dims(img_array, axis=0)
    # Preprocess the input image for ResNet50
    img_array = preprocess_input(img_array)
    return img_array


def predict_class(image_path, model_path):
    try:
        # Load the model
        model = tf.keras.models.load_model(model_path)

        input_image = load_and_preprocess_image(image_path)

        # Prediction using the loaded model
        prediction = model.predict(input_image)

        # Ambil indeks kelas dengan nilai prediksi tertinggi
        kelas_terprediksi = prediction.argmax()

        # Kode pengkondisian untuk menentukan kelas
        if kelas_terprediksi == 0:
            return "Kelas: Anorganik"
        elif kelas_terprediksi == 1:
            return "Kelas: B3"
        elif kelas_terprediksi == 2:
            return "Kelas: Organik"
        else:
            return "Kelas tidak dikenali"

    except Exception as e:
        return str(e)
