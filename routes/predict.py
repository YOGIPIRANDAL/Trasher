# app/routes/predict_routes.py
import cv2
from PIL import Image
from flask import Blueprint, request, jsonify, url_for
from ultralytics import YOLO

predict_bp = Blueprint("predict", __name__)
model = YOLO('routes/best.pt')


@predict_bp.route("/predict", methods=["POST"])
def predict():
    if "file" in request.files:
        # Decode base64 to image
        photo = request.files["file"]
        photo = Image.open(photo.stream)

        # Perform inference
        output = model(photo)
        names = model.names
        img = output[0].orig_img
        detected_objects = {}

        for r in output:
            for c in r.boxes.cls:
                class_name = names[int(c)]
                detected_objects[class_name] = detected_objects.get(class_name, 0) + 1
            for box, conf in zip(r.boxes.data, r.boxes.conf):
                x_min, y_min = int(box[0]), int(box[1])
                x_max, y_max = int(box[2]), int(box[3])
                confidence = round(float(conf), 2)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255,0,0), 1)

                # Tambahkan nama kelas dan confidence level ke dalam kotak pembatas
                label_text = f"{class_name}: {confidence}"
                cv2.putText(img, label_text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        # Simpan gambar ke file
        cv2.imwrite("static/result.png", img)

        # Generate URL for the saved image
        result_image_url = url_for('static', filename='result.png', _external=True)

        # Sertakan informasi label dalam respons JSON
        response_data = {
            "result_image": result_image_url,
            "detected_labels": detected_objects
        }

        return jsonify(response_data)

    return jsonify({"error": "No photo_base64 provided"}), 400
