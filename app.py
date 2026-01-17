from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from PIL import Image
from io import BytesIO
from ultralytics import YOLO  # or any other model

app = Flask(__name__)
CORS(app) 

# Load model once
model = YOLO("best.pt")
model2 = YOLO("pestDiseaseModel.pt")

# =========================
# Helper: format label
# =========================
def format_label(label: str):
    label = label.replace("_", " ")
    label = label.title()
    return label



@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_url = data.get('image_url')

    if not image_url:
        return jsonify({'error': 'No image_url provided'}), 400

    # Download image from URL
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")

    # Run inference
    results = model(image)

    # Parse results (example for bounding boxes)
    '''predictions = []
    for r in results:
        for box in r.boxes:
            predictions.append({
                'class': model.names[int(box.cls)],
                'confidence': float(box.conf),
                'bbox': box.xyxy[0].tolist()
            })

    return jsonify({'predictions': predictions})'''

    # Get best prediction
    best_pred = None
    best_conf = 0

    for r in results:
        for box in r.boxes:
            conf = float(box.conf)
            if conf > best_conf:
                best_conf = conf
                best_pred = {
                    'class': model.names[int(box.cls)],
                    'confidence': conf,
                    'bbox': box.xyxy[0].tolist()
                }

    return jsonify({'prediction': best_pred})

@app.route('/predict_pest', methods=['POST'])
def predict_pest():
    data = request.get_json()
    image_url = data.get('image_url')

    if not image_url:
        return jsonify({'error': 'No image_url provided'}), 400

    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        return jsonify({'error': 'Failed to load image', 'details': str(e)}), 400

    results = model2(image)
    r = results[0]

    class_id = int(r.probs.top1)
    confidence = float(r.probs.top1conf)

    raw_label = model2.names[class_id]
    label = format_label(raw_label)

    return jsonify({
        "prediction": {
            "class": label,
            "confidence": round(confidence, 4)
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
