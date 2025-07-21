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
    predictions = []
    for r in results:
        for box in r.boxes:
            predictions.append({
                'class': model.names[int(box.cls)],
                'confidence': float(box.conf),
                'bbox': box.xyxy[0].tolist()
            })

    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
