from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf
import numpy as np
import requests
from io import BytesIO

app = Flask(__name__)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="best_float16.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = (640, 640)
CLASSES = ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"]

def preprocess_image(image: Image.Image):
    image = image.resize(IMG_SIZE).convert("RGB")
    img_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_image(image: Image.Image):
    input_tensor = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    stage_index = int(np.argmax(output))
    confidence = float(output[stage_index])
    return CLASSES[stage_index], confidence

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_url = data.get("image_url")

    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        stage, confidence = predict_image(image)

        return jsonify({
            "stage": stage,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
