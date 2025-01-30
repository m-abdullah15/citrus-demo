from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Load the trained model
MODEL_PATH = 'citrus_disease_model_densenet.keras'  # Update with your actual model path
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels
CLASS_LABELS = ['Citrus canker', 'Citrus greening', 'Citrus mealybugs', 
                'Die back', 'Foliage damaged', 'Healthy leaf', 
                'Powdery mildew', 'Shot hole', 'Spiny whitefly', 'Yellow leaves']  # Update with actual labels

def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match model input
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image  # Add batch dimension

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))
    processed_image = preprocess_image(image)
    
    prediction = model.predict(processed_image)
    predicted_class = CLASS_LABELS[np.argmax(prediction)]

    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
