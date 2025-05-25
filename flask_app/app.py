from flask import Flask, request, render_template
import torch
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys

# Add project root to sys.path to access models directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.cnn_pytorch import get_pytorch_model
from models.cnn_tensorflow import get_tensorflow_model

app = Flask(__name__)

# Define paths to models
MODEL_DIR = os.path.join(project_root, 'models')
PYTORCH_MODEL_PATH = os.path.join(MODEL_DIR, 'thierno_model.torch')
TENSORFLOW_MODEL_PATH = os.path.join(MODEL_DIR, 'thierno_model.tensorflow.keras')

# Load models with error handling
def load_pytorch_model():
    try:
        if os.path.exists(PYTORCH_MODEL_PATH):
            model = get_pytorch_model()
            model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=torch.device('cpu'), weights_only=True))
            model.eval()
            return model
        else:
            print(f"PyTorch model not found at: {PYTORCH_MODEL_PATH}")
            return None
    except Exception as e:
        print(f"Error loading PyTorch model: {e}")
        return None

def load_tensorflow_model():
    try:
        if os.path.exists(TENSORFLOW_MODEL_PATH):
            model = tf.keras.models.load_model(TENSORFLOW_MODEL_PATH)
            return model
        else:
            print(f"TensorFlow model not found at: {TENSORFLOW_MODEL_PATH}")
            return None
    except Exception as e:
        print(f"Error loading TensorFlow model: {e}")
        return None

# Preprocess image for both models
def preprocess_image(image):
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Class labels
CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Initialize models
pytorch_model = load_pytorch_model()
tensorflow_model = load_tensorflow_model()

# Check if at least one model is loaded
if pytorch_model is None and tensorflow_model is None:
    print("Error: No models could be loaded. Exiting.")
    exit(1)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    model_options = []
    if pytorch_model is not None:
        model_options.append(('pytorch', 'Modèle PyTorch'))
    if tensorflow_model is not None:
        model_options.append(('tensorflow', 'Modèle TensorFlow'))

    if request.method == 'POST':
        model_choice = request.form.get('model')
        file = request.files.get('image')
        
        if not model_options:
            prediction = "Error: No models available."
        elif file and model_choice in [opt[0] for opt in model_options]:
            try:
                image = Image.open(file).convert('RGB')
                image_processed = preprocess_image(image)
                
                if model_choice == 'pytorch' and pytorch_model is not None:
                    with torch.no_grad():
                        output = pytorch_model(image_processed)
                        pred_idx = torch.argmax(output, dim=1).item()
                elif model_choice == 'tensorflow' and tensorflow_model is not None:
                    image_array = image_processed.numpy()
                    image_array = np.transpose(image_array, (0, 2, 3, 1))
                    output = tensorflow_model.predict(image_array)
                    pred_idx = np.argmax(output, axis=1)[0]
                else:
                    prediction = "Error: Selected model not available."
                    return render_template('index.html', prediction=prediction, model_options=model_options)
                
                prediction = CLASS_LABELS[pred_idx]
            except Exception as e:
                prediction = f"Error processing image: {str(e)}"
        else:
            prediction = "Error: Invalid model selection or no image uploaded."
    
    return render_template('index.html', prediction=prediction, model_options=model_options)

if __name__ == '__main__':
    app.run(debug=True)