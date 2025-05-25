import torch
import tensorflow as tf
import os
from PIL import Image
from torchvision import transforms
import sys

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.cnn_pytorch import get_pytorch_model
from models.cnn_tensorflow import get_tensorflow_model

class Predictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pytorch_model = None
        self.tensorflow_model = None
        self.class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

    def load_models(self):
        model_dir = os.path.join(project_root, 'models')
        self._load_pytorch_model(model_dir)
        self._load_tensorflow_model(model_dir)

    def _load_pytorch_model(self, model_dir):
        model_path = os.path.join(model_dir, 'thierno_model.torch')
        if os.path.exists(model_path):
            model = get_pytorch_model()
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            self.pytorch_model = model.to(self.device)
        else:
            raise FileNotFoundError(f"PyTorch model not found at: {model_path}")

    def _load_tensorflow_model(self, model_dir):
        model_path = os.path.join(model_dir, 'thierno_model.tensorflow.keras')
        if os.path.exists(model_path):
            self.tensorflow_model = tf.keras.models.load_model(model_path)
        else:
            raise FileNotFoundError(f"TensorFlow model not found at: {model_path}")

    def preprocess_image(self, image):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
        image = transform(image).unsqueeze(0)
        return image

    def predict(self, filepath, framework="pytorch"):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Image not found at: {filepath}")
        
        image = Image.open(filepath).convert('RGB')
        image_processed = self.preprocess_image(image)
        
        if framework == "pytorch" and self.pytorch_model is not None:
            image_processed = image_processed.to(self.device)
            with torch.no_grad():
                output = self.pytorch_model(image_processed)
                pred_idx = torch.argmax(output, dim=1).item()
        elif framework == "tensorflow" and self.tensorflow_model is not None:
            image_array = image_processed.numpy()
            image_array = np.transpose(image_array, (0, 2, 3, 1))
            output = self.tensorflow_model.predict(image_array)
            pred_idx = np.argmax(output, axis=1)[0]
        else:
            raise ValueError("Framework not supported or model not loaded")
        
        return self.class_labels[pred_idx]