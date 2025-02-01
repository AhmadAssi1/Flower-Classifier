# utils.py
import numpy as np
import tensorflow as tf
from PIL import Image
import tensorflow_hub as hub

# Load the Keras model
def load_model(model_path):
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    return model

# Preprocess image before feeding to the model
def process_image(image_path):
    # Open image using PIL
    image = Image.open(image_path)
    
    # Resize image to 224x224 as required by the model
    image = image.resize((224, 224))
    
    # Convert image to numpy array
    image_array = np.asarray(image)
    
    # Normalize pixel values to between 0 and 1
    image_array = image_array / 255.0
    
    # Add batch dimension (needed for prediction)
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

# Convert the label map JSON to a dictionary
def load_category_names(category_names_path):
    import json
    with open(category_names_path, 'r') as f:
        category_names = json.load(f)
    return category_names
