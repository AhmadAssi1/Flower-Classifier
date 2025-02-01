# predict.py
import argparse
import numpy as np
import tensorflow as tf
from utils import load_model, process_image, load_category_names

def predict(image_path, model, top_k=5):
    # Process the image
    processed_image = process_image(image_path)
    
    # Make predictions
    predictions = model.predict(processed_image)
    
    # Get the top k predictions
    top_k_indices = predictions[0].argsort()[-top_k:][::-1]
    top_k_probs = predictions[0][top_k_indices]
    top_k_classes = top_k_indices.tolist()
    
    return top_k_probs, top_k_classes

def print_predictions(probs, classes, category_names=None):
    # Print out the top predictions with labels
    for i in range(len(probs)):
        class_label = str(classes[i])
        class_name = category_names.get(class_label, 'Unknown') if category_names else class_label
        print(f"Class {i+1}: {class_name} with probability {probs[i]:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Predict flower class from an image using a trained model.')
    
    # Required arguments
    parser.add_argument('image_path', type=str, help='Path to the image to be predicted.')
    parser.add_argument('model_path', type=str, help='Path to the trained model (.h5 file).')
    
    # Optional arguments
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes (default=5).')
    parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping labels to flower names.')
    
    args = parser.parse_args()
    
    # Load the model
    model = load_model(args.model_path)
  



    # Load category names if provided
    category_names = None
    if args.category_names:
        category_names = load_category_names(args.category_names)
    
    # Get predictions
    probs, classes = predict(args.image_path, model, args.top_k)
    
    # Print predictions
    print_predictions(probs, classes, category_names)

if __name__ == '__main__':
    main()
