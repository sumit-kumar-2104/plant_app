import os
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)

# Load the YOLOv8 model (you can replace this with a custom-trained model path if needed)
model = YOLO(
    "C:\\Users\\ASUS\\Downloads\\Research New\\Codes and Data\\test\\runs\\detect\\train30\\weights\\best.pt")  # Use your model path here


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Check if the file is in the request
    if 'file' not in request.files:
        return render_template('index.html', prediction_text="No file provided.")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction_text="No image selected for uploading.")

    try:
        # Open the image and make predictions
        img = Image.open(file)  # Open the uploaded image file
        results = model.predict(source=img)  # Pass the image to the YOLOv8 model

        # Extract predictions
        predictions = []
        for result in results:
            for box in result.boxes:
                predictions.append({
                    'class': result.names[box.cls[0].item()],  # Predicted class label
                    'confidence': round(box.conf[0].item(), 2),  # Confidence score
                    'box': [round(coord, 2) for coord in box.xyxy[0].tolist()]  # Bounding box coordinates
                })

        if predictions:
            prediction_text = "Predictions:\n"
            for pred in predictions:
                prediction_text += f"Class: {pred['class']}, Confidence: {pred['confidence']}, Box: {pred['box']}\n"
        else:
            prediction_text = "No objects detected."

        return render_template('index.html', prediction_text=prediction_text)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error processing image: {e}")


if __name__ == "__main__":
    # Make the app accessible from your local network
    app.run(host='192.168.1.37', port=5000, debug=True)
