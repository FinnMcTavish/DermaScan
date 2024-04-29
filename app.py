# Import necessary libraries
from flask import Flask, render_template, request, jsonify
import os
from PIL import Image
import numpy as np
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Load your trained machine learning model
model = tf.keras.models.load_model('my_model.h5')

# Define function to preprocess the input image
def preprocess_image(image):
    # Resize image to match model input shape
    img = image.resize((256, 256))
    # Convert image to numpy array
    img_array = np.array(img) / 255.0  # Normalize pixel values
    # Expand dimensions to match model input shape
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# Define function to make predictions
def predict(image):
    preprocessed_img = preprocess_image(image)

    # Perform prediction using your loaded model
    predictions = model.predict(preprocessed_img)
    # Get the predicted class
    threshold = 0.5

    if predictions[0][0] < threshold:
        predicted_class = "Scar"
    else:
        predicted_class = "Vitiligo" 
    return predicted_class

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result',methods=['GET', 'POST'])
def new_page():
    if request.method == 'POST':
        file = request.files['imageUpload']
        # Open the image file
        image = Image.open(file)
        # Perform prediction
        prediction = predict(image)
        # Return prediction as JSON response
        return render_template('Prediction.html', prediction=prediction)
    return render_template('Prediction.html')

if __name__ == '__main__':
    app.run(debug=True)
