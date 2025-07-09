from flask import Flask, render_template, request, redirect
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model('pattern_sense_model.h5')

# Class names 
class_names = ['animal', 'cartoon', 'floral', 'geometry','ikat','plain','polka dot','squares','stripes','tribal']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join('static/uploads', file.filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            prediction = model.predict(img_array)
            predicted_class = class_names[np.argmax(prediction)]

            confidence = round(100 * np.max(prediction), 2)

            return render_template('predict.html', 
                                   prediction=predicted_class, 
                                   confidence=confidence,
                                   img_path=filepath)

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
