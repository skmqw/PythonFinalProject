from __future__ import division, print_function
import os
import numpy as np

# Keras
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/my_model.h5'

# Loading the trained model
model = load_model(MODEL_PATH)
# Necessary to make everything ready to run on the GPU ahead of time
model.make_predict_function()
print('loaded the model successfully. Starting the web applications')


def model_predict(img_path, model):
    # adjust the size as the size should match the training data size
    img = image.load_img(img_path, target_size=(50, 50))

    # Preprocessing the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # predict the model
    preds = model.predict(img)
    pred = np.argmax(preds, axis=1)
    return pred


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        pred = model_predict(file_path, model)
        os.remove(file_path)  # removes file from the server after prediction has been returned

        # Arrange the correct return according to the model.

        str1 = 'Malaria Detected'
        str2 = 'NO Malaria Detected'
        if pred[0] == 0:
            return str2
        else:
            return str1
    return None


# run app locally
if __name__ == '__main__':
    app.run(debug=True)
