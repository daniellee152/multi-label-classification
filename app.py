import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import pandas as pd
import os

# Keras
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow_hub as hub

#Flask
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
from werkzeug.utils import secure_filename
from flask import Flask, render_template

IMG_SIZE = 224
CHANNELS = 3
OUTPUT = ['dancing','eating','glasses','jumping','laying','man','running','sitting','smiling','standing', 'walking','woman']

#Load saved model
MODEL_PATH = 'multi_label_v2_fine_tune.h5'
model = load_model(MODEL_PATH, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)
def encode(filepath):
    img = image.load_img(filepath, target_size=(IMG_SIZE,IMG_SIZE,CHANNELS))
    img = image.img_to_array(img)
    img = img/255
    img = np.expand_dims(img, axis=0)
    return img
    
def predict(image):
    prediction = (model.predict(image) > 0.5).astype('int')
    prediction = pd.Series(prediction[0])
    prediction.index = OUTPUT
    prediction = prediction[prediction==1].index.values
    return prediction


app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
bootstrap = Bootstrap(app)

class UploadForm(FlaskForm):
    upload = FileField('Select an image:', validators=[
        FileRequired(),
        FileAllowed(['jpg', 'png', 'jpeg', 'JPEG', 'PNG', 'JPG'], 'Images only!')
    ])
    submit = SubmitField('Classify')

@app.route('/', methods=['GET', 'POST'])
def home():
    form = UploadForm()
    if form.validate_on_submit():
        f = form.upload.data
        filename = secure_filename(f.filename)
        file_url = os.path.join('static', filename)
        f.save(file_url)
        form = None
        image = encode(file_url)
        prediction = predict(image)
    else:
        file_url = None
        prediction = None
    return render_template("index.html", form=form, file_url=file_url, prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)


