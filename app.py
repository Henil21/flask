from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer



import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import pathlib


# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
# MODEL_PATH = 'models/model_resnet.h5'

# Load your trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
# print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


loading=tf.keras.models.load_model("Tumor_model.h5")
# loading.summary()


data_dir=pathlib.Path("Tumor_MRI/brain_tumor_dataset/")

class_name=class_name=np.array(sorted([item.name for item in data_dir.glob("*")]))
class_name
# class_name=["No","Yes"]

def load_prep_img(filename,img_shape=224):

  # read in the image
  img=tf.io.read_file(filename)
  # Decode the read file into a tensor
  img=tf.image.decode_image(img)
  # resize the image
  img=tf.image.resize(img,size=[img_shape,img_shape])
  # Rescale the image (get all values between 0 and 1)
  img=img/225
  return img


def pred_and_plot(model,filename,class_name=class_name):
    """
    Imports an image locate at filename ,make a prediction with model
    and plot the images with the predicted class as title
    """
  #  import the target image and preprocess it
    img=load_prep_img(filename)
    pred=model.predict(tf.expand_dims(img,axis=0))
    pred_class=class_name[int(tf.round(pred))]
  #  plot the image and predicted class
    # plt.imshow(img)
    img = cv.imread("test.jpg")
#cv.rectangle(img,(29,2496),(604,2992),(255,0,0),5)
    plt.imshow(img)
    plt.title(pred_class)
    plt.axis(False);
    plt.show()

# x=pred_and_plot(loading,"normal.jpg")




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

    #     # Make prediction
    #     preds = model_predict(file_path, loading)

    #     # Process your result for human
    #     # pred_class = preds.argmax(axis=-1)            # Simple argmax
    #     pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
    #     result = str(pred_class[0][0][1])               # Convert to string
    #     return result
    # return None



        img=load_prep_img(file_path)
        pred=loading.predict(tf.expand_dims(img,axis=0))
        pred_class=class_name[int(tf.round(pred))]
  #  plot the image and predicted class
    # plt.imshow(img)
        img = cv.imread(file_path)
#cv.rectangle(img,(29,2496),(604,2992),(255,0,0),5)
        plt.imshow(img)
        plt.title(pred_class)
        plt.axis(False);
        plt.show()
        pred_class = str(pred_class[0][0][1]) 
        return pred_class   
    return pred_class









if __name__ == '__main__':
    app.run(debug=True)

