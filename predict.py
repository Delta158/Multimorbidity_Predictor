import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime, timedelta
import time

import tensorflow as tf
tf.config.list_physical_devices('GPU')
# tf.get_logger().setLevel(3)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



from preprocess import do_all
from models import compile_model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def get_target(data, disease):
    train_admissions = data.shape[1] -1
    # print(train_admissions)
    a = data[:, :train_admissions, :] # first (num_admissions -1)
    b = []
    # print(data.shape[0])
    for i in range(data.shape[0]):  # number of patients
        # a.append(data[i][:num])  # first (num_admissions -1)
        b.append(data[i][-1][disease])   # final admission
        # print(disease)

    return np.array(a), np.array(b)


# print(test_data.shape)
# print(X_test.shape)
import sys

def predict_diseases(models, data):

    if len(data.shape) == 2:
        data = np.reshape(data, (1, data.shape[0], data.shape[1]))
        # print("Reshaping data into compatible format.", data.shape)
    
    all_predictions = [[] for _ in range(data.shape[0])]   
    # print(np.array(all_predictions).shape) 


    iterations = data.shape[-1]

    for i in range(iterations):
        # Use the first n-1 admissions to make prediction
        X_test , _ = get_target(data, i)
        X_test = X_test.astype('float32')
        # print(X_test.shape)

        model = models[i]

        print(f"predicting {diseases[i]} ...")
        predictions = model.predict(X_test, verbose=1)
        rounded_predictions = np.round(predictions).astype(int)
        # print(rounded_predictions.shape)

        all_predictions = np.concatenate((all_predictions, rounded_predictions), axis=1).astype(int)

    return all_predictions


# Check the number of command-line arguments
if len(sys.argv) < 3:
    print("Usage: python predict.py category_mapping models_dir target")
    sys.exit(1)

# Access command-line arguments

cat_map = sys.argv[1]
mod_dir = sys.argv[2]
patient = sys.argv[3]

diseases = [1, 140, 240, 280, 290, 320, 390, 460, 520, 580, 630, 680, 710, 740, 760, 780, 800, 'V', 'E']

print(f"loading models from '{mod_dir}' directory")

models = []
for i in range(len(diseases)):
    print(f"loading {diseases[i]} model")
    m = tf.keras.models.load_model(f'models/Disease_{diseases[i]}_model.keras')
    models.append(m)

print("finished loading models")


test_data = np.loadtxt("test_patient").astype(int)
print()
print("starting predictions")
print()

predictions = predict_diseases(models, test_data)

# print(predictions)

pred_diseases = []

for i in range(len(diseases)):
    if predictions[0][i] == 1:
        pred_diseases.append(diseases[i])

print()
print("predicted diseases")
for i in pred_diseases:
    print(i, end=' ')

with open('test_patient', 'a+') as file:
    # Append text to the end of the file
    file.write(" \n")
    file.write(" \n")
    file.write(" ".join(str(i) for i in pred_diseases))

