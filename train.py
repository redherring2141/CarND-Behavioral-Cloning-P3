# Import necessary libraries
import numpy as np
import pandas as pd
import os
import h5py

from genAugData import*
from model import *


def load_csv(path_csv, names_col, noheader=False):
    csv = pd.read_csv(path_csv, header=None, names=names_col)
    if noheader:
        csv = csv[1:]
        
    return csv


# Returns the steering angle for images reference by the dataframe
def getSteeringAng(data, st_col, st_cal, filtering=None):
    n_cols = len(st_cal)
    print("Calibrations={0}, Rows={1}".format(n_cols, data.shape[0]))
    angs = np.zeros(data.shpae[0]*n_cols, dtype=np.float32)

    i = 0
    for idx, row in data.iterrows():
        st_ang = row[st_col]
        for (j,calib) in enumerate(st_cal):
            angs[i*n_cols + j] = st_ang+calib
        i = i + 1
    return np.clip(angs, -1, 1)


# Define column header names
names_colhead = ["Center", "Left", "Right", "Steering Angle", "Throttle", "Brake", "Speed"]

# Load csv files which includes datasets
data1_csv = load_csv(path_dataset1 + "/driving_log.csv", names_colhead)
data1_csv["Steering Angle"] = data1_csv["Steering Angle"].astype(float) 
print("Dataset1 has {0} rows".format(len(data1_csv)))

data2_csv = load_csv(path_dataset2 + "/driving_log.csv", names_colhead)
data2_csv["Steering Angle"] = data2_csv["Steering Angle"].astype(float) 
print("Dataset2 has {0} rows".format(len(data2_csv)))

data3_csv = load_csv(path_dataset3 + "/driving_log.csv", names_colhead)
data3_csv["Steering Angle"] = data3_csv["Steering Angle"].astype(float) 
print("Dataset3 has {0} rows".format(len(data3_csv)))

udacity_csv = load_csv(path_udacity + "/driving_log.csv", names_colhead, noheader=True)
udacity_csv["Steering Angle"] = udacity_csv["Steering Angle"].astype(float) 
print("Udacity Dataset has {0} rows".format(len(udacity_csv)))

# Define the columns of interest and steering angle calibration
names_stang = ["Center", "Left", "Right"]
calib_stang = [0, 0.25, -0.25]

# Make training set as an ensemble of datasets by concatenating them
ensemble = [data1_csv, udacity_csv, data2_csv]
train_csv = pd.concat(ensemble)

# Make validation set
valid_csv = data3_csv

# Load the model and make training and validation sets
batch_divider = 160
batch_size = len(train_csv) * 3 // batch_divider

dnn_model = nvidiaPilotNet()
train_set = genImgData(train_csv, (160,320,3), names_stang, "Steering Angle", calib_stang, batch_size=batch_size)
valid_set = genImgData(valid_csv, (160,320,3), names_stang, "Steering Angle", calib_stang, batch_size=(batch_size*batch_divider)//5, data_aug_pct=0.0)
x_val, y_val = next(valid_set)

dnn_model.summary()

# Train the model
dnn_model.fit_generator(train_set, validation_data=(x_val,y_val), samples_per_epoch=batch_size*batch_divider, nb_epoch=2, verbose=1)
model_fin = dnn_model.layers[-2]
model_fin.save("./models/model.h5")
print("Model training and saving completed")
