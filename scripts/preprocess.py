## develop  the preprocess
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
strProjectFolder="datasets/R3-Yahoo/"

DataTest = pd.read_csv(os.path.join(strProjectFolder, "ydata-ymusic-rating-study-v1_0-test.txt"),sep="\\t",header=None)
DataTest.columns=["UserID", "ItemID", "Rating"]

DataTrain = pd.read_csv(os.path.join(strProjectFolder, "ydata-ymusic-rating-study-v1_0-train.txt"),sep="\\t",header=None)
DataTrain.columns=["UserID", "ItemID", "Rating"]

preprocess_dir="datasets/preprocessed/"
os.makedirs(preprocess_dir, exist_ok=True)
DataTest.to_csv(preprocess_dir+"train.csv")
# split the original test to 3 parts, auxlliary, validation, test
auxilliary_ratio=0.1
valid_ratio=0.1
UserID=DataTest.UserID.unique()
UserID.sort()
n_User=len(UserID)
aux_end=int(auxilliary_ratio*n_User)
vali_end=aux_end+int(valid_ratio*n_User)

auxilliary=UserID[:aux_end]
Data_aux=DataTest[DataTest.UserID.isin(auxilliary)]
Data_aux.to_csv(preprocess_dir+"aux.csv")

validation=UserID[aux_end:vali_end]
Data_vali=DataTest[DataTest.UserID.isin(validation)]
Data_vali.to_csv(preprocess_dir+"vali.csv")

test=UserID[vali_end:]
Data_test=DataTest[DataTest.UserID.isin(test)]
Data_test.to_csv(preprocess_dir+"test.csv")