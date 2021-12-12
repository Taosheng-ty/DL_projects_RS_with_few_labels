import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
strProjectFolder="datasets/R3-Yahoo/"

DataTest = pd.read_csv(os.path.join(strProjectFolder, "ydata-ymusic-rating-study-v1_0-test.txt"),sep="\\t",header=None)
DataTest.columns=["UserID", "ItemID", "Rating"]
DataTrain = pd.read_csv(os.path.join(strProjectFolder, "ydata-ymusic-rating-study-v1_0-train.txt"),sep="\\t",header=None)
DataTrain.columns=["UserID", "ItemID", "Rating"]
bins = np.arange(0,6,0.5)+0.5
output_dir="output/label_statistics/"
os.makedirs(output_dir, exist_ok=True)
##plot DataTrain  label statistics
ax=DataTrain.Rating.plot.hist(bins=bins)
plt.xticks(np.arange(1, 6))
plt.xlabel("Preference label")
plt.ylabel("Frequency")
fig = ax.get_figure()
fig.savefig(output_dir+'train_label_statistic.pdf',bbox_inches="tight")
fig.clear()
##plot DataTest  label statistics
ax=DataTest.Rating.plot.hist(bins=bins)
plt.xticks(np.arange(1, 6))
plt.xlabel("Preference label")
plt.ylabel("Frequency")
fig = ax.get_figure()

fig.savefig(output_dir+'test_label_statistic.pdf',bbox_inches="tight")
fig.clear()