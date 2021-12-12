import os
import numpy as np
import pandas as pd
import tensorflow as tf
from Base import Train, Predict, Plot, Utility


def main(boolNormalize, boolDeep, boolBias):
    strProjectFolder = os.path.join(os.path.dirname(__file__))
    strDatasetFolder=os.path.join(strProjectFolder,"datasets/preprocessed")
    if boolNormalize:
        if boolDeep:
            strOutputPath = "Output/" + "Deep" + "Normal/"
        else:
            if boolBias:
                strOutputPath = "Output/" + "Bias" + "Normal/"
            else:
                strOutputPath = "Output/" + "unBias" + "Normal/"
    else:
        if boolDeep:
            strOutputPath = "Output/" + "Deep/" 
        else:
            if boolBias:
                strOutputPath = "Output/" + "Bias/" 
            else:
                strOutputPath = "Output/" + "unBias/"
#     test_only=True
    if test_only:
        DataTest = pd.read_csv(os.path.join(strDatasetFolder, "test.csv"), usecols=["UserID", "ItemID", "Rating"])
        arrayTestUser = DataTest["UserID"].values
        arrayTestItem= DataTest["ItemID"].values
        arrayTestRate = DataTest["Rating"].values
        if test_only:
            arrayTestPredict = Predict.makePredict(arrayTestUser, arrayTestItem, \
                                                   strProjectFolder=strProjectFolder, strOutputPath=strOutputPath)
            TestMSE = np.mean(np.square(np.squeeze(arrayTestPredict) - arrayTestRate))
            TestRMSE = np.sqrt(np.mean(np.square(np.squeeze(arrayTestPredict) - arrayTestRate)))    
            print("TestMSE,TestRMSE",TestMSE,TestRMSE)
            return 
                
    os.makedirs(os.path.join(strProjectFolder,strOutputPath), exist_ok=True)
    DataTrain = pd.read_csv(os.path.join(strDatasetFolder, "train.csv"), usecols=["UserID", "ItemID", "Rating"])
    DataAux = pd.read_csv(os.path.join(strDatasetFolder, "aux.csv"), usecols=["UserID", "ItemID", "Rating"])
    DataValid = pd.read_csv(os.path.join(strDatasetFolder, "vali.csv"), usecols=["UserID", "ItemID", "Rating"])

#     DataItem = Utility.getLabelEncoder(DataItem)

    DataTrain = DataTrain.sample(frac=1.0, random_state=10)
    intUserSize = len(DataTrain["UserID"].drop_duplicates())
    intItemSize = len(DataTrain["ItemID"].drop_duplicates())


    intLatentSize = 32
    
    arrayTrainUser = DataTrain["UserID"].values
    arrayTrainItem= DataTrain["ItemID"].values
    arrayTrainRate = DataTrain["Rating"].values

    arrayValidUser = DataValid["UserID"].values
    arrayValidItem= DataValid["ItemID"].values
    arrayValidRate = DataValid["Rating"].values

   
    arrayTrainRateAvg = np.mean(arrayTrainRate)
    arrayTrainRateStd = np.std(arrayTrainRate)


    Train.getTrain(arrayTrainUser=arrayTrainUser, arrayTrainMovie=arrayTrainItem, arrayTrainRate=arrayTrainRate
                , arrayValidUser=arrayValidUser, arrayValidMovie=arrayValidItem, arrayValidRate=arrayValidRate
                , intUserSize=intUserSize
                , intMovieSize=intItemSize
                , intLatentSize=intLatentSize
                , boolBias=boolBias
                , boolDeep=boolDeep
                , strProjectFolder=strProjectFolder, strOutputPath=strOutputPath)

    Plot.plotModel(strProjectFolder, strOutputPath)
    Plot.plotLossAccuracyCurves(strProjectFolder, strOutputPath)

#     if not boolDeep:
#         Plot.plotItemEmbeddingTSNE(DataItem, strProjectFolder, strOutputPath)

    arrayTrainPredict = Predict.makePredict(arrayTrainUser, arrayTrainItem, strProjectFolder=strProjectFolder, strOutputPath=strOutputPath)
    trainMSE = np.mean(np.square(np.squeeze(arrayTrainPredict) - arrayTrainRate))
    trainRMSE = np.sqrt(np.mean(np.square(np.squeeze(arrayTrainPredict) - arrayTrainRate)))

    arrayValidPredict = Predict.makePredict(arrayValidUser, arrayValidItem, strProjectFolder=strProjectFolder, strOutputPath=strOutputPath)
    ValidMSE = np.mean(np.square(np.squeeze(arrayValidPredict) - arrayValidRate))
    ValidRMSE = np.sqrt(np.mean(np.square(np.squeeze(arrayValidPredict) - arrayValidRate)))
    print(trainMSE, trainRMSE, ValidMSE, ValidRMSE)
    

if __name__ == "__main__":
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config_tf.log_device_placement = True  # to log device placement (on which device the operation ran)
                                        # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config_tf)
   
    main(boolNormalize=False, boolDeep=False, boolBias=True)



