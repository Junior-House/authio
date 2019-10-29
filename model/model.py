import numpy as np
import tensorflow as tf
import pandas as pd
import keras
from keras.models import Sequential
import keras.layers as layers
import matplotlib.pyplot as plt

# Function: prepareDataset
# Description: 
#   Accepts the file names for the processed valid and invalid
#   data, as well as the train-test split values, loads the data,
#   and splits the data. 
def prepareDataset(validDataFileName, invalidDataFileName, dataSplit, invalidCount=2000) -> tuple:
    assert len(dataSplit) == 2 and dataSplit[0] + dataSplit[1] == 1.0

    # load valid data
    validDataframe = pd.read_csv(validDataFileName)
    validDataframe['label'] = 1
    cols = validDataframe.columns.tolist()
    cols.insert(0, cols.pop(cols.index('label')))
    validDataframe = validDataframe.reindex(columns=cols)
    validDataframe = validDataframe.as_matrix()

    # load invalid data
    invalidDataframe = pd.read_csv(invalidDataFileName)
    invalidDataframe['label'] = 0
    cols = invalidDataframe.columns.tolist()
    cols.insert(0, cols.pop(cols.index('label')))
    invalidDataframe = invalidDataframe.reindex(columns=cols)
    invalidDataframe = invalidDataframe.as_matrix()
    np.random.shuffle(invalidDataframe)
    invalidDataframe = invalidDataframe[:invalidCount, :]

    # create train/test split
    validTrainData = validDataframe[:int(validDataframe.shape[0] * dataSplit[0]), :]
    validTestData = validDataframe[int(validDataframe.shape[0] * dataSplit[0]):, :]
    invalidTrainData = invalidDataframe[:int(invalidDataframe.shape[0] * dataSplit[0]), :]
    invalidTestData= invalidDataframe[int(invalidDataframe.shape[0] * dataSplit[0]):, :]
    trainData = np.vstack((validTrainData, invalidTrainData))
    testData = np.vstack((validTestData, invalidTestData))
    
    # return scrambled data
    np.random.shuffle(trainData)
    np.random.shuffle(testData)
    return (trainData, testData)

# Function: sliceDataset
# Description: 
#   Accepts a tuple of train and test data matrices and 
#   organizes the data into train/test data and train/test
#   labels.
def sliceDataset(data) -> tuple:
    trainData = data[0]
    testData = data[1]

    # organize between labels and data
    xTrain = trainData[:, 1:]
    xTest = testData[:, 1:]
    yTrain = trainData[:, 0]
    yTest = testData[:, 0]

    # one-hot encode data
    outputDim = 2
    yTrain = keras.utils.to_categorical(yTrain, outputDim)
    yTest = keras.utils.to_categorical(yTest, outputDim)
    return (xTrain, xTest, yTrain, yTest)

# Function: logisticRegressionModel
# Description: 
#   Accepts the train/test data and train/test labels, 
#   runs, and returns a simple logistic regression model.
def logisticRegressionModel(data) -> None:
    xTrain, xTest, yTrain, yTest = data
    inputDim, outputDim = xTrain.shape[1], 2

    # build model
    model = Sequential()
    model.add(layers.Dense(outputDim, input_dim=inputDim, activation='softmax'))

    # define hyperparameters
    batchSize = 32
    numEpochs = 20

    # compile model
    model.summary()
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy']) 
    model.fit(xTrain, yTrain, batch_size=batchSize, epochs=numEpochs, verbose=True, validation_data=(xTest, yTest))
    return model

# Function: evaluateModel
# Description: 
#   Accepts a Keras model and a dataset and 
#   evaluates custom metrics over the model.
def evaluateModel(model, data):
    xTrain, xTest, yTrain, yTest = data
    true_positives, true_negatives = 0, 0
    false_positives, false_negatives = 0, 0

    # compute custom metrics
    results = tf.nn.softmax(model.predict(xTest))
    for i in range(results.shape[0]):
        y = int(yTest[i][1])
        yHat = int(results[i][1] > results[i][0])
        if (y and yHat): true_positives += 1
        elif (not y and not yHat): true_negatives += 1
        elif (y and not yHat): false_positives += 1
        else: false_negatives += 1
            
    # print metric results
    print("\nTrue positives: {} / {}".format(true_positives, true_positives + false_negatives))
    print("True negatives: {} / {}".format(true_negatives, true_negatives + false_positives))
    print("False positives: {} / {}".format(false_positives, true_negatives + false_positives))
    print("False negatives: {} / {}".format(false_negatives, true_positives + false_negatives))
    print("\nAccuracy: {}".format((true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)))
    print("Recall: {}".format((true_positives) / (true_positives + false_negatives)))
    print("Precision: {}".format((true_positives) / (true_positives + false_positives)))
    print("F1: {}".format((2 * true_positives) / (2 * true_positives + false_positives + false_negatives)))

def main() -> None:
    data = prepareDataset('data\\processed-valid-data.csv', 'data\\processed-invalid-data.csv', (0.90, 0.10))
    data = sliceDataset(data)
    logisticRegressionModel(data)

if __name__ == "__main__":
    main()