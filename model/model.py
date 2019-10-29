import numpy as np
import tensorflow as tf
import pandas as pd
import keras
from keras.models import Sequential
import keras.layers as layers
import matplotlib.pyplot as plt
import keras_metrics as km

# Function: prepareDataset
# Description: 
#   Accepts the file names for the processed valid and invalid
#   data, as well as the train-test split values, loads the data,
#   and splits the data. 
def prepareDataset(validDataFileName, invalidDataFileName, dataSplit, invalidCount=1000, useValidCount=False) -> tuple:
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
    if useValidCount: invalidDataframe = invalidDataframe[:validDataframe.shape[0], :]
    else: invalidDataframe = invalidDataframe[:invalidCount, :]

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
#   runs, and returns a trained logistic regression model.
def logisticRegressionModel(data) -> None:
    xTrain, xTest, yTrain, yTest = data
    inputDim, outputDim = xTrain.shape[1], 2
    
    # define hyperparameters
    batchSize = 48
    numEpochs = 50

     # build model
    model = Sequential()
    model.add(layers.Dense(outputDim, input_dim=inputDim, activation='softmax'))

    # compile model
    model.summary()
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy']) 
    model.fit(xTrain, yTrain, batch_size=batchSize, epochs=numEpochs, verbose=True, validation_data=(xTest, yTest))
    return model

# Function: logisticRegressionModel
# Description: 
#   Accepts the train/test data and train/test labels, 
#   runs, and returns a trained shallow neural network model.
def shallowNeuralNetworkModel(data) -> None:
    xTrain, xTest, yTrain, yTest = data
    inputDim, outputDim = xTrain.shape[1], 2
    
    # define hyperparameters
    batchSize = 64
    numEpochs = 50
    layer_one_size = 5

    # build model
    model = Sequential()
    model.add(layers.Dense(layer_one_size, input_dim=inputDim, activation='tanh'))
    model.add(layers.Dense(outputDim, activation='softmax'))

    # compile model
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', km.binary_true_positive()]) 
    model.fit(xTrain, yTrain, batch_size=batchSize, epochs=numEpochs, verbose=True, validation_data=(xTest, yTest))
    return model

# Function: logisticRegressionModel
# Description: 
#   Accepts the train/test data and train/test labels, 
#   runs, and returns a trained deep neural network model.
def deepNeuralNetworkModel(data) -> None:
    xTrain, xTest, yTrain, yTest = data
    inputDim, outputDim = xTrain.shape[1], 2
    
    # define hyperparameters
    batchSize = 48
    numEpochs = 32
    layer_one_size = 16
    layer_two_size = 16
    layer_three_size = 16

    # build model
    model = Sequential()
    model.add(layers.Dense(layer_one_size, input_dim=inputDim, activation='tanh'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(layer_one_size, activation='tanh'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(layer_two_size, activation='tanh'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(layer_three_size, activation='tanh'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(outputDim, activation='softmax'))

    # compile model
    model.summary()
    optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', km.binary_true_positive()]) 
    history = model.fit(xTrain, yTrain, batch_size=batchSize, epochs=numEpochs, verbose=True, validation_data=(xTest, yTest))

    # plot model progress
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='best')
    plt.show()
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
    data = prepareDataset('data\\processed-valid-data.csv', 'data\\processed-invalid-data.csv', (0.95, 0.05), useValidCount=True)
    data = sliceDataset(data)
    model = deepNeuralNetworkModel(data)
    evaluateModel(model, data)

if __name__ == "__main__":
    main()