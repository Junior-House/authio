import csv
import numpy as np
import os

# Function: collectLabelledData
# Description: 
#   Accepts the name of the raw data CSV file and the ID of
#   the valid user, labels the row data correctly, and returns 
#   the resultant data as a list of lists.
def collectLabelledData(rawDataFileName, userId) -> list:
    labelledData = []

    # operate on raw data file
    with open(rawDataFileName, 'r') as rawDataFile:
        rawData = csv.reader(rawDataFile, delimiter=',', lineterminator='\n')

        # iteratively process data
        for rawRow in rawData:
            labelledRow = []

            # label data
            if int(rawRow[0]) == userId: labelledRow.append(1)
            else: labelledRow.append(0)

            # copy keystroke data
            labelledRow += rawRow[3:]
            labelledData.append(labelledRow)

    return labelledData

# Function: processLabelledData
# Description: 
#   Accepts the labelled data, as well as the valid processed data and invalid 
#   processed data file names, collects the labelled data, processes this data, 
#   and writes the invalid and valid output.
def processLabelledData(labelledData, validDataFileName, invalidDataFileName) -> None:
    validLabelledData = []
    invalidLabelledData = []

    # convert data to matrix
    dataMatrix = np.array(labelledData, dtype=np.float32)
    dataMean = dataMatrix.mean(axis=0)[1:]
    dataStd = dataMatrix.std(axis=0)[1:]

    # split labelled data
    for data in labelledData:
        dataArr = np.array(data[1:], dtype=np.float32)

        # normalize data
        dataArr = (dataArr - dataMean) / dataStd

        # sort data
        if data[0] == 1: validLabelledData.append(dataArr.tolist())
        else: invalidLabelledData.append(dataArr.tolist())

    # write data
    writeCSVData(validDataFileName, validLabelledData)
    writeCSVData(invalidDataFileName, invalidLabelledData)

# Function: writeCSVData
# Description: 
#   Accepts a target CSV file name and the data as a list
#   of lists, and writes the data contents row-by-row to
#   the target file.
def writeCSVData(targetFileName, data) -> None:

    # operate on the target data file
    with open(targetFileName, 'w') as targetFile:
        target = csv.writer(targetFile, delimiter=',', lineterminator='\n')

        # iteratively write data
        for row in data:
            target.writerow(row)

# Function: buildData
# Description: 
#   Accepts the ID for the valid user and builds the
#   valid and invalid data set.  
def buildData(validId) -> None:
    try:
        os.remove('data/processed-valid-data.csv')
        os.remove('data/processed-invalid-data.csv')
    except: pass

    labelledData = collectLabelledData('data/raw-data.csv', validId)
    processLabelledData(labelledData, 'data/processed-valid-data.csv', 'data/processed-invalid-data.csv')