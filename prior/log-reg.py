import collections, math, random
from scipy.special import expit
import numpy as np

class LogisticRegression:

    def __init__(self, train, test, T, wSize):
        self.train = train
        self.test = test
        self.w = np.array([0] * wSize)
        self.wSize = wSize
        self.d = math.log(T / (1 - T))
        self.T = T
        self.trained = False

    def SGA(self, epochs, eta, step, b1 = 0.9, b2 = 0.999, e = 10e-8, s = 1):
        m = np.array([0] * self.wSize)
        v = np.array([0] * self.wSize)

        it = 1
        for _ in range(epochs):
            random.shuffle(self.train)
            for data in self.train:
                if it % 1000 == 0: print("\rEpoch {} of SGD.".format(it), end = '', flush = True)
                x, y = data

                # update gradient
                if step == 'normal':
                    grad = x * (y - expit(s * np.dot(self.w, x)))
                    update = eta * grad
                
                # update ADAM hyperparameters
                elif step == 'adam':
                    grad = x * (y - expit(s * np.dot(self.w, x)))
                    m = b1 * m + (1 - b1) * grad
                    v = b2 * v + (1 - b2) * np.square(grad)
                    m_hat = m / (1 - np.power(b1, it))
                    v_hat = v / (1 - np.power(b2, it))
                    update = eta * m_hat / (np.sqrt(v_hat) + e)

                # update weights
                self.w = self.w + update
                it += 1

        self.trained = True
        
    def trainLR(self, epochs, eta, step):
        self.SGA(epochs, eta, step)

    def testDemo(self, attempt, ordering):
        values = []
        for key in ordering: values.append(attempt[0][key])
        res = expit(np.dot(self.w, np.array(values)))
        return res

    def testLR(self, s = 1):
        assert self.trained == True
        totalV, totalIV, corrV, corrIV = 0, 0, 0, 0
        predictV, predictIV = 0, 0

        # parse test results
        for data in self.test:
            x, y = data
            prediction = int(np.dot(self.w, x) > self.d)
            if prediction == 1: predictV += 1
            elif prediction == 0: predictIV += 1
            print("Y: {}, Probability Prediction: {}".format(y, expit(s * np.dot(self.w, x))))
            if y == 0:
                totalIV += 1
                if prediction == 0: corrIV += 1
            elif y == 1:
                totalV += 1
                if prediction == 1: corrV += 1
        accuracy = (corrV + corrIV) / (totalV + totalIV)
        precision = corrV / predictV
        recall = corrV / (corrV + (predictIV - corrIV))

        # print results
        print("\n\n***** Test Results *****")
        print("Total Accuracy: {}".format(accuracy))
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("False-positive rate: {}".format((predictV - corrV) / predictIV))
        print("F1 Score: {}".format(2 * (precision * recall) / (precision + recall)))
        print("Predicted valid: {}".format(predictV))
        print("Correct valid: {} out of {}".format(corrV, totalV))
        print("Predicted invalid: {}".format(predictIV))
        print("Correct invalid: {} out of {}".format(corrIV, totalIV))
        print("Train Size: {}".format(len(self.train)))
        print("Test Size: {}\n".format(len(self.test)))
        
