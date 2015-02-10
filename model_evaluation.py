"""
This module provides an utility class to evaluate the loss of the built model and to display random 10 images with the values of actual and predicted class """

import cv2
from data_load import DataLoader
import numpy as np


class ResultEvaluator(object):
    
    def __init__(self,data_loader):
        self.testX = data_loader.testX
        self.testY = data_loader.testY
        self.size = data_loader.size
    
    def display(self,dbn):
        for i in np.random.choice(np.arange(0, len(self.testY)), size = (10,)):
            # classification
            pred = dbn.predict(np.atleast_2d(self.testX[i]))
            # reshape the feature vector to be an image of given size
            image = (self.testX[i] * 255).reshape(self.size[0], self.size[1]).astype("uint8")
            # show the image and prediction
            print "Actual class is {0}, predicted {1}".format(self.testY[i], pred[0])
            cv2.imshow("Class", image)
            cv2.waitKey(0)

    def evaluate_loss(self, y_pred, eps=1e-15):
        predictions = np.clip(y_pred, eps, 1 - eps)
        
        # normalize row sums to 1
        predictions /= predictions.sum(axis=1)[:, np.newaxis]
        
        actual = np.zeros(y_pred.shape)
        n_samples = actual.shape[0]
        actual[np.arange(n_samples), self.testY.astype(int)] = 1
        vectsum = np.sum(actual * np.log(predictions))
        loss = -1.0 / n_samples * vectsum
        return loss


