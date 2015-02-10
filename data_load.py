"""
This module provides an utility class to load images from the training set and splits the data into
traininig and test sets
"""

import numpy as np
from os import listdir
from PIL import Image, ImageOps
from sklearn.cross_validation import train_test_split 
from os.path import isfile, join, isdir


class DataLoader(object):
    classes_ammount = 0
    classes_names = []
    size = 25, 25
    
    def __init__(self, workspace_dir):
        self.workspace_dir = workspace_dir
        dataset, class_labels = self.load(workspace_dir)
        
        # scale the data to the range [0, 1] and then construct the training
        # and testing splits
        (self.trainX, self.testX, self.trainY, self.testY) = \
            train_test_split(dataset / 255.0, class_labels, test_size = 0.33)
            
    def load(self, workspace_dir):            
        print "Loading data..."
        images = []
        class_labels = []
        label = 0

        self.classes_names = [f for f in listdir(workspace_dir) if isdir(join(workspace_dir, f))]
        self.classes_ammount = len(self.classes_names)
        for folder in self.classes_names:
            files = [f for f in listdir(join(workspace_dir, folder)) if isfile(join(workspace_dir,folder,f))]
            for f in files:
                if f[-4:] != ".jpg":
                    continue
                jpgfile = Image.open(join(workspace_dir, folder, f))
                jpgfile = ImageOps.fit(jpgfile, self.size, Image.ANTIALIAS)
                images.append(np.asarray(jpgfile).reshape(self.size[0] * self.size[1],))
                class_labels.append(label)
            label += 1
    
        class_labels = np.asarray(class_labels)
        images = np.asarray(images)
        
        print "Download is finished"
        return images, class_labels


    

