import pickle
import pandas
import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random


def mostra(img, name='Name'):
    img1 = img.copy()

    img1 = img1.astype(np.uint8)

    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name, img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class Handler():
    def __init__(self):
        self.datadir = "data\\US\\originals"
        self.categories = ["benign", "malignant"]
        self.data = []
        self.img_size = 52

    def write_datafile(self):
        self.read_data()

        #random.shuffle(self.data)

        X = []
        Y = []

        for features, label in self.data:
            X.append(features)
            Y.append(label)

        X = np.array(X).reshape(-1, self.img_size, self.img_size, 1)

        pickle_out = open("X.pickle", "wb")
        pickle.dump(X, pickle_out)
        pickle_out.close()

        pickle_out = open("Y.pickle", "wb")
        pickle.dump(Y, pickle_out)
        pickle_out.close()

    def read_data(self):
        for category in self.categories:
            path = os.path.join(self.datadir, category)

            class_num = self.categories.index(category)
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array, (self.img_size, self.img_size))
                    self.data.append([new_array, class_num])
                except Exception as e:
                    pass

    def read_datafile(self):
        pickle_in = open("X.pickle", "rb")
        X = pickle.load(pickle_in)

        #mostra(X_nd[1])

        pickle_in = open("Y.pickle", "rb")
        Y = pickle.load(pickle_in)

        return X, Y
