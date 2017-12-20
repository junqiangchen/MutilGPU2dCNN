from __future__ import division
from cnn2d.layer import dense_to_one_hot
from cnn2d.model import cnn2dModule
import numpy as np
import pandas as pd


def train():
    '''
    Preprocessing for dataset
    '''
    # Read data set (Train data from CSV file)
    csv_data = pd.read_csv('train.csv')
    data = csv_data.iloc[:, :].values
    np.random.shuffle(data)

    # Extracting images and labels from given data
    # For images
    images = data[:, 1:]
    images = images.astype(np.float)
    # Normalize from [0:255] => [0.0:1.0]
    images = np.multiply(images, 1.0 / 255.0)
    # For labels
    labels_flat = data[:, 0]
    labels_count = np.unique(labels_flat).shape[0]
    labels = dense_to_one_hot(labels_flat, labels_count)
    labels = labels.astype(np.uint8)
    # Split data into training & validation
    train_images = images[0:]
    train_labels = labels[0:]

    cnn2d = cnn2dModule(64, 64, 3, 2)
    cnn2d.train(train_images, train_labels, "E:\pythonworkspace\\test\\model\\cnn2d",
                "E:\\pythonworkspace\\\test\\log", 0.0001, 0.8, 0.7, 5, 100)


def predict():
    test_imagesdata = pd.read_csv("test.csv")
    test_images = test_imagesdata.iloc[:, 1:].values
    test_labels = test_imagesdata[[0]].values
    test_images = test_images.astype(np.float)
    # convert from [0:255] => [0.0:1.0]
    test_images = np.multiply(test_images, 1.0 / 255.0)
    cnn2d = cnn2dModule(64, 64, 3, 2)
    predictvalue = cnn2d.prediction("E:\pythonworkspace\\test\\model\\cnn2d",
                                    test_images)
    print(predictvalue[0])


def main(argv):
    if argv == 1:
        train()
    else:
        predict()


if __name__ == "__main__":
    main(2)
