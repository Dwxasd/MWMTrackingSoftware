#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Code used for object tracking. Primary purpose is for tracking a mouse in
the Morris Water Maze experiment.

cnn.py: this file contains the code for implementing a convolutional neural
network to detect a mouse location during swimming during the tracking.

"""


class Network():
    """
    Convolutional Neural Network class
    """

    def __init__(self):
        """constructor"""

        self.model = self.create_model()

    def create_model(self):
        """
        create model

        :return: return the model
        """

        #TODO: Build Keras model

        return

    def train(self):
        """
        train the neural network
        """

        #TODO: Train Keras model

        pass

    def query(self):
        """
        query the neural network to find output
        :return:
        """

        return


if __name__ == '__main__':
    print("Please run the file 'main.py'")
