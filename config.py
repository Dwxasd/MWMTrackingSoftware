#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Code used for object tracking. Primary purpose is for tracking a mouse in
the Morris Water Maze experiment.

config.py: this contains the default configuration for various elements of the
tracking software, such as neural network properties, particle filter
properties, etc.

"""


class Configuration:
    """
    Class for storing and maintaining configuration of the tracking software
    """

    def __init__(self):

        # ================== Neural Network properties ================ #
        self.num_layers = 2

        # ================== Particle Filter properties ================ #
        self.num_particles = 200

        # ====================== Video properties ====================== #
        self.framerate = 30


if __name__ == '__main__':
    print("Please run the file 'main.py'")
