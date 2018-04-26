#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Code used for object tracking. Primary purpose is for tracking a mouse in
the Morris Water Maze experiment.

config.py: this contains the default configuration for various elements of the
tracking software, such as neural network properties, particle filter
properties, etc.

"""
import os


class Configuration:
    """
    Class for storing and maintaining configuration of the tracking software
    """

    def __init__(self):

        # ================== Neural Network properties ================ #
        # directory information
        self.training_dir = 'train_files'

        # cNN properties
        self.num_layers = 2


        # ================== Particle Filter properties ================ #
        # directory information
        self.template_dir = 'templates'
        self.num_particles = 200

        # particle filter properties

        # ====================== Video properties ====================== #
        self.framerate = 30

        # ======================= Misc properties ====================== #
        self.cur_dir = os.path.dirname(os.path.realpath(__file__))
        self.test_file = os.path.join(self.cur_dir, 'testVids/testVid.wmv')
        self.test_out = os.path.join(self.cur_dir, 'output/test_out.mp4')


if __name__ == '__main__':
    print("Please run the file 'main.py'")
