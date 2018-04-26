#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Code used for object tracking. Primary purpose is for tracking a mouse in
the Morris Water Maze experiment.

main.py: this contains the main code for running the tracking software.

"""


class Filter():
    """
    Particle filter class
    """

    def __init__(self, num_particles=200):
        """constructor"""

        self.num_particles = num_particles
        self.model = self.create_model()

    def create_model(self):
        """
        create model

        :return: return the model
        """

        #TODO: create

        return

    def compute_sim(self):
        """
        train the neural network
        """



        pass

    def query(self):
        """
        query the neural network to find output
        :return:
        """

        return


if __name__ == '__main__':
    print("Please run the file 'main.py'")
