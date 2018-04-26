#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Code used for object tracking. Primary purpose is for tracking a mouse in
the Morris Water Maze experiment.

vide_processor.py: this contains the code for generating a video
reader/writer for processing video frames during tracking.

"""


class VideoProcessor():
    """
    Video Processor class
    """

    def __init__(self):
        """constructor"""

        self.video = None

        self.writer = self.create_writer()

    def create_writer(self):
        """
        create model

        :return: return the model
        """

        #TODO: create video writer

        return

    def frame_generator(self):
        """
        yield a frame from a provided video

        :return:
        """

        # initialize
        frame = None

        # yield frame from video
        yield frame


if __name__ == '__main__':
    print("Please run the file 'main.py'")
