#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Code used for object tracking. Primary purpose is for tracking a mouse in
the Morris Water Maze experiment.

util.py: this contains all utility functions for use in the tracking software.

"""
import os
import cv2


def load_files(dirname='train_files'):
    """
    Load files from a specified folder passed in as a string 'dirname'.

    :param dirname: string - directory name to load files for parsing
    :return: output_files: list - list of video files found in dirname directory
    """

    # initialize
    output_files = []

    #TODO: Load files

    return output_files


def show_frame(frame, save_img=False, output_name=''):
    """
    Show video frame to the screen, and also allow for additional options to
    save the image to a file.
    
    :param frame: ndarray - the image frame to be shown
    :param save_img: bool - whether or not to save frame to an output image
    :param output_name: string - file name for output image to be saved 
    :return: 
    """

    cv2.imshow(frame)


if __name__ == '__main__':
    print("Please run the file 'main.py'")
