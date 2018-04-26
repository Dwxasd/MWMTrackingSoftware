#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Code used for object tracking. Primary purpose is for tracking a mouse in
the Morris Water Maze experiment.

main.py: this contains the main code for running the tracking software.

"""

from util import load_files, show_frame
from config import Configuration
from cnn import Network
from video_processor import VideoProcessor
from particle_filter import Filter

##############################################################################
__author__ = "Chris Cadonic"
__credits__ = ["Chris Cadonc"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Chris Cadonic"
__email__ = "chriscadonic@gmail.com"
__status__ = "Development"
##############################################################################


def main():
    """
    Main code for running the tracking software.

    :return:
    """

    # Load configuration for
    config = Configuration()

    # create a convolutional neural net to train and detect mouse location
    network = Network()

    # create a particle filter for tracking
    pfilter = Filter(config.num_particles)

    # load files and parse
    train_videos = load_files(config.training_dir)

    # load template files and parse
    templates = load_files(config.template_dir)

    # load video processor for extracting frames during tracking
    vid_reader = VideoProcessor()

    vid_reader.load_video('testVids/testVid.mp4')

    image_generator = vid_reader.frame_generator()

    frame = image_generator.__next__()
    h, w, d = frame.shape()

    out_path = 'output/testVid.mp4'
    video_out = vid_reader.create_writer(out_path, (w, h), config.framerate)

    #template = cv2.imread('template.jpg')

    frame_num = 1

    while frame is not None:

        print("Processing frame ", frame_num)

        show_frame(frame)

        video_out.write(frame)

        frame = image_generator.next()

        frame_num += 1

    video_out.release()

    return


if __name__ == '__main__':
    main()