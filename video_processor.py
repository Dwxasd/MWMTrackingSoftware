#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Code used for object tracking. Primary purpose is for tracking a mouse in
the Morris Water Maze experiment.

video_processor.py: this contains the code for generating a video
reader/writer for processing video frames during tracking.

"""
import cv2


class VideoProcessor:
    """
    Video Processor class
    """

    def __init__(self):
        """constructor"""

        self.writer = self.create_writer

    def create_writer(self, filename, frame_size, fps=20):
        """Opens and returns a video for writing.
        
        :param filename: string - Filename for output video
        :param frame_size: tuple - width, height of output video resolution
        :param fps: int - frames per second of output video
        
        :return: 
        """

        # assign fourcc codec of video writer
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        filename = filename.replace('.mp4', '.avi')  # export as .avi

        # create video writer
        videoWriter = cv2.VideoWriter(filename, fourcc, fps, frame_size)

        return videoWriter

    def frame_generator(self, filename):
        """
        A frame generator that yields a frame every next() call.
        Will return 'None' if there are no frames left in video.

        :return:
        """

        # if video name is provided; then load new video into processor
        video = cv2.VideoCapture(filename)

        # while video is still opened
        while video.isOpened():
            # get next
            ret, frame = video.read()

            if ret:
                yield frame
            else:
                break

        video.release()
        yield None


if __name__ == '__main__':
    print("Please run the file 'main.py'")
