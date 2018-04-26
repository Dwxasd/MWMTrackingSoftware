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

        self.video = None

        self.writer = self.create_writer

    def create_writer(self, filename, frame_size, fps=20):
        """Opens and returns a video for writing.
        
        :param filename: string - Filename for output video
        :param frame_size: tuple - width, height of output video resolution
        :param fps: int - frames per second of output video
        
        :return: 
        """

        # assign fourcc codec of video writer
        fourcc = cv2.cv.CV_FOURCC(*'MP4V')
        filename = filename.replace('.mp4', '.avi')  # export as .avi

        # create video writer
        videoWriter = cv2.VideoWriter(filename, fourcc, fps, frame_size)

        return videoWriter

    def load_video(self, filename):
        """
        load a video to make that video the current file to be processed
        
        :param filename: string - filename for video
        :return: 
        """

        # store video in VideoProcessor object
        self.video = cv2.VideoCapture(filename)

    def frame_generator(self, filename=''):
        """
        A frame generator that yields a frame every next() call.
        Will return 'None' if there are no frames left in video.

        :return:
        """

        # if video name is provided; then load new video into processor
        if filename:
            self.load_video(filename)

        # while video is still opened
        while self.video.isOpened():
            # get next
            ret, frame = self.video.read()

            if ret:
                yield frame
            else:
                break

        self.video.release()
        yield None


if __name__ == '__main__':
    print("Please run the file 'main.py'")
