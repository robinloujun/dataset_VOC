#! /usr/bin/python
# -*- coding:UTF-8 -*-
# Extract the video frames using OpenCV
# Author: Jun Lou

import os
import cv2 
import argparse

def main(args):
	'''main function'''

	vc = cv2.VideoCapture(args.video_dir) 

	if vc.isOpened(): 
		rval, _ = vc.read() 
		print('Finish reading the video')

		if not os.path.exists(args.img_dir):
			os.makedirs(args.img_dir) 

	else: 
		rval = False 
		print('Video not found')

	count = 0
	img_nr = args.start_index

	while rval: 
		_, frame = vc.read() 
		img_name = format(img_nr, '06d')
		if count % args.freq == 0:
			cv2.imwrite(args.img_dir + '/' + str(img_name) + '.jpg' , frame) 
			print('Image '+str(img_name)+'.jpg extracted...')
			img_nr += 1
		count += 1
		cv2.waitKey(1) 

	vc.release()

	print('Well, all done!')


def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', default = '/home/robin/Documents/video.mp4')
    parser.add_argument('--img_dir', default = '/home/robin/Documents/Imgs')
    parser.add_argument('--freq', default = 10)
    parser.add_argument('--start_index', default = 0)
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())