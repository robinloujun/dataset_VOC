#! /usr/bin/python
# -*- coding:UTF-8 -*-
# Convert the .txt bbox annotation to the format of Pascal VOC 2007
# Author: Jun Lou

import os, sys
import glob
from PIL import Image
import cv2
import argparse

def main(args):
    '''main function'''

    if not os.path.exists(args.xml_dir):
            os.makedirs(args.xml_dir) 

    img_lists = glob.glob(args.img_dir + '/*.jpg')

    img_basenames = []
    for item in img_Lists:
        img_basenames.append(os.path.basename(item))

    img_nr = args.start_index

    img_names = []
    for item in img_basenames:
        doc_name, extension = os.path.splitext(item)
        img_names.append(doc_name)

    for img_name in img_names:
        img = Image.open((args.img_dir + '/' + img_name + '.jpg'))
        width, height = img.size

        if (args.enable_rename):
            file_name = str(img_nr)
            old_name = args.img_dir + '/' + img_name + '.jpg'
            new_name = args.img_dir + '/' + file_name + '.jpg'
            os.rename(old_name, new_name) 
        else:
            file_name = img_name

        txt_annotations = open(args.txt_dir + '/' + img_name + '.txt').read().splitlines()

        # write in xml file
        xml_file = open((args.xml_dir + '/' + file_name + '.xml'), 'w')
        xml_file.write('<annotation>\n')
        xml_file.write('    <folder>' + args.dataset_name + '</folder>\n')
        xml_file.write('    <filename>' + file_name + '.jpg' + '</filename>\n')
        xml_file.write('    <size>\n')
        xml_file.write('        <width>' + str(width) + '</width>\n')
        xml_file.write('        <height>' + str(height) + '</height>\n')
        xml_file.write('        <depth>3</depth>\n')
        xml_file.write('    </size>\n')

        for txt_annotation in txt_annotations:
            coordinate = txt_annotation.split(' ')
            xml_file.write('    <object>\n')
            xml_file.write('        <name>' + args.class_name + '</name>\n')
            xml_file.write('        <pose>Unspecified</pose>\n')
            xml_file.write('        <truncated>0</truncated>\n')
            xml_file.write('        <difficult>0</difficult>\n')
            xml_file.write('        <bndbox>\n')
            xml_file.write('            <xmin>' + str(coordinate[0]) + '</xmin>\n')
            xml_file.write('            <ymin>' + str(coordinate[1]) + '</ymin>\n')
            xml_file.write('            <xmax>' + str(coordinate[2]) + '</xmax>\n')
            xml_file.write('            <ymax>' + str(coordinate[3]) + '</ymax>\n')
            xml_file.write('        </bndbox>\n')
            xml_file.write('    </object>\n')

        xml_file.write('</annotation>')

        img_nr += 1

        if (args.enable_rename):
            print('Renamed the image ' + img_name + '.jpg with ' + file_name + '.jpg and created the xml file ' + str(item_06d) + '.xml')
        else:
            print('Created the xml file ' + file_name + '.xml')

    print('Well, all done!')

def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default = '/home/robin/datasets/Images')
    parser.add_argument('--txt_dir', default = '/home/robin/datasets/Annotations')
    parser.add_argument('--xml_dir', default = '/home/robin/datasets/Annotations_xml')
    parser.add_argument('--dataset_name', default = 'VOC2007')
    parser.add_argument('--class_name', default = 'car')
    parser.add_argument('--start_index', default = 0)
    parser.add_argument('--enable_rename', default = false)
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
