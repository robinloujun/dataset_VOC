#! /usr/bin/python
# -*- coding:UTF-8 -*-
# Data augmentation for object detection dataset
# cf: https://github.com/maozezhong/CV_ToolBox/blob/master/DataAugForObjectDetection/DataAugmentForObejctDetection.py
# Author: Zezhone Mao
# Modifier: Jun Lou

import time
import random
import cv2
import os
import math
import numpy as np
from skimage.util import random_noise
from skimage import exposure
import xml.etree.ElementTree as ET

class data_augmentation():
    def __init__(self, rotation_rate = 0.5, max_rotation_angle = 5, 
                crop_rate = 0.5, shift_rate = 0.5, change_light_rate = 0.5,
                add_noise_rate = 0.5, flip_rate = 0.5, 
                cutout_rate = 0.5, cut_out_length = 50, cut_out_holes = 1, cut_out_threshold = 0.5):
        self.rotation_rate = rotation_rate
        self.max_rotation_angle = max_rotation_angle
        self.crop_rate = crop_rate
        self.shift_rate = shift_rate
        self.change_light_rate = change_light_rate
        self.add_noise_rate = add_noise_rate
        self.flip_rate = flip_rate
        self.cutout_rate = cutout_rate

        self.cut_out_length = cut_out_length
        self.cut_out_holes = cut_out_holes
        self.cut_out_threshold = cut_out_threshold
    
    def _addNoise(self, img):
        '''
        input:
            img: image array
        output:
            img: image array with additional noise
        '''
        return random_noise(img, mode = 'gaussian', clip = True) * 255

    
    def _changeLight(self, img):
        # flag > 1: darker
        # flag > 1: brighter
        flag = random.uniform(0.5, 1.5) 
        return exposure.adjust_gamma(img, flag)
    
    def _cutout(self, img, bboxes, length = 100, n_holes = 1, threshold = 0.5):
        '''
        original version：https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        Randomly mask out one or more patches from an image.
        Args:
            img : a 3D numpy array,(h,w,c)
            bboxes: bounding box list in the form [[x_min, y_min, x_max, y_max, label_name]....]
            n_holes (int): Number of patches to cut out of each image.
            length (int): The length (in pixels) of each square patch.
        '''
        
        def cal_iou(boxA, boxB):

            # determine the (x, y)-coordinates of the intersection rectangle
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            if xB <= xA or yB <= yA:
                return 0.0

            # compute the area of intersection rectangle
            interArea = (xB - xA + 1) * (yB - yA + 1)

            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            # iou = interArea / float(boxAArea + boxBArea - interArea)
            iou = interArea / float(boxBArea)

            # return the intersection over union value
            return iou

        if img.ndim == 3:
            h,w,c = img.shape
        else:
            _,h,w,c = img.shape
        
        mask = np.ones((h,w,c), np.float32)

        for n in range(n_holes):
            
            overlap = True
            
            while overlap:
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - length // 2, 0, h)
                y2 = np.clip(y + length // 2, 0, h)
                x1 = np.clip(x - length // 2, 0, w)
                x2 = np.clip(x + length // 2, 0, w)

                overlap = False
                for box in bboxes:
                    if cal_iou([x1,y1,x2,y2], box) > threshold:
                        overlap = True
                        break
            
            mask[y1: y2, x1: x2, :] = 0.

        img = img * mask

        return img

    def _rotate_img_bbox(self, img, bboxes, angle = 5, scale = 1.):
        '''
        cf: https://blog.csdn.net/u014540717/article/details/53301195crop_rate
        input:
            img: image array, (h,w,c)
            bboxes: bounding box list in the form [[x_min, y_min, x_max, y_max, label_name]....]
            angle: rotational angle
            scale: default = 1
        output:
            rot_img: ratated image arraz
            rot_bboxes: new bounding box list in the form [[x_min, y_min, x_max, y_max, label_name]....]
        '''

        w = img.shape[1]
        h = img.shape[0]

        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
        nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0,2] += rot_move[0]
        rot_mat[1,2] += rot_move[1]

        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags = cv2.INTER_LANCZOS4)

        rot_bboxes = list()
        for bbox in bboxes:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            point1 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymin, 1]))
            point2 = np.dot(rot_mat, np.array([xmax, (ymin+ymax)/2, 1]))
            point3 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymax, 1]))
            point4 = np.dot(rot_mat, np.array([xmin, (ymin+ymax)/2, 1]))

            concat = np.vstack((point1, point2, point3, point4))

            concat = concat.astype(np.int32)

            rx, ry, rw, rh = cv2.boundingRect(concat)
            rx_min = rx
            ry_min = ry
            rx_max = rx+rw
            ry_max = ry+rh

            rot_bboxes.append([rx_min, ry_min, rx_max, ry_max, bbox[4]])
        
        return rot_img, rot_bboxes

    def _crop_img_bboxes(self, img, bboxes):
        '''
        input:
            img: image array
            bboxes: bounding box list in the form [[x_min, y_min, x_max, y_max, label_name]....]
        output:
            crop_img: cropped image array
            crop_bboxes: cropped box list in the form [[x_min, y_min, x_max, y_max, label_name]....]
        '''

        w = img.shape[1]
        h = img.shape[0]
        x_min = w
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])
        
        d_to_left = x_min
        d_to_right = w - x_max
        d_to_top = y_min
        d_to_bottom = h - y_max

        crop_x_min = int(x_min - random.uniform(0, d_to_left))
        crop_y_min = int(y_min - random.uniform(0, d_to_top))
        crop_x_max = int(x_max + random.uniform(0, d_to_right))
        crop_y_max = int(y_max + random.uniform(0, d_to_bottom))

        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max = min(w, crop_x_max)
        crop_y_max = min(h, crop_y_max)

        crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        
        crop_bboxes = list()
        for bbox in bboxes:
            crop_bboxes.append([bbox[0]-crop_x_min, bbox[1]-crop_y_min, bbox[2]-crop_x_min, bbox[3]-crop_y_min, bbox[4]])
        
        return crop_img, crop_bboxes
  
    # 平移
    def _shift_pic_bboxes(self, img, bboxes):
        '''
        cf: https://blog.csdn.net/sty945/article/details/79387054
        input:
            img: image array
            bboxes: bounding box list in the form [[x_min, y_min, x_max, y_max, label_name]....]
        output:
            shift_img: shifted image array
            shift_bboxes: shifted bounding box list in the form [[x_min, y_min, x_max, y_max, label_name]....]
        '''

        w = img.shape[1]
        h = img.shape[0]
        x_min = w
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])
        
        d_to_left = x_min
        d_to_right = w - x_max
        d_to_top = y_min
        d_to_bottom = h - y_max

        x = random.uniform(-(d_to_left-1) / 3, (d_to_right-1) / 3)
        y = random.uniform(-(d_to_top-1) / 3, (d_to_bottom-1) / 3)
        
        M = np.float32([[1, 0, x], [0, 1, y]])
        shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        shift_bboxes = list()
        for bbox in bboxes:
            shift_bboxes.append([bbox[0]+x, bbox[1]+y, bbox[2]+x, bbox[3]+y, bbox[4]])

        return shift_img, shift_bboxes

    def _filp_pic_bboxes(self, img, bboxes):
        '''
            cf: https://blog.csdn.net/jningwei/article/details/78753607
        input:
            img: image array
            bboxes: bounding box list in the form [[x_min, y_min, x_max, y_max, label_name]....]
        output:
            flip_img: flipped image array
            flip_bboxes: flipped bounding box list in the form [[x_min, y_min, x_max, y_max, label_name]....]
        '''

        import copy
        flip_img = copy.deepcopy(img)
        if random.random() < 0.5: 
            horizon = True
        else:
            horizon = False
        h,w,_ = img.shape
        if horizon:
            flip_img =  cv2.flip(flip_img, 1)
        else:
            flip_img = cv2.flip(flip_img, 0)

        flip_bboxes = list()
        for box in bboxes:
            x_min = box[0]
            y_min = box[1]
            x_max = box[2]
            y_max = box[3]
            if horizon:
                flip_bboxes.append([w-x_max, y_min, w-x_min, y_max, box[4]])
            else:
                flip_bboxes.append([x_min, h-y_max, x_max, h-y_min, box[4]])

        return flip_img, flip_bboxes

    def run(self, img, bboxes):
        '''
        input:
            img: image array
            bboxes: bounding box list in the form [[x_min, y_min, x_max, y_max, label_name]....]
        output:
            img: image array after data augmentation
            bboxed: bounding box list in the form [[x_min, y_min, x_max, y_max, label_name]....] after data augmentation
        '''
        change_num = 0
        print('------')
        while change_num < 1:

            if random.random() < self.crop_rate:
                print('Cropped')
                change_num += 1
                img, bboxes = self._crop_img_bboxes(img, bboxes)
            
            if random.random() > self.rotation_rate:
                print('Rotated')
                change_num += 1
                angle = random.sample([90, 180, 270],1)[0]
                scale = random.uniform(0.5, 0.8)
                img, bboxes = self._rotate_img_bbox(img, bboxes, angle, scale)
            
            if random.random() < self.shift_rate:
                print('Shifted')
                change_num += 1
                img, bboxes = self._shift_pic_bboxes(img, bboxes)
            
            if random.random() > self.change_light_rate:
                print('Light rate changed')
                change_num += 1
                img = self._changeLight(img)
            
            if random.random() < self.add_noise_rate:
                print('Noise added')
                change_num += 1
                img = self._addNoise(img)

            if random.random() < self.cutout_rate:
                print('Cutout done')
                change_num += 1
                img = self._cutout(img, bboxes, length=self.cut_out_length, n_holes=self.cut_out_holes, threshold=self.cut_out_threshold)

            if random.random() < self.flip_rate:
                print('Flipped')
                change_num += 1
                img, bboxes = self._filp_pic_bboxes(img, bboxes)
            print('\n')
        return img, bboxes    

def parse_xml(xml_path):
    '''
    input:
        xml_path: dir to xml file
    output:
        bboxes: bounding box list in the form [[x_min, y_min, x_max, y_max, label_name]....]
    '''

    tree = ET.parse(xml_path)       
    root = tree.getroot()
    objs = root.findall('object')
    coords = list()
    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        box = obj.find('bndbox')
        x_min = int(box[0].text)
        y_min = int(box[1].text)
        x_max = int(box[2].text)
        y_max = int(box[3].text)
        coords.append([x_min, y_min, x_max, y_max, name])
    return coords

def main(args):
    '''main function'''

    import shutil

    need_aug_num = 1                  

    data_aug = data_augmentation()

    if not os.path.exists(args.img_dir_aug):
            os.makedirs(args.img_dir_aug) 
    if not os.path.exists(args.xml_dir_aug):
            os.makedirs(args.xml_dir_aug) 

    img_nr = args.start_index

    for i in range(1,5):
        for parent, _, files in os.walk(args.img_dir_src):
            for file in files:
                
                count = 0

                while count < need_aug_num:
                    img_path = os.path.join(parent, file)
                    xml_path = os.path.join(args.xml_dir_src, file[:-4]+'.xml')
                    img = cv2.imread(img_path)
                    bboxes = parse_xml(xml_path)

                    auged_img, auged_bboxes = data_aug.run(img, bboxes)
                    count += 1

                    image_name = str(format(img_nr, '06d'))

                    cv2.imwrite(img_output_path + '/' + image_name + '.jpg', auged_img)

                    height, width, _ = auged_img.shape

                    # write in xml file
                    xml_file = open((xml_output_path + '/' + image_name + '.xml'), 'w')
                    xml_file.write('<annotation>\n')
                    xml_file.write('    <folder>' + dataset_name + '</folder>\n')
                    xml_file.write('    <filename>' + image_name + '.jpg' + '</filename>\n')
                    xml_file.write('    <size>\n')
                    xml_file.write('        <width>' + str(width) + '</width>\n')
                    xml_file.write('        <height>' + str(height) + '</height>\n')
                    xml_file.write('        <depth>3</depth>\n')
                    xml_file.write('    </size>\n')

                    for each_bbox in auged_bboxes:

                        # check the new bboxs
                        if int(each_bbox[0]) < 0:
                            each_bbox[0] = 0
                        if int(each_bbox[1]) < 0:
                            each_bbox[1] = 0
                        if int(each_bbox[2]) > width:
                            each_bbox[0] = width - 1
                        if int(each_bbox[3]) > height:
                            each_bbox[0] = height - 1

                        xml_file.write('    <object>\n')
                        xml_file.write('        <name>' + each_bbox[4] + '</name>\n')
                        xml_file.write('        <pose>Unspecified</pose>\n')
                        xml_file.write('        <truncated>0</truncated>\n')
                        xml_file.write('        <difficult>0</difficult>\n')
                        xml_file.write('        <bndbox>\n')
                        xml_file.write('            <xmin>' + str(int(each_bbox[0])) + '</xmin>\n')
                        xml_file.write('            <ymin>' + str(int(each_bbox[1])) + '</ymin>\n')
                        xml_file.write('            <xmax>' + str(int(each_bbox[2])) + '</xmax>\n')
                        xml_file.write('            <ymax>' + str(int(each_bbox[3])) + '</ymax>\n')
                        xml_file.write('        </bndbox>\n')
                        xml_file.write('    </object>\n')

                    xml_file.write('</annotation>')

                    img_nr += 1

                    print('Data augmentation done with new ' + image_name + '.jpg and ' + image_name + '.xml')

    print('Well, all done!')

def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir_src', default = '/home/robin/datasets/VOC2007/JPEGImages')
    parser.add_argument('--xml_dir_src', default = '/home/robin/datasets/VOC2007/Annotations')
    parser.add_argument('--img_dir_aug', default = '/home/robin/datasets/VOC2007/JPEGImages_aug')
    parser.add_argument('--xml_dir_aug', default = '/home/robin/datasets/VOC2007/Annotations_aug')
    parser.add_argument('--dataset_name', default = 'VOC2007')
    parser.add_argument('--start_index', default = 0)
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())