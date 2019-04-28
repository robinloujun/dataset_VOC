### Build up the dataset for object detection in the form of VOC

#### Download the raw datasets

```shell
cd $DATAPATH/data
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

tar -xvf VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar
```

#### Content

- Extract the video frames using OpenCV
    ```shell
    python ./video_to_image.py \
    --video_dir [dir to the video] \
    --img_dir [dir to the extracted image files] \
    --freq [frequence] \
    --start_index [start index of the image files]
    ```

- Convert the images from the format png to jpg using OpenCV
    ```shell
    python ./png_to_jpg.py
    ```
    
- Convert the .txt bbox annotation to the format of Pascal VOC 2007
    ```shell
    python ./annotation_converter.py \
    --img_dir [dir to the image files] \
    --txt_dir [dir to the original txt annotation files] \
    --xml_dir [dir to the converted xml annotation files] \
    --dataset_name [name of dataset] \
    --class_name [name of class] \
    --start_index [start index of the image files]
    --enable_rename [boolean to enable the rename option]
    ```

- Data augmentation for object detection dataset
    ```shell
    python ./data_aug.py \
    --img_dir_src [dir to the original image files] \
    --xml_dir_src [dir to the original annotation files] \
    --img_dir_aug [dir to the new image files] \
    --xml_dir_aug [dir to the new annotation files] \
    --dataset_name [name of dataset] \
    --start_index [start index of the image files]
    ```
    
#### Useful tool for labeling

- [LabelImg](https://github.com/tzutalin/labelImg) in Ubuntu
- [RectLabel](https://rectlabel.com/) in Mac OS
