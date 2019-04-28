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

#### Useful tool for labeling

- [LabelImg](https://github.com/tzutalin/labelImg) in Ubuntu
- [RectLabel](https://rectlabel.com/) in Mac OS
