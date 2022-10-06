# LLT
Pytorch implementation of "Learnable Lookup Table for Neural Network Quantization", CVPR 2022

[[CVF]](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Learnable_Lookup_Table_for_Neural_Network_Quantization_CVPR_2022_paper.pdf) [[Supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Wang_Learnable_Lookup_Table_CVPR_2022_supplemental.pdf)


## Overview

<p align="center"> <img src="Figs/overview.png" width="70%"> </p>

## Image Classification (CIFAR-10)

### Requirements
- PyTorch
- TorchVision
- numpy

### Train
```bash
python train.py --arch resnet20 --epochs 200 --batch_size 128 --learning_rate 0.01 --weight_decay 1e-4 --w_bits 4 --a_bits 4
```

### Test
```bash
python test.py --arch resnet20 --batch_size 128 --w_bits 4 --a_bits 4
```

### Results

<p align="center"> <img src="Figs/cifar10.png" width="100%"> </p>

## Image Super-Resolution

### Requirements
- PyTorch == 1.1.0
- numpy
- skimage
- imageio
- cv2

## Train
### 1. Prepare training data 

1.1 Download DIV2K training data (800 training + 100 validtion images) from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) or [SNU_CVLab](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar).

1.2 Specify '--dir_data' based on the HR and LR images path. In option.py, '--ext' is set as 'sep_reset', which first convert .png to .npy. If all the training images (.png) are converted to .npy files, then set '--ext sep' to skip converting files.

For more informaiton, please refer to [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch).

### 2. Begin to train
```bash
python main.py --model EDSR --scale 4 --w_bits 4 --a_bits 4 --save EDSR_w4a4 --pre_train model/EDSR/EDSR_x4.pth --patch_size 48 --batch_size 12
```

## Test
### 1. Prepare test data 
Download [benchmark datasets](https://github.com/xinntao/BasicSR/blob/a19aac61b277f64be050cef7fe578a121d944a0e/docs/Datasets.md) (e.g., Set5, Set14 and other test sets) and prepare HR/LR images in `testsets/benchmark` following the example of `testsets/benchmark/Set5`.


### 2. Begin to test
```bash
python main.py --dir_data testsets --data_test Set5 --model EDSR --scale 4 --w_bits 4 --a_bits 4 --pre_train experiment/EDSR_w4a4/model/model_40.pt --test_only --save_results
```

### Results

<p align="center"> <img src="Figs/sr.png" width="100%"> </p>

<p align="center"> <img src="Figs/sr_visual.png" width="100%"> </p>


## Point Cloud Classification

### Train & Test

To be updated

### Results

<p align="center"> <img src="Figs/point.png" width="50%"> </p>


## Citation
```
@InProceedings{Wang2022Learnable,
  author    = {Wang, Longguang and Dong, Xiaoyu and Wang, Yingqian and Liu, Li and An, Wei and Guo, Yulan},
  title     = {Learnable Lookup Table for Neural Network Quantization},
  booktitle = {CVPR},
  year      = {2022},
  pages     = {12423--12433},
}
```

## Acknowledgements
Part of the code is borrowed from [APot](https://github.com/yhhhli/APoT_Quantization). We thank the authors for sharing the codes.

