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

### Train & Test

To be updated

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

