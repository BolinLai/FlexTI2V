# Incorporating Flexible Image Conditioning into Text-to-Video Diffusion Models without Training


### [Project Page](https://bolinlai.github.io/projects/FlexTI2V/) | [Paper](http://arxiv.org/pdf/2505.20629)

#### [Bolin Lai](https://bolinlai.github.io/), [Sangmin Lee](https://sites.google.com/view/sangmin-lee), [Xu Cao](https://www.irohxucao.com/), [Xiang Li](https://ryanxli.github.io/), [James M. Rehg](https://rehg.org/)


<img src="https://bolinlai.github.io/projects/FlexTI2V/figures/teaser.png"/>


### TODO (Actively Updating...)
- [ ] Code for inference
- [ ] Diffusion code
- [x] Example images
- [ ] Add more instructions
- [ ] Dataset
- [ ] Evaluation


## Contents

- [Problem Formulation](#problem-formulation)
- [Setup](#setup)
- [Run](#run)
- [BibTex](#bibtex)
- [Acknowledge](#acknowledgement)


## Problem Formulation

<img src="https://bolinlai.github.io/projects/FlexTI2V/figures/tasks.png"/>

Comparison with classic TI2V tasks. Our task requires video generation conditioned on any number of images at any positions, which unifies existing classic TI2V tasks. The images with blue and pink edges are condition images, and images with green edges are generated video frames.


## Setup

### Environment

```shell
pip install -r requirements.txt
```

### Dataset

### Pre-trained Checkpoints


## Run


## BibTex

If you find our paper helpful to your work, please cite with this BibTex.

```BibTex
@article{lai2025incorporating,
  title={Incorporating Flexible Image Conditioning into Text-to-Video Diffusion Models without Training},
  author={Lai, Bolin and Lee, Sangmin and Cao, Xu and Li, Xiang and Rehg, James M},
  journal={arXiv preprint arXiv:2505.20629},
  year={2025}
}
```


## Acknowledgement
