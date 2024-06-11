# LENS - Locational Encoding with Neuromorphic Systems
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![stars](https://img.shields.io/github/stars/AdamDHines/LENS.svg?style=flat-square)](https://github.com/AdamDHines/LENS/stargazers)
[![QUT Centre for Robotics](https://img.shields.io/badge/collection-QUT%20Robotics-%23043d71?style=flat-square)](https://qcr.ai)
![GitHub repo size](https://img.shields.io/github/repo-size/AdamDHines/LENS.svg?style=flat-square)

This repository contains code for **LENS** - **L**ocational **E**ncoding with **N**euromorphic **S**ystems. LENS combines neuromorphic algoriths, sensors, and hardware to perform accurate, real-time robotic localization using visual place recognition. LENS uses the [SPECK<sup>tm</sup>](https://www.synsense.ai/products/speck-2/) dynamic vision sensor and neuromorphic processor to learn places and perform online recognition, deployed on a hexapod robotic platform.

## License & Citation
This repository is licensed under the [MIT License](./LICENSE)

## Installation and setup
Project is neither hosted on PyPi or conda-forge (yet, will be done when repository is made public). Simplest way is to directly install the dependencies with `pip` or `conda/mamba`.

```console
pip install python torch torchvision numpy pandas tqdm prettytable scikit-learn matplotlib sinabs samna sinabs-dynapcnn opencv-python

conda create -n vprtemponeuro -c pytorch python torch torchvision numpy pandas tqdm prettytable scikit-learn matplotlib sinabs samna sinabs-dynapcnn opencv-python
```

### Get the repository
Activate the environment & download the Github repository
```console
conda activate vprtemponeuro
git clone https://github.com/AdamDHines/VPRTempoNeuro.git
cd ~/VPRTempoNeuro
```

## Datasets
At the moment, using pre-recorded datasets (see Tobi and Gokul) - likely will be using either QCR Event or Brisbane Event datasets for the initial figures and testing of the system.

Using dvs_tools.py is cool and will make nice videos. Test.

## Issues, bugs, and feature requests
If you encounter problems whilst running the code or if you have a suggestion for a feature or improvement, please report it as an [issue](https://github.com/AdamDHines/VPRTempoNeuro/issues).
