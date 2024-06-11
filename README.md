# LENS - Locational Encoding with Neuromorphic Systems
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![stars](https://img.shields.io/github/stars/AdamDHines/LENS.svg?style=flat-square)](https://github.com/AdamDHines/LENS/stargazers)
[![QUT Centre for Robotics](https://img.shields.io/badge/collection-QUT%20Robotics-%23043d71?style=flat-square)](https://qcr.ai)
![GitHub repo size](https://img.shields.io/github/repo-size/AdamDHines/LENS.svg?style=flat-square)

This repository contains code for **LENS** - **L**ocational **E**ncoding with **N**euromorphic **S**ystems. LENS combines neuromorphic algoriths, sensors, and hardware to perform accurate, real-time robotic localization using visual place recognition. LENS uses the [SPECK<sup>TM</sup>](https://www.synsense.ai/products/speck-2/) dynamic vision sensor and neuromorphic processor to learn places and perform online recognition, deployed on a hexapod robotic platform.

## License & Citation
This repository is licensed under the [MIT License](./LICENSE)

If you use our code, please cite our [paper]():
```
@inproceedings{hines202xlens,
      title={Robotic localization and navigation using a compact neuromorphic ecosystem}, 
      author={Adam D. Hines and Michael Milford and Tobias Fischer},
      year={202x},
      booktitle={}
      
}
```

## Installation and setup
All that is needed to run LENS is to download the code and install the dependencies.

### Get the code
Get the code by cloning the repository or downloading a [.zip](https://docs.github.com/en/repositories/working-with-files/using-files/downloading-source-code-archives#downloading-source-code-archives-from-the-repository-view) 

```console
$ git clone git@github.com:AdamDHines/LENS.git
$ cd ~/LENS
```

### Install dependencies
All dependencies can be instlled from our [PyPi package]() or local `requirements.txt` source.

```python
# Install from our PyPi package
> pip install lens

# Install from local requirements.txt
> pip install -r requirements.txt
```


```console
pip install python torch torchvision numpy pandas tqdm prettytable scikit-learn matplotlib sinabs samna sinabs-dynapcnn opencv-python

conda create -n vprtemponeuro -c pytorch python torch torchvision numpy pandas tqdm prettytable scikit-learn matplotlib sinabs samna sinabs-dynapcnn opencv-python
```

### Get the repository
Activate the environment & download the Github repository
```console
conda activate lens
git clone https://github.com/AdamDHines/LENS.git
cd ~/LENS
```

## Datasets
At the moment, using pre-recorded datasets (see Tobi and Gokul) - likely will be using either QCR Event or Brisbane Event datasets for the initial figures and testing of the system.

Using dvs_tools.py is cool and will make nice videos. Test.

## Issues, bugs, and feature requests
If you encounter problems whilst running the code or if you have a suggestion for a feature or improvement, please report it as an [issue](https://github.com/AdamDHines/VPRTempoNeuro/issues).
