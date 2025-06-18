<p align="center">
  <img src="./assets/logo.png" alt="LENS Logo" width="600"/>
</p>

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
[![Documentation Status](https://readthedocs.org/projects/lens-vpr/badge/?version=latest&style=flat)](https://lens-vpr.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![QUT Centre for Robotics](https://img.shields.io/badge/collection-QUT%20Robotics-%23043d71?style=flat-square)](https://qcr.ai)
[![stars](https://img.shields.io/github/stars/AdamDHines/LENS.svg?style=flat-square)](https://github.com/AdamDHines/LENS/stargazers)
[![Downloads](https://static.pepy.tech/badge/lens-vpr?style=flat-square)](https://pepy.tech/project/lens-vpr)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/lens-vpr.svg?style=flat-square)](https://anaconda.org/conda-forge/lens-vpr)
![PyPI - Version](https://img.shields.io/pypi/v/lens-vpr?style=flat-square)
[![GitHub repo size](https://img.shields.io/github/repo-size/AdamDHines/LENS.svg?style=flat-square)](./README.md)

This repository contains code for **LENS** - **L**ocational **E**ncoding with **N**euromorphic **S**ystems. LENS combines neuromorphic algorithms, sensors, and hardware to perform accurate, real-time robotic localization using visual place recognition (VPR). 

LENS performs VPR with the SynSense [Speck<sup>TM</sup>](https://www.synsense.ai/products/speck-2/) development kits, featuring a combination of a dynamic vision sensor and neuromorphic System-on-Chip processor for real-time, energy-efficient localization. 

LENS can also be used with conventional CPU, GPU, and Apple Silicon (MPS) devices to perform event-based VPR thanks to the [Sinabs](https://sinabs.readthedocs.io/en/v2.0.0/) spiking network architecture.

_For more information, please visit the [LENS Documentation](https://lens-vpr.readthedocs.io/en/latest/)_.

## Getting started
For reproducibility and simplicity, we use [pixi](https://prefix.dev/) for package management and installation. If not already installed, please run the following command in your terminal:

```console
curl -fsSL https://pixi.sh/install.sh | bash
```

_You will be prompted to restart your terminal once installed. For more information, please refer to the [pixi documentation](https://pixi.sh/latest/)._ 

Run the following in your terminal to clone the LENS repository and navigate to the project directory:
```console
git clone git@github.com:AdamDHines/LENS.git
cd ~/LENS
```

_For alternative package and dependency installation, please see the [LENS documentation](https://lens-vpr.readthedocs.io/en/latest/installation.html#conda)._

## Quick demo
Get started using our demo dataset and pre-trained model to evaluate the system. Run the following in your command terminal to see the demo:

```console
pixi run demo
```

### Train and evaluate new model
Test out training and evaluating a new model with our ultra-fast learning method using our provided demo dataset by running the following in your command terminal:

```console
pixi run train
pixi run evaluate
```



_For a full guide on training and evaluating your own datasets, please visit the [LENS documentation](https://lens-vpr.readthedocs.io/en/latest/train_setup.html)._

### Optimize network hyperparameters
To get the best localization performance on benchmark or custom datasets, you can tune your network hyperparameters using [Weights & Biases](https://wandb.ai/site) through our convenient optimizer script: 

```console
pixi run optimizer
```

_For detailed instructions on setting up Weights & Biases and the optimizer, please refer to the [LENS documentation](https://lens-vpr.readthedocs.io/en/latest/optimizer_setup.html)._

### Deployment on neuromorphic hardware
LENS was developed using a SynSense Speck2fDevKit. If you have one of these kits, deploying to it is simple. Try out LENS using our pre-trained model and datasets by deploying simulated event streams on-chip:

```console
pixi run sim-speck
```

Additionally, models can be deployed onto the Speck2fDevKit for low-latency and energy efficient VPR with sequence matching in real-time:
```console
pixi run on-speck
```

_For more details on deployment to the Speck2fDevKit, please visit the [LENS documentation](https://lens-vpr.readthedocs.io/en/latest/sp_overview.html)._

## Dataset
For all data relating to our manuscript, we have a dedicated permanent repository at https://zenodo.org/records/15392412, as well as including all data in this repository, which can found in the [./lens/data](./lens/data) folder.

We acknowledge the Brisbane-Event-VPR dataset from https://zenodo.org/records/4302805.

## License and citation
This repository is licensed under the permissive [MIT License](./LICENSE). If you use our code, please cite our [paper](https://www.science.org/doi/10.1126/scirobotics.ads3968):

```
@article{HinesLENS2025,
  author = {Adam D. Hines  and Michael Milford  and Tobias Fischer },
  title = {A compact neuromorphic system for ultra–energy-efficient, on-device robot localization},
  journal = {Science Robotics},
  volume = {10},
  number = {103},
  pages = {eads3968},
  year = {2025},
  doi = {10.1126/scirobotics.ads3968},
  URL = {https://www.science.org/doi/abs/10.1126/scirobotics.ads3968}
}
```

## Issues, bugs, and feature requests
If you encounter problems whilst running the code or if you have a suggestion for a feature or improvement, please report it as an [issue](https://github.com/AdamDHines/VPRTempoNeuro/issues).
