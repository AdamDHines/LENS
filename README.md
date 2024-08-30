# :eye: LENS - Locational Encoding with Neuromorphic Systems
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![QUT Centre for Robotics](https://img.shields.io/badge/collection-QUT%20Robotics-%23043d71?style=flat-square)](https://qcr.ai)
[![stars](https://img.shields.io/github/stars/AdamDHines/LENS.svg?style=flat-square)](https://github.com/AdamDHines/LENS/stargazers)
[![Downloads](https://static.pepy.tech/badge/lens-vpr)](https://pepy.tech/project/lens-vpr)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/lens-vpr.svg)](https://anaconda.org/conda-forge/lens-vpr)
![PyPI - Version](https://img.shields.io/pypi/v/lens-vpr)
[![GitHub repo size](https://img.shields.io/github/repo-size/AdamDHines/LENS.svg?style=flat-square)](./README.md)

This repository contains code for **LENS** - **L**ocational **E**ncoding with **N**euromorphic **S**ystems. LENS combines neuromorphic algoriths, sensors, and hardware to perform accurate, real-time robotic localization using visual place recognition (VPR). LENS can be used with the SynSense Speck2fDevKit board which houses a [SPECK<sup>TM</sup>](https://www.synsense.ai/products/speck-2/) dynamic vision sensor and neuromorphic processor for online VPR.

## License and citation
This repository is licensed under the [MIT License](./LICENSE). If you use our code, please cite our arXiv paper:

```
@misc{hines2024lens,
      title={A compact neuromorphic system for ultra energy-efficient, on-device robot localization}, 
      author={Adam D. Hines and Michael Milford and Tobias Fischer},
      year={2024},
      eprint={2408.16754},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2408.16754}, 
}
```


## Installation and setup
To run LENS, please download this repository and install the required dependencies.

### Get the code
Get the code by cloning the repository.
```console
git clone git@github.com:AdamDHines/LENS.git
cd ~/LENS
```

### Install dependencies
All dependencies can be instlled from our [conda-forge package](https://anaconda.org/conda-forge/lens-vpr), [PyPi package](https://pypi.org/project/lens-vpr/), or local `requirements.txt`. For the conda-forge package, we recommend using [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) or [miniforge](https://github.com/conda-forge/miniforge). Please ensure your Python version is <= 3.11.

#### conda package
```console
# Create a new environment and install packages
micromamba create -n lens-vpr -c conda-forge lens-vpr

# samna package is not available on conda-forge, so pip install it
micromamba activate lens-vpr
pip install samna
```

#### pip
```console
# Install from our PyPi package
pip install lens-vpr

# Install from local requirements.txt
pip install -r requirements.txt
```

## Quick start
Get started using our pretrained models and datasets to evaluate the system. For a full guide on training and evaluating your own datasets, please visit our [Wiki](https://github.com/AdamDHines/LENS/wiki).

### Run the inferencing model
To run a simulated event stream, you can try our pre-trained model and datasets. Using the `--sim_mat` and `--matching` flag will display a similarity matrix and perform Recall@N matching based on a ground truth matrix.

```console
python main.py --sim_mat --matching
```

### Train a new model
New models can be trained by parsing the `--train_model` flag. Try training a new model with our provided reference dataset.

```console
# Train a new model
python main.py --train_model
```

### Optimize network hyperparameters
For new models on custom datasets, you can optimize your network hyperparameters using [Weights & Biases](https://wandb.ai/site) through our convenient `optimizer.py` script.

```console
# Optimize network hyperparameters
python optimizer.py
```

For more details, please visit the [Wiki](https://github.com/AdamDHines/LENS/wiki/Setting-up-and-using-the-optimizer).

### Deployment on neuromoprhic hardware
If you have a SynSense Speck2fDevKit, you can try out LENS using our pre-trained model and datasets by deploying simulated event streams on-chip.

```console
# Generate a timebased simulation of event streams with pre-recorded data
python main.py --simulated_speck --sim_mat --matching
```

Additionally, models can be deployed onto the Speck2fDevKit for low-latency and energy efficient VPR with sequence matching in real-time. Use the `--event_driven` flag to start the online inferencing system.

```console
# Run the online inferencing model
python main.py --event_driven
```

For more details on deployment to the Speck2fDevKit, please visit the [Wiki](https://github.com/AdamDHines/LENS/wiki/Deploying-to-Speck2fDevKit).


## Issues, bugs, and feature requests
If you encounter problems whilst running the code or if you have a suggestion for a feature or improvement, please report it as an [issue](https://github.com/AdamDHines/VPRTempoNeuro/issues).
