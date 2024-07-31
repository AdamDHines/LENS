# :eye: LENS - Locational Encoding with Neuromorphic Systems
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
@arxiv{hines202xlens,
      title={Robotic localization and navigation using a compact neuromorphic ecosystem}, 
      author={Adam D. Hines and Michael Milford and Tobias Fischer},
      year={202x},
      booktitle={}
      
}
```

## Installation and setup
All that is needed to run LENS is to download the code and install the dependencies.

### Get the code
Get the code by cloning the repository.
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

## Usage
LENS was developed with and deployed on the Speck2fDevKit from [SynSense](https://www.synsense.ai/). However, it is not required to have a SPECK<sup>TM</sup> to be use LENS. We provide a general framework for training and inferencing data from a dynamic vision sensor for visual place recognition.

Please [click here]() to see the full documentation, which describes in more detail the full functionality of LENS.

### Run the inferencing model
To run a simulated event stream, you can try our pre-trained model and datasets. Parse the `--sim_mat` and `--matching` flag to see the similarity matrix and matching to the provided ground truth.

```console
$ python main.py --sim_mat --matching
```

### Train a new model
To train your own model, you first will need to collect your own dataset (see [Data Collection](##data-collection) for SPECK<sup>TM</sup> or [Datasets]() for event data sourced elsewhere). 

```console
# Train a new model
$ python main.py --train_model --reference dataset001 --reference_places -80
```

`--reference` is the name of the dataset you're training and `--reference_places` is the number of images in the dataset you wish to train. This trained model can then be evaluated on any of the above inferencing methods.

For a full reference on training your own model, please see the [LENS Documentation]().

### Deployment on Speck2fDevKit
If you have a Speck2fDevKit, you can try out LENS using our pre-trained model and datasets conveniently provided in this repository.

```console
# Generate a timebased simulation of event streams with pre-recorded data
$ python main.py --simulated_speck --sim_mat
```

`--simulated_speck` will take pre-recorded data from the SPECK<sup>TM</sup>'s built in dynamic vision sensor and create a timebased simulation of event spikes which are sent to the neuromorphic processor to perform visual place recognition. `--sim_mat` generates a similarity matrix between the database reference and incoming queries.

If you have collected data and trained your own model (see below, [Data Collection](##data-collection) & [Training a new model](###training-a-new-model), then you can evaluate the model with online inferencing by running the following.

```console
# Run the online inferencing model
$ python main.py --event_driven
```

This will open up a `samnagui` instance whereby you can see the events collected by SPECK<sup>TM</sup> after passing through an event pre-processing layer and the power consumption over time. On each `--timebin` iteration, the model will return the predicted place. Power consumption and output spikes is output as a `.npy` file in the `./lens/outputs/` subfolder. 

For limitations and requiremnts for training and inferencing models, please see [Limitations](##limitations).

## Data Collection
If using a Speck2fDevKit, you can collect event data and produce temporal frame representations as images.

```console
$ python main.py --collect_data --data_name experiment001
```

This will export to the `./lens/dataset/<data>/<camera>/` folder with .png images which you can then train a new model on. A .csv file of for the image names will be automatically generated.

## Limitations
For deployment on SPECK<sup>TM</sup>, models must be no bigger than 144kB in total (see [Memory Constraints and Network Sizing](https://sinabs.readthedocs.io/en/v2.0.0/speck/overview.html#memory-constraints-and-network-sizing). If customizing the network training for on-chip deployment, input size, feature layer size multiplier, and number of learned places must be carefully considered. Generally speaking, a neuron model size of 100 input, 200 feature, and 80 output neurons is roughly the maximum sizing. These constraints do not impact off-chip deployment.

See more at the [LENS Documentation]().

## Issues, bugs, and feature requests
If you encounter problems whilst running the code or if you have a suggestion for a feature or improvement, please report it as an [issue](https://github.com/AdamDHines/VPRTempoNeuro/issues).
