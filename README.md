# PEM Rare-Event Sampling in CARLA

Code for associated paper: [Testing Rare Downstream Safety Violations via Upstream Adaptive Sampling of Perception Error Models](https://arxiv.org/abs/2209.09674).

It simulates automated braking scenario where a car is following another one in front, and then the other car brakes at red light, we must brake to avoid crashing. Uses obstacle detector based on the yolo architecture trained on the kitti dataset (picture below?). Safety specification is written in STL (see the code for this), with 

Description of what the heck this thing is here

## Setup

### Installations
- Install and run the CARLA simulator. Instructions [here](https://carla.readthedocs.io/en/latest/start_quickstart/)
- Install python package pre-requisites using `python -m pip install -r requirements.txt`

### Pre-trained models

Ok so this is actually a little bit tricky! Can be found in models/det_baseline_full/pem_class_train_full for the ML-NN one used in the paper (probably want to rename?). However, if you wanna train from scratch...probably need to import some of the PyYoloKITTI PEM Stuff? Can we get the training regime stuff in here? Actually...this is probably a waste of time to train up. Just reference the PyTorch YOLO-KITTI work? Should be enough. Also Pyro etc.

## Running

### Running Simulations

Want to be able to do `runSim.py <n-stages> <n-sims> <timesteps> <pem> <metric> <exp-name>`

The descriptions of the arguments are:

- __n-stages__: The number of adaptive simulation stages to run in cross-entropy importance sampling. Corresponds to the $K$ parameter in paper.
- __n-sims__:
- __timesteps__:
- __pem__:
- __metric__:
- __exp-name__:
- __render__:

If run with render=True, then you should see the following experiments in your CARLA server window:

**LITTLE GIF HERE**

### Running Analysis
