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

## How to Run

### Running Simulations

To run the adaptive importance sampling experiment for the automated braking scenario from the paper, run the command:

`runSim.py <n-stages> <n-sims> <timesteps> <pem> <metric> <exp-name>`

Here are a description of the arguments:

- __n-stages__: The number of adaptive simulation stages to run in cross-entropy importance sampling. Corresponds to the $K$ parameter in paper.
- __n-sims__: The number of simulation rollouts sampled per adaptive stage. Corresponds to $N_{\kappa}$ parameter in paper.
- __timesteps__: Time steps per simulation episode. Parameter $T$ in paper.
- __pem__: Path to pre-trained perception error model. Try `models/det_baseline_full/pem_class_train_full`
- __metric__: STL Robustness metric used for evaluation. Choices are `["classic", "agm", "smooth-cumulative"]`
- __exp-name__: Name given to experiment. Simulation rollout logs will be saved to `sim_data/{exp-name}-{metric}-K{n-stages}-e{n-sims}-t{timesteps}/<current-timestamp>`, and learned proposal samplers will be saved to `models/CEMs/{exp-name}-{metric}-K{n-stages}-e{n-sims}-t{timesteps}/<current-timestamp>`
- __render__: True/False flag of whether to render simulations or not.

If run with render=True, then you should see the following experiments in your CARLA server window:

![Automated Braking Simulation](images/samplingExperimentFollow.gif)
### Running Analysis


