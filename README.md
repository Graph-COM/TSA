# TSA
Source code of "Structural Alignment Improves Graph Test-Time Adaptation"

## How TSA tackles test-time distribution shifts?

**1. Neighborhood Alignment** Recalibrate the influence of neighboring nodes during message aggregation to address *conditional structure shift* (CSS).

**2. SNR-inspired Adjustment**: Optimize the test-time combination of self-node representations and neighborhood-aggregated representations based on the *signal-to-noise ratio* (SNR).

**3. Decision Boundary Refinement**: Mitigate mismatches caused by *label* and *feature* shifts.

![TSA image](https://github.com/Graph-COM/TSA/blob/main/images/tsa.png?raw=true)

## Installation

The code is based on Python 3.10 and Cuda 12.1. We recommend to setup the enviornment using `conda`. After cloning the repository, run the following command to create the `tsa` environment.

```
cd TSA
conda env create -f env.yaml
conda activate tsa
```

## Datasets
