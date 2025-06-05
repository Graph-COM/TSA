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

All datasets are saved in `./data/` directory. The `CSBM` and `Arxiv` datasets atasets will be loaded automatically at runtime. The raw data for `DBLP_ACM`, `MAG`, and `Pileup` should be downloaded manually from the following sources:

* `DBLP_ACM`: [Google Drive](https://drive.google.com/file/d/1DzQ3QN9yjQxU4vtYkXyCiJKFw7oCCPSM/view). We follow the preprocessing procedure adopted from [UDAGCN](https://github.com/TrustAGI-Lab/UDAGCN).
* `MAG`: [Zenodo](https://zenodo.org/records/10681285). We follow the preprocessing procedure adopted from [PairAlign](https://github.com/Graph-COM/Pair-Align).
* `Pileup`: [Zenodo](https://zenodo.org/records/8015774). We follow the preprocessing procedure adopted from [StruRW](https://github.com/Graph-COM/StruRW).


Place the downloaded files according to the directory structure shown below. The preprocessed data, including dataset splitting, will be generated automatically after the first run:

```
data/
├── DBLP_ACM/
│ ├── acm/
│ │ └── raw/
│ │ ├── acm_docs.txt
│ │ ├── acm_edgelist.txt
│ │ └── acm_labels.txt
│ ├── dblp/
│ │ └── raw/
│ │ ├── dblp_docs.txt
│ │ ├── dblp_edgelist.txt
│ │ └── dblp_labels.txt
│
├── MAG/
│ └── raw/
│ ├── CN_labels_20.pt
│ ├── DE_labels_20.pt
│ ├── FR_labels_20.pt
│ ├── JP_labels_20.pt
│ ├── RU_labels_20.pt
│ ├── US_labels_20.pt
│ ├── label_stat.csv
│ └── papers.csv
│
├── Pileup/
│ └── raw/
│ ├── test_gg_PU10.root
│ ├── test_gg_PU30.root
│ ├── test_gg_PU50.root
│ ├── test_gg_PU140.root
│ ├── test_qq_PU10.root
│ └── test_qq_PU30.root
```
