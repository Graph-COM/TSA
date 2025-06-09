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
│ │ │ └── acm_docs.txt
│ │ │ ├── acm_edgelist.txt
│ │ │ └── acm_labels.txt
│ ├── dblp/
│ │ └── raw/
│ │ │ └── dblp_docs.txt
│ │ │ ├── dblp_edgelist.txt
│ │ │ └── dblp_labels.txt
│
├── MAG/
│ └── raw/
│ │ └── CN_labels_20.pt
│ │ ├── DE_labels_20.pt
│ │ ├── FR_labels_20.pt
│ │ ├── JP_labels_20.pt
│ │ ├── RU_labels_20.pt
│ │ ├── US_labels_20.pt
│ │ ├── label_stat.csv
│ │ └── papers.csv
│
├── Pileup/
│ └── raw/
│ │ └──test_gg_PU10.root
│ │ ├── test_gg_PU30.root
│ │ ├── test_gg_PU50.root
│ │ ├── test_gg_PU140.root
│ │ ├── test_qq_PU10.root
│ │ └── test_qq_PU30.root
```

## Usage

```
python src/main.py data= <DATASET> adapter=<METHOD> model=<MODEL> [Options]

```
* `adapter`: TSA variants include `TSA_T3A`, `TSA_LAME`, and `TSA_TENT`.
* `model`: We include `GSN`, `GPRGNN`, and `GCN`.
* `data`: `CSBM`, `MAG`, `Pileup`, `Arxiv`, and `DA` (DBLP_ACM). We use the number to indicate the column index in the corresponding dataset’s result table in the paper. For example, `MAG1` refers to US ➝ CN and `MAG2` refers to US ➝ DE.


**Example of CSBM dataset:**

Run `TSA_T3A` under `CSS` with `GSN` backbone.

```
python src/main.py data=CSBM1 adapter=TSA_T3A model=GSN model_config.gnn_dim=20 model_config.cls_dim=20 adapter_config.filter_K=20 adapter_config.scale_lr=0.1 adapter_config.pa_ratio=1.0 adapter_config.scale_thre=1.0
```

**Example of MAG dataset:**

Run `TSA_T3A` under shift from `US ➝ CN` with `GSN` backbone.

```
python src/main.py data=MAG1 adapter=TSA_T3A model=GSN model_config.gnn_dim=300 model_config.cls_dim=300 adapter_config.filter_K=20 adapter_config.scale_lr=0.05 adapter_config.pa_ratio=0.5 adapter_config.scale_thre=1.0 
```

**Example of Pileup dataset:**

Run `TSA_T3A` under shift from `PU30 ➝ PU10` with `GSN` backbone.

```
python src/main.py data=Pileup2 adapter=TSA_T3A model=GSN model_config.gnn_dim=50 model_config.cls_dim=50 adapter_config.filter_K=20  adapter_config.pa_ratio=0.5 adapter_config.scale_lr=0.1 adapter_config.scale_thre=0.1
```

**Example of Arxiv dataset:**

Run `TSA_LAME` under shift from `1950-2007 ➝ 2016-2018` with `GSN` backbone.

```
python src/main.py data=Arxiv2 adapter=TSA_LAME model=GSN model_config.gnn_dim=300 model_config.cls_dim=300 adapter_config.pa_ratio=0.01 adapter_config.scale_lr=0.001 adapter_config.scale_thre=0.1
```

**Example of DBLP_ACM dataset:**

Run `TSA_T3A` under shift from `DBLP ➝ ACM` with `GPRGNN` backbone.

```
python src/main.py data=DA1 adapter=TSA_T3A model=GPRGNN model_config.gnn_dim=128 model_config.cls_dim=40 adapter_config.filter_K=20 adapter_config.scale_lr=0.1 adapter_config.pa_ratio=0.5 adapter_config.scale_thre=1.0
```

For detailed hyperparameters, please see `configs/adapter/`.

## Graph Test-Time Adaptation with other Baselines

We implemented multiple baseline methods in `src/adaptation/`. To run the baseline methods, set the argument `adapter` to the corresponding values. This include:

* **Graph TTA Methods**: `GTrans` ([Paper](https://arxiv.org/abs/2210.03561)), `SOGA` ([Paper](https://arxiv.org/abs/2112.00955)), `HomoTTT` ([Paper](https://dl.acm.org/doi/10.1145/3649507)), `Matcha-T3A`, `Matcha-LAME`, and `Matcha-TENT` ([Paper](https://arxiv.org/abs/2410.06976)).

* **Non-Graph TTA Methods**: `ActMAD` ([Paper](https://arxiv.org/abs/2211.12870)), `T3A` ([Paper](https://proceedings.neurips.cc/paper/2021/hash/1415fe9fea0fa1e45dddcff5682239a0-Abstract.html)), `LAME` ([Paper](https://arxiv.org/abs/2201.05718)), and `TENT`([Paper](https://arxiv.org/abs/2006.10726)).

For detailed hyperparameters, please see `configs/adapter/`.
