# LocalPer

This repository contains the source code for the paper [A Distribution-based Featurization of Networks
via Persistence diagrams of Local Motifs]().

<!-- [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/hzxie/GRNet.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/hzxie/GRNet/context:python) -->

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-377/)

<!-- ## Cite this work

```
@inproceedings{parthasarathy2020distribution,
  title={A Distribution-based Featurization of Networks via Persistence diagrams of Local Motifs},
  author={Srinivasan Parthasarathy, Minghao Tian, Yusu Wang},
  year={2020}
}
``` -->

## Datasets

We use the [TUDatasets](https://chrsmrrs.github.io/datasets/) for graph classification task

## Prerequisites
#### Clone the Code Repository

```
git clone https://github.com/epsiloncat/LocalPer.git
```

#### Install Python Denpendencies

```
cd LocalPer
pip install -r requirements.txt
```

## Get Started

To cluster synthetic datasets (random graphs) using LocalPer, you can simply use the following command:

```
python localper_clustering_random_graphs.py -S small -N 10 -k 1 -b0 50 -b1 50
```

To cluster synthetic datasets (random graphs) using MMD+SWK, use the following command:

```
python mmd_clustering_random_graphs.py -S small -N 10 -k 1
```

To run a single 10-fold test on a real network dataset in TUDatasets using LocalPer, say IMDB-BINARY, you can use the following command:

```
python localper_single_tenfold_realdata.py -n IMDB-BINARY
```
To run a 10 10-fold test on IMDB-BINARY, you can use the following command:

```
python localper_ten_tenfold_realdata.py -n IMDB-BINARY
```

## License

This project is open sourced under MIT license.