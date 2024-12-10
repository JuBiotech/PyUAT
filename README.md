# Python Uncertainty-Aware Tracking (PyUAT)

![pipeline](https://jugit.fz-juelich.de/IBG-1/ModSim/imageanalysis/uat/badges/main/pipeline.svg)
![coverage](https://jugit.fz-juelich.de/IBG-1/ModSim/imageanalysis/uat/badges/main/coverage.svg?job=coverage)



Efficient open-source Python implementation of the Uncertainty-Aware Tracking approach: [https://doi.org/10.1093/bioinformatics/bty776](https://doi.org/10.1093/bioinformatics/bty776)


![Tracking gif](https://github.com/JuBiotech/PyUAT/blob/images/tracked.gif)

## Usage

You can either use our colab notebooks for examples for cell tracking or install and run the examples on your computer locally. We recommend the usage of Linux ubuntu for a local installation.

### Basic Example

<a target="_blank" href="https://colab.research.google.com/github/JuBiotech/PyUAT/blob/main/example_simple.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

We provide a basic example for using the tracking configurations described in our paper.

### Customize cell models

<a target="_blank" href="https://colab.research.google.com/github/JuBiotech/PyUAT/blob/main/example_simple_custom.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

We provide an example showing the integration of a new custom model (in that case cell age) into the tracking configuration.

## Local Installation

Install PyUAT from pypi

```bash
pip install uatrack
```

## Developer Installation

```bash
git clone https://github.com/JuBiotech/PyUAT
cd PyUAT
pip install -e .
```

## Local usage

Try our [`example_simple.ipynb`](example_simple.ipynb) notebook to get started with the tracking.


## Data Availability

We utilize the publicly available `Tracking one-in-a-million` dataset introduced by [Seiffarth et al.](https://doi.org/10.48550/arXiv.2411.00552) at the ECCV 2024.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7260137.svg)](https://doi.org/10.5281/zenodo.7260137)
