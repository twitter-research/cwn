# Cell Complex Networks

This repository contains the code used in the experimental section of the NeurIPS 2021 submission:
> Weisfeiler and Lehman Go Cellular: CW Networks

## Installation

We use `Python 3.8` and `PyTorch 1.7.0` on `CUDA 10.2` for this project.
Please follow these steps to prepare the virtual environment needed to run any experiment.

Create the environment:
```
conda create --name scn_102 python=3.8
conda activate scn_102
```

Install torch:
```
conda install -y pytorch=1.7.0 torchvision cudatoolkit=10.2 -c pytorch
```

Install torch-geometric:
```
sh pyG_install.sh cu102
```

Install other required packages via pip:
```
pip install -r requirements.txt
```

Install graph-tool via conda:
```
sh graph-tool_install.sh
```

At this point you should be good to go.