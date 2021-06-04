# Cell Complex Networks

This repository contains the code used in the experimental section of the NeurIPS 2021 submission:
> Weisfeiler and Lehman Go Cellular: CW Networks

## Installation

We use `Python 3.8` and `PyTorch 1.7.0` on `CUDA 10.2` for this project.
Please open a terminal window and follow these steps to prepare the virtual environment needed to run any experiment.

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

At this point you should be good to go. Please remind to always activate the environment before running any of the commads listed in the following.

## Testing

We provide a series of Python unit tests that can be conveniently run by `pytest`.
We suggest running all tests in the repository after installation to verify everything is in place. Simply run:
```
pytest .
```
This command will recursively fetch all unit tests present in the repository and run them. A summary will be printed out in the end. All tests should pass (typically showed in green).

## Experiments on molecular benchmarks

### CIN

In order to run an experiment on a molecular benchmark simply execute:
```
sh exp/<benchmark>.sh
```
where `<benchmark>` with one amongst `zinc`, `zinc-full`, `molhiv`.

These shell scripts will run the `exp/run_mol_exp.py` script passing the required parameters for each experiment.

Internally, for a specified range of random seeds, the script will run the trainings sequentially, compute final performance statistics and print them on screen in the end. Additionally, the script will write these results under `exp/results/<benchmark>/`.

_Note_: before the training starts, the script will download the corresponding graph datasets and perform the appropriate ring-lifting procedure.

### CIN-small

Imposing the parameter budget: in order to run the 'smaller' `CIN` counterparts on these benchmarks, it is sufficient to add the suffix `-small` to the `<benchmark>` placeholder.
```
sh exp/<benchmark>-small.sh`
```
For example, `sh exp/zinc-small.sh` will run the training on ZINC with parameter budget.

## SR synthetic experiment

In order to run an experiment on the SR benchmark, run the following:
```
sh exp/sr.sh
```
The shell script will internally run the `exp/run_sr_ex.py` script, passing the required parameters. The script will instantiate and run a CIN model on all the SR families, repeating each experiment with 5 different random seeds. It will then print on screen the failure rate statistics on every family, and also dump this result on file, under `exp/results/sr/`.

_Note_: before the inference starts, the script will perform the appropriate ring-lifting procedure on the SR graphs in the family.

Finally, the following command will run the MLP-sum (strong) baseline described in the paper:
```
sh exp/sr_base.sh
```