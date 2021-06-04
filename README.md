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

At this point you should be good to go.

## Testing

We provide a series of Python unit tests that can be conveniently run by `pytest`.
We suggest running all tests in the repository after installation to verify everything is in place.
Open a terminal window and move to the repository root folder. Activate the environment created by following the procedure above. Simply run:
```
pytest .
```
This command will recursively fetch all unit tests present in the repository and run them. A summary will be printed out in the end. All tests should pass (showed in green).

## Experiments on molecular benchmarks

In order to run an experiment on a molecular benchmark follow the following steps.
- Open a terminal window
- Activate the virtual environment
- Run `sh exp/<benchmark>.sh`
Do not forget to replace `<benchmark>` with one amongst `zinc`, `zinc-full`, `molhiv`

These shell scripts will run the `exp/run_mol_exp.py` script passing the required parameters for each experiment.

Internally, for a specified range of random seeds, the script will run the trainings sequentially, compute final performance statistics and print them on screen in the end. Additionally, the script will write these results under `exp/results/<benchmark>/`.