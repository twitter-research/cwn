# Message Passing Simplicial Networks

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
CUDA=cu102
TORCH=1.7.0
pip install torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse==0.6.8 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster==1.5.8 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric==1.6.3
```
or, as an alternative:
```
sh pyG_install.sh
```

Install other required packages via pip:
```
pip install pytest==6.2.1
pip install gudhi==3.4.0
pip install ogb==1.3.1
pip install pyyaml==3.12
pip install jupyter==1.0.0
pip install matplotlib==2.2.2
```
or, as an alternative:
```
pip install -r requirements.txt
```

Install graph-tool via conda:
```
conda install -c conda-forge graph-tool==2.29
```
or, as an alternative:
```
sh graph-tool_install.sh
```

At this point you should be good to go. :)


## Tuning SparseSIN on TU datasets

#### Unpack dataset folder

First and foremost, decompress the dataset folder with:
```
unzip datasets.zip
```
When prompted whether to replace datasets already present in the repository, simply enter `N` (None).

#### Define the search space

Grids are defined via `yaml` configuration files under `exp/tuning_configurations/`.
Choose one of them or define a new one following the same scheme as in `exp/tuning_configurations/template.yml`.

#### Prepare the scripts

Open file `./exp/run_tu_tuning.py`. Change global variable `__max_devices__` to the number of GPUs you want to run the tuning on, in parallel. The deafult is `8`.

Open file `./exp/launch_tu_tuning.sh`.
On lines `2, 3`, specify the device index range on which the tuning will run in parallel. The range must include the same number of devices specified in the step above. For example, if you have set `__max_devices__ = 8`, then specify `lower=0` and `upper=7`.
Mind that you do have a certain freedom to define how many devices to use and which ones. For example, suppose you have a server with `8` devices, but you want to only use devices `3, 4, 5, 6`, it is sufficient to set `__max_devices__ = 4`, `lower=3`, `upper=6`.

On line `4`, specify the grid you want to use and the folder where to save the results. Change `<path_to_grid>` with the absolute path to the `yaml` file specifying the grid. For example, if this repository is rooted at `/home/ubuntu/git/scn`, and the config file is called `IMDBB_ring_grid.yaml`, then replace `<path_to_grid>` with `/home/ubuntu/git/scn/exp/tuning_configurations/IMDBB_ring_grid.yaml`.
On line `5`, change `<exp_name>` with the name you want to give to the tuning experiment. This will determine where name of the folder where results will be stored. An example could be `imdbb_20210515_rings`.

#### Launch the jobs

Suggestion: open a `tmux` or `screen` session from where to execute the following commands.

Activate the virtual environment. Open a terminal and run:
```
conda activate scn_102
```

Then, from the root repository folder, launch the tuning with:
```
sh ./exp/launch_tu_tuning.sh
```

At this point the jobs should be running in parallel in the background, and you should see their outputs printed on your terminal window. 

#### Inspect the tuning results

The tuning jobs will write the results under `exp/results/<dataset_name>_tuning_<exp_name>/`.
You can inspect tuning results via Jupyter Notebook / Lab. Open notebook `exp/analyse_TU_tuning.ipynb`. In the second cell, set the `exp_name` and `dataset` variables. Run all cells.

The last cell will print the top-5 scores and the configurations hyperparameters in variable `inspect_args`.
