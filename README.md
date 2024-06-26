# LEIGNN

## Dataset
We use ISO17, H2O, and CH4 as examples to illustrate how to use LEIGNN.

- **ISO17 Dataset** [1]: Available at [ISO17](http://quantum-machine.org/datasets/).
- **H2O and CH4 Datasets**: Available at [H2O and CH4](https://zenodo.org/records/10208201).

## Requirements
Required Python packages include:  
- `ase==3.22.1`
- `config==0.5.1`
- `lmdb==1.4.1`
- `matplotlib==3.7.2`
- `numpy==1.24.4`
- `pandas==2.1.3`
- `pymatgen==2023.5.10`
- `scikit_learn==1.3.0`
- `scipy==1.11.4`
- `torch==1.13.1`
- `torch_geometric==2.2.0`
- `torch_scatter==2.1.0`
- `tqdm==4.66.1`

Alternatively, install the environment using the provided YAML file at `./environment/environment.yaml`.

## Logger
For logging, we recommend using wandb. More details are available at https://wandb.ai/. Training logs and trained models are stored in the `./wandb` directory.

## Step-by-Step Guide

### Data Preprocessing
Download the data from [Zenodo](https://zenodo.org/record/8208800).
The downloaded data are preprocessed by default. If you wish to preprocess them from scratch, run:
- `python preprocess_ISO17.py --data_root /path/to/iso17 --num_workers 8` for the ISO17 dataset.
- `python preprocess_MD.py --data_root /path/to/CH4 --num_workers 8` for the CH4 dataset.
- `python preprocess_MD.py --data_root /path/to/H2O --num_workers 8` for the H2O dataset.

Replace `/path/to/` with your directory paths.

### Train the Model
To train LEIGNN, run:
- `python train_ISO17.py --data_root /path/to/iso17 --num_workers 4` for ISO17.
- `python train_MD.py --data_root /path/to/CH4 --systems CH4 --num_workers 4` for CH4.
- `python train_MD.py --data_root /path/to/H2O --systems H2O --num_workers 4` for H2O.

Replace `/path/to/` with your directory paths.

### Test the Model
To test LEIGNN on ISO17, run:
- `python test_ISO17.py --data_root /path/to/iso17 --model_dir ./wandb/run-20231031_144315-LEIGNN_20231031_144314/`

Replace `/path/to/` and `./wandb/run-20231031_144315-LEIGNN_20231031_144314/` with your directory path.

### MD Simulations
After training LEIGNN on the CH4 dataset, run MD simulations with:
- `python simulate.py --data_root /path/to/ch4 --model_dir ./wandb/run-20231124_233309-LEIGNN_CH4_20231124_233308`
Replace `/path/to/` and `./wandb/run-20231124_233309-LEIGNN_CH4_20231124_233308` with your directory path.
Evaluate and visualize MD simulation results using `performance_CH4.ipynb`.

## Acknowledgements
Some part of code in this project were adapted from [OCP](https://github.com/Open-Catalyst-Project/ocp) and [MDsim](https://github.com/kyonofx/MDsim). We gratefully acknowledge the contributions from these sources.
## Reference
[1] Schütt K, Kindermans P J, Sauceda Felix H E, et al. "Schnet: A continuous-filter convolutional neural network for modeling quantum interactions." Advances in Neural Information Processing Systems, 2017, 30.
