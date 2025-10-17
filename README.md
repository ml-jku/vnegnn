
# VN-EGNN: E(3)-Equivariant Graph Neural Networks with Virtual Nodes Enhance Protein Binding Site Identification

[![](https://img.shields.io/badge/dataset-zenodo-orange?style=plastic&logo=zenodo)](https://zenodo.org/records/17365855)



# Overview

Implementation of the VN-EGNN, state-of-the-art method for protein binding site identfication, by Florian Sestak, Lisa Schneckenreiter, Johannes Brandstetter, Sepp Hochreiter, Andreas Mayr, Günter Klambauer. This repository contains all code, instructions and model weights necessary to run the method or to retrain a model. If you have any question, feel free to open an issue or reach out to: <sestak@ml.jku.at>.

![](visualizations/overview.jpg)

# Installation

## Requirements
- Python 3.9+
- PyTorch 2.1+ (2.7+ recommended)
- CUDA 11.8+ or 12.x (for GPU support)
- PyTorch Geometric 2.4+

## Quick Setup

### 1. Clone the repository:
```bash
git clone https://github.com/ml-jku/vnegnn
cd vnegnn
```

### 2. Create and activate conda environment:
```bash
conda env create -f environment.yaml
conda activate vnegnn
```

### 3. Install PyTorch with CUDA support:

Choose the appropriate command based on your CUDA version. Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) for other configurations.

For CUDA 12.x:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install PyTorch Geometric and CUDA-dependent extensions:

The `torch-scatter`, `torch-sparse`, and `torch-cluster` packages are CUDA-version specific and must match your PyTorch and CUDA versions.

First, check your PyTorch and CUDA versions:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
```

Then install PyTorch Geometric and extensions. Replace `${TORCH}` with your PyTorch version (e.g., `2.1.0`, `2.7.0`) and `${CUDA}` with your CUDA version (e.g., `cu121`, `cu118`, `cpu`):

```bash
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

**Examples:**


For PyTorch 2.7.0 with CUDA 12.8:
```bash
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
```

> **Note:** See the [PyTorch Geometric Installation Guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for the full list of available wheel versions.

### Verify Installation

```bash
python -c "import torch; import torch_geometric; print(f'PyTorch: {torch.__version__}'); print(f'PyG: {torch_geometric.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Setup environment variables

Setup the environment variables for logging in ``.env``, a template can be found here ``.env.template``.

```bash
source .env
```

# Data
The datasets are processed and be downloaded from [this](https://zenodo.org/records/17365855) link. Place the datasets in the folder ``data/data``.
Run the following commands, to setup files used for training:
```bash
./process_data.sh
```
To rerun the Equipocket baseline you need to specify MSMS path, for surface genration.
```bash
./process_data_equipocket.sh
```

The splits for each experiments are provided in the uploaded dataset, e.g. COACH420 (``data/data/coach420/splits``).

# Experiments
Experiment are logged via Weights and Biases, use the [RUN_ID] to evalute the model. To reproduce the results in our publication, run the following commands for the individuall experiments. The evaluation metrics are loggend in wandb and can then be exported as csv for further processing.
If you run an experiment, our script saves the graph as dataset, as described in [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html).


## VN-EGNN
```
# Train
python src/train.py experiment=vnegnn

# Eval
python src/eval.py wandb_run_id=[RUN_ID]
```

## VN-EGNN (Train PDBBind2020)
```
# Train
python src/train.py experiment=vnegnn_pdbbind2020

# Eval
python src/eval.py wandb_run_id=[RUN_ID]
```

## VN-EGNN (Train GRASP benchmark)
We also compared VN-EGNN on a different scPDB dataset split proposed by [GrASP](https://pubs.acs.org/doi/10.1021/acs.jcim.3c01698). The datasets for this, can be found in their provided repository. (To rerun their experiments, place their datasets under ``data/grasp`` and run the data processing pipline desribed above on this folder.)
```
# Train
python src/train.py experiment=vnegnn_pdbbind2020

# Eval
python src/eval.py wandb_run_id=[RUN_ID]
```

## Baseline Equipocket
```
# Train
python src/train.py experiment=equipocket

# Eval
python src/eval.py wandb_run_id=[RUN_ID]
```

# Project structure
## Configruation
Configuration is managed with Hydra configs, structured as follows.

```
📁 configs
├── 📁 callbacks                # Callbacks (e.g. checkpointing, ...)
├── 📁 data                     # Dataset configs
├── 📁 debug                    # Debug configs
├── 📁 experiment               # Contains all experiments reported in the publication.
├── 📁 extras                   # Extra configurations.
├── 📁 hydra                    # Hydra configurations.
├── 📁 local                    # Local setup files.
├── 📁 logger                   # Logger setup (wandb logger was used for all experiments)
├── 📁 model                    # Model configurations
├── 📁 paths                    # Paths setup.
├── 📁 trainer                  # Lighting trainer configuration
├── 📄 eval.yaml                # Train config.
└── 📄 train.yaml               # Eval config.
```

## Source code
The following shows the structure of the source code. The training pipeline is setup with [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/).

```
📁 src
├── 📁 datasets                    # Dataset implementations
│   ├── 📄 binding_dataset.py      # Binding site dataset class
│   ├── 📄 equipocket_dataset.py   # Equipocket dataset class
│   └── 📄 utils.py                # Dataset utilities
├── 📁 models                      # Model architectures
│   ├── 📁 equipocket              # Equipocket baseline models
│   │   ├── 📄 baseline_models.py  # Baseline model implementations
│   │   ├── 📄 egnn_clean.py       # Clean EGNN implementation
│   │   ├── 📄 equipocket.py       # Equipocket model
│   │   └── 📄 surface_egnn.py     # Surface-based EGNN
│   └── 📁 vnegnn                  # VN-EGNN models
│       ├── 📄 aggregation.py      # Aggregation layers
│       ├── 📄 utils.py            # Model utilities
│       └── 📄 vnegnn.py           # VN-EGNN implementation
├── 📁 modules                     # Training components
│   ├── 📄 callbacks.py            # Custom Lightning callbacks
│   ├── 📄 cluster.py              # Clustering utilities
│   ├── 📄 ema.py                  # Exponential moving average
│   ├── 📄 losses.py               # Loss functions
│   ├── 📄 metrics.py              # Evaluation metrics
│   └── 📄 schedulers.py           # Learning rate schedulers
├── 📁 utils                       # Utility functions
│   ├── 📄 constants.py            # Constants and definitions
│   ├── 📄 graph.py                # Graph processing utilities
│   ├── 📄 instantiators.py        # Hydra instantiation helpers
│   ├── 📄 logging_utils.py        # Logging utilities
│   ├── 📄 misc.py                 # Miscellaneous utilities
│   ├── 📄 protein.py              # Protein processing
│   ├── 📄 pylogger.py             # Python logger
│   ├── 📄 rich_utils.py           # Rich text formatting
│   ├── 📄 tensor_utils.py         # Tensor manipulation
│   ├── 📄 torch_utils.py          # PyTorch utilities
│   └── 📄 utils.py                # General utilities
├── 📁 wrappers                    # Lightning module wrappers
│   ├── 📄 base.py                 # Base wrapper class
│   ├── 📄 bindingsites.py         # VNEGNN wrapper
│   └── 📄 equipocket.py           # Equipocket wrapper
├── 📄 train.py                    # Training script
└── 📄 eval.py                     # Evaluation script
```





## Citation

```
@misc{sestak2024vnegnn,
    title={VN-EGNN: E(3)-Equivariant Graph Neural Networks with Virtual Nodes Enhance Protein Binding Site Identification},
    author={Florian Sestak and Lisa Schneckenreiter and Johannes Brandstetter and Sepp Hochreiter and Andreas Mayr and Günter Klambauer},
    year={2024},
    eprint={2404.07194},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## License

MIT

## Acknowledgements

The ELLIS Unit Linz, the LIT AI Lab, the Institute for Ma-
chine Learning, are supported by the Federal State Upper
Austria. We thank the projects AI-MOTION (LIT-2018-
6-YOU-212), DeepFlood (LIT-2019-8-YOU-213), Medi-
cal Cognitive Computing Center (MC3), INCONTROL-
RL (FFG-881064), PRIMAL (FFG-873979), S3AI (FFG-
872172), DL for GranularFlow (FFG-871302), EPILEP-
SIA (FFG-892171), AIRI FG 9-N (FWF-36284, FWF-
36235), AI4GreenHeatingGrids(FFG- 899943), INTE-
GRATE (FFG-892418), ELISE (H2020-ICT-2019-3 ID:
951847), Stars4Waters (HORIZON-CL6-2021-CLIMATE-
01-01). We thank Audi.JKU Deep Learning Center,
TGW LOGISTICS GROUP GMBH, Silicon Austria Labs
(SAL), FILL Gesellschaft mbH, Anyline GmbH, Google,
ZF Friedrichshafen AG, Robert Bosch GmbH, UCB Bio-
pharma SRL, Merck Healthcare KGaA, Verbund AG, GLS
(Univ. Waterloo) Software Competence Center Hagen-
berg GmbH, TÜV Austria, Frauscher Sensonic, TRUMPF
and the NVIDIA Corporation. We acknowledge EuroHPC
Joint Undertaking for awarding us access to Karolina at
IT4Innovations, Czech Republic; MeluXina at LuxProvide,
Luxembourg; LUMI at CSC, Finland.

![](visualizations/1odi_3lpk.png)
