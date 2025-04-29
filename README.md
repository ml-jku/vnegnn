
# VN-EGNN: E(3)-Equivariant Graph Neural Networks with Virtual Nodes Enhance Protein Binding Site Identification

[![Open in HuggingFace](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/ml-jku/vnegnn)
[![](https://img.shields.io/badge/paper-arxiv2310.06763-red?style=plastic&logo=GitBook)](https://arxiv.org/abs/2404.07194)
[![](https://img.shields.io/badge/model-pink?style=plastic&logo=themodelsresource)](https://huggingface.co/fses91/VNEGNN-MODEL)
[![](https://img.shields.io/badge/project_page-blue?style=plastic&logo=internetcomputer)](https://linktr.ee/vnegnn)

## News

🔥 ***April 10 2024***:  *The trained VNEGNN model and processed dataset are released, as in the paper!*

## Overview

Implementation of the VN-EGNN, state-of-the-art method for protein binding site identfication, by Florian Sestak, Lisa Schneckenreiter, Johannes Brandstetter, Sepp Hochreiter, Andreas Mayr, Günter Klambauer. This repository contains all code, instructions and model weights necessary to run the method or to retrain a model. If you have any question, feel free to open an issue or reach out to: <sestak@ml.jku.at>.

![](visualizations/overview.jpg)

## Installation

[![](https://img.shields.io/badge/PyTorch-2.1.2-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.2.1-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

To reproduce the results please use Python 3.9, PyTorch version 2.1.2, Cuda 12.1, PyG version 2.3.0.

Clone the repo:

```
git clone https://github.com/ml-jku/vnegnn
```

Setup dependencies:

```
conda create --name vnegnn python=3.9
conda activate vnegnn
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pyg==2.4.0 -c pyg
conda env update --name vnegnn --file environment.yaml
```

Activate the environment:

```
conda activate vnegnn
```

## Usage Examples

Following commands can be executed in the directory of the cloned repository.

### Predict

Provide your protein in `.pdb` format.

```
python predict.py -i protein.pdb -o output -c model.ckpt -d cuda:0               # run prediction on GPU

python predict.py -i protein.pdb -o output -c model.ckpt -d cuda:0 -v            # run prediction on GPU and create visualization
```

#### Predict Output

The model outputs:

- `prediction.csv`: holding the predicted positions (x,y,z) of the virtual nodes and the corresponding ranks
- `visualization.pse`: PyMOL session file containing the protein structure with the predicted virtual node positions, only created if the `-v` flag is used.a

## Data

In our training and evaluation process, we adopted the methodology outlined in [Equipocket](https://arxiv.org/abs/2302.12177) with a modification: we utilized a single validation set in place of the 5-fold cross-validation, due to computational limitations.
The processed datasets will be released soon, because of the large size of the datasets (graph with esm embeddings), we will upload the pipeline to process the raw data.

### Raw Datasets

#### SC-PDB

<https://github.com/jivankandel/PUResNet/blob/main/scpdb_subset.zip>

#### COACH420/HOLO4K

<https://github.com/rdk/p2rank-datasets>

These two datasets do not include ligands in a pre-extracted format. Instead, the MOAD database contains the ligand information, according to the criteria of *relevant ligand* as specified in the MOAD database.

Ligand Information:
<http://www.bindingmoad.org/files/csv/every_bind.csv>

#### PDBBind

<http://www.pdbbind.org.cn/download/PDBbind_v2020_other_PL.tar.gz>

## Replicate results

If you want to replicated the results from the paper, put the downloaded datasets in the specific folders.

- `bindingsite_test_data/coach420/raw`

- `bindingsite_test_data/holo4k/raw`

- `bindingsite_test_data/holo4k_split_chain/raw`

  This dataset is the same as the holo4k dataset, but the chains are split into separate files. Look in the paper for more information.

- `bindingsite_test_data/pdbbind2020/raw`

The holo4k and the coach420 dataset were created by extracting the relevant ligans from the MOAD database. All ligand protein complexes were put into a different folder, at inference we combine them by there pdb id to evalutate the model based on all the bindingsites for a given protein.

### Download data
Download the dataset from  [zenodo](https://zenodo.org/records/10782177), the original processing was done from the MODA database, which is not available anymore.

### Process the datasets

This will create a `lmdb` database for each dataset. The `lmdb` and avoids processing all the data again if you want to create a graph with different parameters.

```
python process_dataset.py --input_path data/scpdb/scpdb_subset_puresnet/raw --output_path data/scpdb/scpdb_subset_puresnet/raw --device cuda:0 --n_jobs 8
python process_dataset.py --input_path data/bindingsite_test_data/coach420/raw --output_path data/bindingsite_test_data/coach420/raw --device cuda:0 --n_jobs 8
python process_dataset.py --input_path data/bindingsite_test_data/holo4k/raw --output_path data/bindingsite_test_data/holo4k/raw --device cuda:0 --n_jobs 8
python process_dataset.py --input_path data/bindingsite_test_data/holo4k_split_chain/raw --output_path data/bindingsite_test_data/holo4k_split_chain/raw --device cuda:0 --n_jobs 8
python process_dataset.py --input_path data/bindingsite_test_data/pdbbind2020/raw --output_path data/bindingsite_test_data/pdbbind2020/raw --device cuda:0 --n_jobs 8
```

### Train

Train the *heterogenous* or *homogenous* model with the parameters as used in the paper.

```python
python train.py --config-name=config_binding_hetero        # heterogenous model, top performing model in the paper
```

For training on [SLURM](https://www.schedmd.com/) cluster with [submitit](https://github.com/facebookincubator/submitit)  used the `conf/hydra/meluxina.yaml` as blueprint.

```python
python train.py --config-name=config_binding_hetero hydra=meluxina --multirun        # homogenous model traind on SLURM hydra=meluxina
```

### Run on test dataset

```python
# Run this from the root of the project
from src.models.egnn import EGNNClassifierGlobalNodeHetero
from hydra.utils import instantiate
import pytorch_lightning as pl

path_to_config = "/path/to_config/"
cfg = OmegaConf.create(run.config)  
  
datamodule = instantiate(cfg.datamodule)
ckpt_file = "/path/to/your/checkpoint.ckpt"
    
model = EGNNClassifierGlobalNodeHetero.load_from_checkpoint(
    ckpt_file,
    strict=False,
    segmentation_loss=instantiate(cfg.model.segmentation_loss),
    global_node_pos_loss=instantiate(cfg.model.global_node_pos_loss),
)
model.eval()

trainer = pl.Trainer(devices=1)
loader = datamodule.named_test_loaders["test_coach420"] # name of dataloader ["test_coach420", "test_pdbbind2020", "test_holo4k"]
trainer.test(model, dataloaders=loader)               

# CAUTION: for the test_holo4k dataset, if you evalute it like this you will get lower scores than reported in the paper,
# For the results in the paper we splitted the proteins into chains, run the predictions and combined them (clean code for this procedure will be released in future)
# The intuition behind this step is explained in the paper.
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
