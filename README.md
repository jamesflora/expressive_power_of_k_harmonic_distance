# The Expressive Power of k-Harmonic Distances for Message Passing Neural Networks

Code for the NeurIPS submission The Expressive Power of k-Harmonic Distances for Message Passing Neural Networks.


## Instructions

### Python environment setup with Conda

```bash
conda create -n kharmonic python=3.10
conda activate kharmonic

conda install pytorch=1.13 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg=2.2 -c pyg -c conda-forge
pip install pyg-lib -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

conda install openbabel fsspec rdkit -c conda-forge

pip install pytorch-lightning yacs torchmetrics
pip install performer-pytorch
pip install tensorboardX
pip install ogb
pip install wandb
pip install brec

conda clean --all
```

### Running an experiment
```bash
conda activate kharmonic

# Running an arbitrary config file in the `configs` folder
python main.py --cfg configs/<config_file>.yaml  wandb.use False
```

For example, to run ZINC with effective resistance and 4 layers of message passing:
```bash
conda activate kharmonic

# Running the ZINC experiment
python main.py --cfg configs/ZINC/zinc-resistance-4hop.yaml  wandb.use False
```

### W&B logging
To use W&B logging, set `wandb.use True` and set entity with `wandb.entity`.


## Citation
The papers that developed the original code for this repository are as follows: 

```bibtex
  @inproceedings{black2024comparing,
  title={Comparing Graph Transformers via Positional Encodings},
  author={Black, Mitchell and Wan, Zhengchao and Mishne, Gal and Nayyeri, Amir and Wang, Yusu},
  booktitle={International Conference on Machine Learning},
  year={2024},
  organization={PMLR}
  }
```

```bibtex
@article{muller2024attending,
title={Attending to Graph Transformers},
author={Luis M{\"u}ller and Mikhail Galkin and Christopher Morris and Ladislav Ramp{\'a}{\v{s}}ek},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2024}
}
```
```bibtex
@article{rampasek2022GPS,
  title={{Recipe for a General, Powerful, Scalable Graph Transformer}}, 
  author={Ladislav Ramp\'{a}\v{s}ek and Mikhail Galkin and Vijay Prakash Dwivedi and Anh Tuan Luu and Guy Wolf and Dominique Beaini},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  year={2022}
}
```

