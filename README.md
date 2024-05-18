# Diploma thesis at HSE AMI. Analysis of Neural Networks Internal Representations During Transfer Learning

This repo is the source code of my bachelor thesis: [pdf](https://github.com/tgritsaev/ens-hse-diploma/tree/master/diploma.pdf)

**Analysis of Neural Networks Internal Representations During Transfer Learning** \
is done by Timofei Gritsaev supervised by [Ildus Sadrtdinov](https://scholar.google.com/citations?user=XhqNegUAAAAJ&hl=en)\*

## Abstract

<div align="justify">
<img align="right" width=35% src="https://github.com/isadrtdinov/ens-for-transfer/blob/master/images/logo.jpg" />
Transfer learning and ensembling are two popular techniques for improving the performance and robustness of neural networks. In practice, ensembles trained from a single pre-trained checkpoint are often used due to the significant expense of pre-training. Nevertheless, the most naive approach to fine-tune ensemble results in similar models and suboptimal accuracy. In this work, we discuss Snapshot Ensembling (SSE) and its modification StarSSE. We improve the former using similarity losses: Representation Topology Distance (RTD) and Mean Squared Error (MSE). Our modifications reduce quality degradation, forced by losing pre-trained knowledge. We research possibilities of ensemble training time reduction via layers freeze, which leads to an accuracy-time trade-off. Then, we show that successful ideas like weights orthogonalization in supervised training can be useless in the transfer learning setup. Finally, we show the possibility of increasing StarSSE diversity using directly inducing its loss.
</div>

## Code

### Environment
The project requirements are listed in `requirements.txt` file. To create a pip/conda environment:

```
# using pip
pip install -r requirements.txt

# using Conda
conda create --name ens_for_transfer --file requirements.txt
```

Note:
- As during fine-tuning the images are resized to `224x224` and given `batch_size=256`, training requires a GPU with at least **32 Gb memory**, e.g., **NVIDIA V100/A100**.
- Logging is done with the [`wandb`](https://wandb.ai/) library, so make sure to log in before launching the experiments.

### Download pre-trained checkpoint

- BYOL ResNet-50 ImageNet pre-trained checkpoints are available [here](https://drive.google.com/drive/folders/1BONZZ6pytC3yP2EXcZJaB07z4eKmtx20?usp=sharing)

### Experiments

Scripts for launching experiments are located in the [`scripts/`](https://github.com/tgritsaev/ens-hse-diploma/tree/master/scripts) directory. 

To launch experiments with optimal StarSSE run the following command:
```sh
python scripts/byol/starsse.py
```
For optimal StarSSE-CE run the following command:
- For training SSE with different cycle hyperparameters
```sh
python scripts/byol/byol_starsse_ce.py
```
and go on
