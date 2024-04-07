
  
# ContinualMT  

## Quick Links  
  
 - [Introduction](#introduction)  
 - [Dataset](#dataset)  
 - [Architecture](#architecture)  
 - [Installation](#installation)  
 - [Preparing and Preprocessing](#preparing-and-preprocessing)  
 - [Domain Incremental Training](#domain-incremental-training)  
 - [Reference](#reference)  
 
## Introduction    
We introduce ContinualMT, an adaptable continual learning framework tailored for neural machine translation (NMT). It is crafted to promote the research of continual learning (CL) within the realm of NMT. 
    
Our repository encompasses a PyTorch implementation of a suite of state-of-the-art (SoTA) methods, all adhering to a unified training and evaluation protocol. 
Presently, the supported methods include:

* Some widely employed baselines for continual learning:
  * **Seq-FT**: Continual finetuning of a sequence of domains, without any specific attention paid to the issues of forgetting or transfer.
  * **ONE**: Individually finetuning pretrained NMT model for each domain.
  * **Adapter-ONE**: Adds adapter to finetuning pretrained NMT model for each domain 
  * **KD**: Naive knoweldge Distillation  
  * **EWC**: [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796), Kirkpatrick et al., PNAS 2017 

* Recent proposed methods for NMT continual learning:
  * **Dynamic-KD**: [Continual learning for neural machine translation](https://aclanthology.org/2021.naacl-main.310/), Cao et al., NAACL 2021
  * **PTE**: [Pruning-then-expanding model for domain adaptation of neural machine translation](https://aclanthology.org/2021.naacl-main.308/), Gu et al., NAACL 2021
  * **F-MALLOC**: [F-MALLOC: Feed-forward Memory Allocation for Continual Learning in Neural Machine Translation](https://openreview.net/pdf?id=-RUVD4K_IkA), accepted by NAACL 2024

We are actively working on implementing other methods and adding them to this framework!

## Dataset  

Currently, our framework focus on the multi-stage domain incremental training of NMT system. Within our framework, you can utilize machine translation data from various domains for domain incremental training.
We offer a representative multi-domain machine translation dataset: [OPUS multi-domains dataset](https://aclanthology.org/2020.acl-main.692/). This dataset comprises German-English parallel data across five domains: Medical, Law, IT, Koran, and Subtitle. The dataset can be found [here](https://github.com/roeeaharoni/unsupervised-domain-clusters).
  
## Architecture 
Our implementation is built upon [fairseq](https://github.com/facebookresearch/fairseq), with the following modifications:

`./approaches`: code for supported continual learning approaches
`./cl_scripts`: bash scripts for continual training
`./cl_scripts_slurm`: slurm scripts for continual training
`./lcheckpoints`: all training checkpoints are saved in this folder
`./logs`: training logs
`./pretrained_models`: folder for pretrained NMT models
`./task_sequence`: reference sequences for OPUS multi-domain MT data

## Installation
Firstly, build the enviorment from the provied YAML file.
```conda env create --name CLMT --file CLMT.yaml```

Install fairseq (```pip install --editable .```), [moses](https://github.com/moses-smt/mosesdecoder) and [fastBPE](https://github.com/glample/fastBPE).

## Preparing and Preprocessing

### Pre-trained model
Download the [pre-trained WMT19 German-English model](https://github.com/facebookresearch/fairseq/blob/v0.12.3/examples/translation/README.md) from fairseq, along with the dictionaries and the bpecodes.

### Data
Firstly, navigate to the data folder by running ```cd ./examples/translation```. Ensure that you have set the paths for the Moses scripts, fastBPE, the model dictionaries, and the BPE codes in the scripts.

For general domain MT data, you can simply execute the provided preprocessing script  ```prepare-wmt17de2en.sh```, which automatically download and prepare the data.

For domain incremental training data, download the [mult-domain data](https://github.com/roeeaharoni/unsupervised-domain-clusters) and unzip it. Then process each domain with the ```prepare-domain-adapt.sh``` script.

Finally, use the ```preprocess.sh``` script to prepare the binary files for fairseq.

## Domain incremental training
We offer training bash scripts for all supported approaches in ```./cl_scripts``` and ```./cl_scripts_slurm```. For more detailed information, please refer to the individual readme files located in each directory.

## Extending for new approaches
Expanding our framework is straightforward. To integrate new CL approaches, you simply need to make modifications within the ```./approaches```, ```./cl_scripts``` and ```./cl_scripts_slurm``` directories.
  
## Reference  
We highly appreciate your act of staring and citing. Your attention to detail and recognition is greatly valued.  
  

