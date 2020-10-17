SEAL\_OGB -- An Implementation of SEAL for OGB Link Prediction Tasks
===============================================================================

About
-----
SEAL is a GNN-based link prediction method. It first extracts a k-hop enclosing subgraph for each target link, then applies a Double Radius Node Labeling (DRNL) scheme to give each node an integer label as its additional feature. Finally, these labeled enclosing subgraphs are fed to a graph neural network to predict link existences.

This repository implements SEAL with the PyTorch-Geometric library, and tests SEAL in the Open Graph Benchmark (OGB) datasets. SEAL ranked 1st place on 3 out of 4 link prediction datasets in the [OGB Leaderboard](https://ogb.stanford.edu/docs/leader_linkprop/). It additionally supports Planetoid like datasets, such as Cora, CiteSeer and PubMed, where random 0.85/0.05/0.1 split and AUC metric are used. Using custom datasets is also easy by replacing the Planetoid dataset with your own.

The original implementation of SEAL is [here](https://github.com/muhanzhang/SEAL).

The original paper of SEAL is:
> M. Zhang and Y. Chen, Link Prediction Based on Graph Neural Networks, Advances in Neural Information Processing Systems (NIPS-18). [\[PDF\]](https://arxiv.org/pdf/1802.09691.pdf)

Requirements
------------

Latest tested combination: Python 3.8.5 + PyTorch 1.6.0 + PyTorch_Geometric 1.6.1 + OGB 1.2.3.

Install [PyTorch](https://pytorch.org/)

Install [PyTorch_Geometric](https://rusty1s.github.io/pytorch_geometric/build/html/notes/installation.html)

Install [OGB](https://ogb.stanford.edu/docs/home/)

Other required python libraries include: numpy, scipy, tqdm etc.

Usages
------

### ogbl-ppa

    python seal_link_pred.py --dataset ogbl-ppa --num_hops 1 --use_feature --use_edge_weight --eval_steps 5 --epochs 20 --dynamic_train --dynamic_val --dynamic_test --train_percent 5 

### ogbl-collab

    python seal_link_pred.py --dataset ogbl-collab --num_hops 1 --use_feature --train_percent 10 

### ogbl-ddi

    python seal_link_pred.py --dataset ogbl-ddi --num_hops 1 --ratio_per_hop 0.2 --use_edge_weight --eval_steps 1 --epochs 10 --dynamic_val --dynamic_test --train_percent 1 

For the above three datasets, append "--runs 10" to do experiments for 10 times and get the average results.

### ogbl-citation

    python seal_link_pred.py --dataset ogbl-citation --num_hops 2 --use_feature --use_edge_weight --eval_steps 10 --epochs 10 --dynamic_train --dynamic_val --dynamic_test --train_percent 2 --val_percent 1 --test_percent 1 

Because this dataset uses mean reciprocal rank (MRR) as the metric where each positive testing link is ranked against 1000 random negative ones, it requires extracting 1001 enclosing subgraphs for *every* testing link. This is very time consuming. Thus, the above command uses "--val_percent 1" and "--test_percent 1" to only evaluate on 1% of validation and test data to get an efficient unbiased estimate of the true MRR. To get the true MRR, please change them to "--val_percent 100" and "test_percent 100" or simply remove them. Also, because this dataset is expensive to evaluate, for the leaderboard results, we first train 10 models without evaluation in parallel, and then evaluate all of them together using the "--test_multiple_models" option. This option enables evaluating multiple pretrained models together with a single subgraph extraction process for each link, thus avoiding extracting subgraphs for testing links repeatedly for 10 times. You need to specify your pretrained model paths in the code. 

### Cora

    python seal_link_pred.py --dataset Cora --num_hops 2 --runs 10

### CiteSeer

    python seal_link_pred.py --dataset CiteSeer --num_hops 2 --runs 10
    
### PubMed

    python seal_link_pred.py --dataset PubMed --num_hops 2 --use_feature --runs 10


For all datasets, if you specify "--dynamic_train", the enclosing subgraphs of the training links will be extracted on the fly instead of preprocessing and saving to disk. Similarly for "--dynamic_val" and "--dynamic_test". You can increase "--num_workers" to accelerate the dynamic subgraph extraction process.

License
-------

SEAL\_OGB is released under an MIT license. Find out more about it [here](https://github.com/facebookresearch/SEAL_OGB/blob/master/LICENSE).

Reference
---------

If you find the code useful, please cite our paper.

    @inproceedings{zhang2018link,
      title={Link prediction based on graph neural networks},
      author={Zhang, Muhan and Chen, Yixin},
      booktitle={Advances in Neural Information Processing Systems},
      pages={5165--5175},
      year={2018}
    }

Muhan Zhang, Facebook AI
10/13/2020
