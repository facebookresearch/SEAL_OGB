SEAL\_OGB -- An Implementation of SEAL for OGB Link Prediction Tasks
===============================================================================

About
-----
This repository supports the following paper:
> M. Zhang, P. Li, Y. Xia, K. Wang, and L. Jin, Labeling Trick: A Theory of Using Graph Neural Networks for Multi-Node Representation Learning. [\[PDF\]](https://arxiv.org/pdf/2010.16103.pdf)

SEAL is a GNN-based link prediction method. It first extracts a k-hop enclosing subgraph for each target link, then applies a labeling trick named Double Radius Node Labeling (DRNL) to give each node an integer label as its additional feature. Finally, these labeled enclosing subgraphs are fed to a graph neural network to predict link existences.

This repository reimplements SEAL with the PyTorch-Geometric library, and tests SEAL on the Open Graph Benchmark (OGB) datasets. SEAL ranked 1st place on 3 out of 4 link prediction datasets in the [OGB Leaderboard](https://ogb.stanford.edu/docs/leader_linkprop/) at the time of submission. It additionally supports Planetoid like datasets, such as Cora, CiteSeer and PubMed, where random 0.85/0.05/0.1 split and AUC metric are used. Using custom datasets is also easy by replacing the Planetoid dataset with your own.

|              | ogbl-ppa | ogbl-collab | ogbl-ddi | ogbl-citation2 |
|--------------|---------------------|-----------------------|--------------------|---------------------|
| Val results |  51.25%&plusmn;2.52%* |    64.95%&plusmn;0.43%* | 28.49%&plusmn;2.69% |   87.57%&plusmn;0.31%* |
| Test results |  48.80%&plusmn;3.16%* |    64.74%&plusmn;0.43%* | 30.56%&plusmn;3.86% |   87.67%&plusmn;0.32%* |

\* State-of-the-art results; evaluation metrics are Hits@100, Hits@50, Hits@20 and MRR, respectively. For ogbl-collab, we have switched to the new [rule](https://ogb.stanford.edu/docs/leader_rules/), where after all hyperparameters are determined on the validation set, we include validation edges in the training graph and retrain to report the test performance. For ogbl-citation2, it is an updated version of the deprecated ogbl-citation.

The original implementation of SEAL is [here](https://github.com/muhanzhang/SEAL).

The original paper of SEAL is:
> M. Zhang and Y. Chen, Link Prediction Based on Graph Neural Networks, Advances in Neural Information Processing Systems (NIPS-18). [\[PDF\]](https://arxiv.org/pdf/1802.09691.pdf)

This repository also implements some other labeling tricks, such as Distance Encoding (DE) and Zero-One (ZO), and supports combining labeling tricks with different GNNs, including GCN, GraphSAGE and GIN.

Requirements
------------

Latest tested combination: Python 3.8.5 + PyTorch 1.6.0 + PyTorch\_Geometric 1.6.1 + OGB 1.2.4.

Install [PyTorch](https://pytorch.org/)

Install [PyTorch\_Geometric](https://rusty1s.github.io/pytorch_geometric/build/html/notes/installation.html)

Install [OGB](https://ogb.stanford.edu/docs/home/)

Other required python libraries include: numpy, scipy, tqdm etc.

Usages
------

### ogbl-ppa

    python seal_link_pred.py --dataset ogbl-ppa --num_hops 1 --use_feature --use_edge_weight --eval_steps 5 --epochs 20 --dynamic_train --dynamic_val --dynamic_test --train_percent 5 

### ogbl-collab

    python seal_link_pred.py --dataset ogbl-collab --num_hops 1 --train_percent 15 --hidden_channels 256 --use_valedges_as_input

According to OGB, this dataset allows including validation links in training when all the hyperparameters are finalized using the validation set. Thus, you should first tune your hyperparameters without "--use_valedges_as_input", and then append "--use_valedges_as_input" to your final command when all the hyperparameters are determined. See [issue](https://github.com/snap-stanford/ogb/issues/84).

### ogbl-ddi

    python seal_link_pred.py --dataset ogbl-ddi --num_hops 1 --ratio_per_hop 0.2 --use_edge_weight --eval_steps 1 --epochs 10 --dynamic_val --dynamic_test --train_percent 1 

For the above three datasets, append "--runs 10" to do experiments for 10 times and get the average results.

### ogbl-citation2

    python seal_link_pred.py --dataset ogbl-citation2 --num_hops 1 --use_feature --use_edge_weight --eval_steps 1 --epochs 10 --dynamic_train --dynamic_val --dynamic_test --train_percent 2 --val_percent 1 --test_percent 1

Because this dataset uses mean reciprocal rank (MRR) as the evaluation metric where each positive testing link is ranked against 1000 random negative ones, it requires extracting 1001 enclosing subgraphs for *every* testing link. This is very time consuming. Thus, the above command uses "--val_percent 1" and "--test_percent 1" to only evaluate on 1% of validation and test data to get a fast unbiased estimate of the true MRR. To get the true MRR, please change them to "--val_percent 100" and "test_percent 100". Also, because this dataset is expensive to evaluate, we first train 10 models with 1% validation data in parallel, record the best epoch's model from each run, and then evaluate all 10 best models together using the "--test_multiple_models --val_percent 100 --test_percent 100" option. This option enables evaluating multiple pretrained models together with a single subgraph extraction process for each link, thus avoiding extracting subgraphs for testing links repeatedly for 10 times. You need to specify your pretrained model paths in "seal_link_pred.py".

### Cora

    python seal_link_pred.py --dataset Cora --num_hops 3 --use_feature --hidden_channels 256 --runs 10

### CiteSeer

    python seal_link_pred.py --dataset CiteSeer --num_hops 3 --hidden_channels 256 --runs 10

### PubMed

    python seal_link_pred.py --dataset PubMed --num_hops 3 --use_feature --dynamic_train --runs 10

For all datasets, if you specify "--dynamic_train", the enclosing subgraphs of the training links will be extracted on the fly instead of preprocessing and saving to disk. Similarly for "--dynamic_val" and "--dynamic_test". You can increase "--num_workers" to accelerate the dynamic subgraph extraction process.

If your dataset is large, using the default train/val/test split function might result in OOM. You can add "--fast_split" in this case to do a fast split, which cannot guarantee edges (i, j) and (j, i) won't both appear in the negative links but has a better scalability.

Other labeling tricks
---------------------

By default SEAL uses the DRNL labeling trick. You can alternatively use other labeling tricks such as DE (distance encoding), DE+, ZO (zero-one labeling), etc., by appending "--node_label de", "--node_label de+", and "--node_label zo".

Heuristic methods
-----------------

This repository also implements two link prediction heuristics: Common Neighbor (CN) and Adamic Adar (AA), which turn out to have surprisingly better performance than many GNN methods on ogbl-ppa and ogbl-collab. An example usage of Common Neighbor is:

    python seal_link_pred.py --use_heuristic CN --dataset ogbl-ppa

License
-------

SEAL\_OGB is released under an MIT license. Find out more about it [here](https://github.com/facebookresearch/SEAL_OGB/blob/master/LICENSE).

Reference
---------

If you find the code useful, please cite our papers.

	@article{zhang2021labeling,
      title={Labeling Trick: A Theory of Using Graph Neural Networks for Multi-Node Representation Learning},
      author={Zhang, Muhan and Li, Pan and Xia, Yinglong and Wang, Kai and Jin, Long},
      journal={Advances in Neural Information Processing Systems},
      volume={34},
      year={2021}
    }

    @inproceedings{zhang2018link,
      title={Link prediction based on graph neural networks},
      author={Zhang, Muhan and Chen, Yixin},
      booktitle={Advances in Neural Information Processing Systems},
      pages={5165--5175},
      year={2018}
    }

Muhan Zhang, Facebook AI

10/13/2020
