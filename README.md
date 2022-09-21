# Awesome Machine Unlearning

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![arXiv](https://img.shields.io/badge/arXiv-2209.02299-b31b1b.svg)](https://arxiv.org/abs/2209.02299)

A collection of academic articles, published methodology, and datasets on the subject of **machine unlearning**.

- [Awesome-Machine-Unlearning](#awesome-machine-unlearning)
  - [ A Framework of Machine Unlearning](#a-framework-of-machine-unlearning)
  - [Surveys](#surveys)
  - [Model-agnostic](#model-agnostic)
  - [Model-intrinsic](#model-intrinsic)
  - [Data-Driven](#data-driven)
  - [Datasets](#datasets)
  

Please read and cite our paper: 
>Nguyen, T.T., Huynh, T.T., Nguyen, P.L., Liew, A.W.C., Yin, H. and Nguyen, Q.V.H., 2022. A Survey of Machine Unlearning. arXiv preprint arXiv:2209.02299.

## Citation

```
@article{nguyen2022survey,
  title={A Survey of Machine Unlearning},
  author={Nguyen, Thanh Tam and Huynh, Thanh Trung and Nguyen, Phi Le and Liew, Alan Wee-Chung and Yin, Hongzhi and Nguyen, Quoc Viet Hung},
  journal={arXiv preprint arXiv:2209.02299},
  year={2022}
}
```

----------

## A Framework of Machine Unlearning
[![timeline](framework.png)](https://www.linkedin.com/in/tamlhp/)

----------

## Surveys
| **Paper Title** | **Venue** | **Year** | 
| --------------- | ---- | ---- | 
| [Making machine learning forget](https://www.sciencedirect.com/science/article/pii/S0267364917302091) | _Annual Privacy Forum_ | 2019 |
| [“Amnesia” - A Selection of Machine Learning Models That Can Forget User Data Very Fast](https://www.semanticscholar.org/paper/%22Amnesia%22-Machine-Learning-Models-That-Can-Forget-Schelter/4e99e7af4b9f08b0a89577cd8ea92a37d4744e1e) | _CIDR_ | 2019 |
| [Humans forget, machines remember: Artificial intelligence and the Right to Be Forgotten](https://www.sciencedirect.com/science/article/pii/S0267364917302091) | _Computer Law & Security Review_ | 2018 |
| [Algorithms that remember: model inversion attacks and data protection law](https://doi.org/10.1098/rsta.2018.0083) | _Philosophical Transactions of the Royal Society A_ | 2018 |
----------

## Model-Agnostic
Model-agnostic machine unlearning methodologies include unlearning processes or frameworks that are applicable for different models. In some cases, they provide theoretical guarantees for only a class of models (e.g. linear models). But we still consider them model-agnostic as their core ideas are applicable to complex models (e.g. deep neural networks) with practical results.

| **Paper Title** | **Venue** | **Year** | 
| --------------- | ---- | ---- | 
| [Humans forget, machines remember: Artificial intelligence and the Right to Be Forgotten](https://www.sciencedirect.com/science/article/pii/S0267364917302091) | _Computer Law & Security Review_ | 2018 |
| [Humans forget, machines remember: Artificial intelligence and the Right to Be Forgotten](https://www.sciencedirect.com/science/article/pii/S0267364917302091) | _Computer Law & Security Review_ | 2018 |
| [Humans forget, machines remember: Artificial intelligence and the Right to Be Forgotten](https://www.sciencedirect.com/science/article/pii/S0267364917302091) | _Computer Law & Security Review_ | 2018 |
----------

## Model-Intrinsic
The model-intrinsic approaches include unlearning methods designed for a specific type of models. Although they are model-intrinsic, their applications are not necessarily narrow, as many ML models can share the same type.
| **Paper Title** | **Year** | **Author** | **Venue** | **Model** | **Code** | **Unlearning For** |
| --------------- | :----: | ---- | :----: | :----: | :----: | ---- |
| [Machine Unlearning: Linear Filtration for Logit-based Classifiers](https://link.springer.com/article/10.1007/s10994-022-06178-9) | 2022 | Baumhauer et al. | Machine Learning | normalizing filtration | - | Softmax classifiers |
| [Deep Unlearning via Randomized Conditionally Independent Hessians](https://openaccess.thecvf.com/content/CVPR2022/html/Mehta_Deep_Unlearning_via_Randomized_Conditionally_Independent_Hessians_CVPR_2022_paper.html) | 2022 | Mehta et al. | CVPR | L-CODEC | [[Code]](https://github.com/vsingh-group/LCODEC-deep-unlearning) | DNN-based Models |
| [Variational Bayesian unlearning](https://dl.acm.org/doi/abs/10.5555/3495724.3497068) | 2022 | Nguyen et al. | NeurIPS | VI | - | Bayesian Models  |
| [Revisiting Machine Learning Training Process for Enhanced Data Privacy](https://dl.acm.org/doi/abs/10.1145/3474124.3474208) | 2021 | Goyal et al. | IC3 | - | - | DNN-based Models  |
| [Knowledge Removal in Sampling-based Bayesian Inference](https://openreview.net/forum?id=dTqOcTUOQO) | 2021 | Fu et al. | ICLR | - | [[Code]](https://github.com/fshp971/mcmc-unlearning) | Bayesian Models  |
| [Mixed-Privacy Forgetting in Deep Networks](https://openaccess.thecvf.com/content/CVPR2021/html/Golatkar_Mixed-Privacy_Forgetting_in_Deep_Networks_CVPR_2021_paper.html) | 2021 | Golatkar et al. | CVPR | - | - | DNN-based Models  |
| [HedgeCut: Maintaining Randomised Trees for Low-Latency Machine Unlearning](https://dl.acm.org/doi/abs/10.1145/3448016.3457239) | 2021 | Schelter et al. | SIGMOD | HedgeCut | [[Code]](https://github.com/schelterlabs/hedgecut) | Tree-based Models  |
| [A Unified PAC-Bayesian Framework for Machine Unlearning via Information Risk Minimization](https://ieeexplore.ieee.org/abstract/document/9596170) | 2021 | Jose et al. | MLSP | PAC-Bayesian| - | Bayesian Models  |
| [DeepObliviate: A Powerful Charm for Erasing Data Residual Memory in Deep Neural Networks](https://arxiv.org/abs/2105.06209) | 2021 | He et al. | arXiv | DEEPOBLIVIATE | - | DNN-based Models  |
| [Bayesian Inference Forgetting](https://arxiv.org/abs/2101.06417) | 2021 | Fu et al. | arXiv | BIF | [[Code]](https://github.com/fshp971/BIF) | Bayesian Models  |
| [Approximate Data Deletion from Machine Learning Models](https://proceedings.mlr.press/v130/izzo21a.html) | 2021 | Izzo et al. | AISTATS | PRU | [[Code]](https://github.com/zleizzo/datadeletion) | Linear Models |
| [Online Forgetting Process for Linear Regression Models](https://proceedings.mlr.press/v130/li21a.html) | 2021 | Li et al. | AISTATS | FIFD-OLS | - | Linear Models  |
| [Forgetting Outside the Box: Scrubbing Deep Networks of Information Accessible from Input-Output Observations](https://link.springer.com/chapter/10.1007/978-3-030-58526-6_23) | 2020 | Golatkar et al. | ECCV | - | - | DNN-based Models  |
| [Influence Functions in Deep Learning Are Fragile](https://www.semanticscholar.org/paper/Influence-Functions-in-Deep-Learning-Are-Fragile-Basu-Pope/098076a2c90e42c81b843bf339446427c2ff02ed) | 2020 | Basu et al. | arXiv | - | - | DNN-based Models  |
| [Deep Autoencoding Topic Model With Scalable Hybrid Bayesian Inference](https://ieeexplore.ieee.org/document/9121755) | 2020 | Zhang et al. | IEEE | DATM | - | Bayesian Models  |
| [Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks](https://arxiv.org/abs/1911.04933) | 2020 | Golatkar et al. | CVPR | - | - | DNN-based Models  |
| [Uncertainty in Neural Networks: Approximately Bayesian Ensembling](https://proceedings.mlr.press/v108/pearce20a.html) | 2020 | Pearce et al. | AISTATS | - | [[Code]](https://teapearce.github.io/portfolio/github_io_1_ens/) | Bayesian Models  |
| [Certified Data Removal from Machine Learning Models](https://proceedings.mlr.press/v119/guo20c.html) | 2020 | Guo et al. | ICML | - | - | DNN-based Models  |
| [DeltaGrad: Rapid retraining of machine learning models](https://proceedings.mlr.press/v119/wu20b.html) | 2020 | Wu et al. | ICML | DeltaGrad | [[Code]](https://github.com/thuwuyinjun/DeltaGrad) | DNN-based Models  |
| [Making AI Forget You: Data Deletion in Machine Learning](https://papers.nips.cc/paper/2019/hash/cb79f8fa58b91d3af6c9c991f63962d3-Abstract.html) | 2019 | Ginart et al. | NeurIPS | - | - | Linear Models  |
| [Bayesian Neural Networks with Weight Sharing Using Dirichlet Processes](https://ieeexplore.ieee.org/document/8566011) | 2018 | Roth et al. | IEEE | DP | [[Code]](https://github.com/wroth8/dp-bnn) | Bayesian Models  |

----------

## Data-Driven
The approaches fallen into this category use data partition, data augmentation and data influence to speed up the retraining process.
| **Paper Title** | **Year** | **Author** | **Venue** | **Model** | **Code** | **Type** |
| --------------- | :----: | ---- | :----: | :----: | :----: | ---- |
| [PUMA: Performance Unchanged Model Augmentation for Training Data Removal](https://ojs.aaai.org/index.php/AAAI/article/view/20846) | 2022 | Wu et al. | AAAI | PUMA | - | Data Influence |
| [Certifiable Unlearning Pipelines for Logistic Regression: An Experimental Study](https://www.mdpi.com/2504-4990/4/3/28) | 2022 | Mahadevan and Mathioudakis | MAKE | - | [[Code]](https://version.helsinki.fi/mahadeva/unlearning-experiments) | Data Influence |
| [Zero-Shot Machine Unlearning](https://arxiv.org/abs/2201.05629) | 2022 | Chundawat et al. | arXiv | - | - | Data Influence |
| [GRAPHEDITOR: An Efficient Graph Representation Learning and Unlearning Approach](https://congweilin.github.io/CongWeilin.io/files/GraphEditor.pdf) | 2022 | Cong and Mahdavi | - | GRAPHEDITOR | [[Code]](https://anonymous.4open.science/r/GraphEditor-NeurIPS22-856E/README.md) | Data Influence |
| [Learning to Refit for Convex Learning Problems](https://arxiv.org/abs/2111.12545) | 2021 | Zeng et al. | arXiv | OPTLEARN | - | Data Influence |
| [Fast Yet Effective Machine Unlearning](https://arxiv.org/abs/2111.08947) | 2021 | Ayush et al. | arXiv | - | - | Data Augmentation |
| [SSSE: Efficiently Erasing Samples from Trained Machine Learning Models](https://openreview.net/forum?id=GRMKEx3kEo) | 2021 | Peste et al. | NeurIPS | SSSE | - | Data Influence |
| [Coded Machine Unlearning](https://ieeexplore.ieee.org/document/9458237) | 2021 | Aldaghri et al. | IEEE | - | - | Data Partitioning |
| [Machine Unlearning](https://ieeexplore.ieee.org/document/9519428) | 2021 | Bourtoule et al. | IEEE | SISA | [[Code]](https://github.com/cleverhans-lab/machine-unlearning) | Data Partitioning |
| [How Does Data Augmentation Affect Privacy in Machine Learning?](https://ojs.aaai.org/index.php/AAAI/article/view/17284/) | 2021 | Yu et al. | AAAI | - | [[Code]](https://github.com/dayu11/MI_with_DA) | Data Augmentation |
| [Amnesiac Machine Learning](https://ojs.aaai.org/index.php/AAAI/article/view/17371) | 2021 | Graves et al. | AAAI | AmnesiacML | [[Code]](https://github.com/lmgraves/AmnesiacML) | Data Influence |
| [Unlearnable Examples: Making Personal Data Unexploitable](https://arxiv.org/abs/2101.04898) | 2021 | Huang et al. | ICLR | - | [[Code]](https://github.com/HanxunH/Unlearnable-Examples) | Data Augmentation |
| [Descent-to-Delete: Gradient-Based Methods for Machine Unlearning](https://proceedings.mlr.press/v132/neel21a.html) | 2021 | Neel et al. | ALT | - | - | Data Influence |
| [Fawkes: Protecting Privacy against Unauthorized Deep Learning Models](https://dl.acm.org/doi/abs/10.5555/3489212.3489302) | 2020 | Shan et al. | USENIX Sec. Sym. | Fawkes | [[Code]](https://github.com/Shawn-Shan/fawkes) | Data Augmentation |
| [PrIU: A Provenance-Based Approach for Incrementally Updating Regression Models](https://dl.acm.org/doi/abs/10.1145/3318464.3380571) | 2020 | Wu et al. | SIGMOD | PrIU/PrIU-opt | - | Data Influence |
| [DeltaGrad: Rapid retraining of machine learning models](https://proceedings.mlr.press/v119/wu20b.html) | 2020 | Wu et al. | ICML | DeltaGrad | [[Code]](https://github.com/thuwuyinjun/DeltaGrad) | Data Influence |


----------

## Datasets
### Type: Image
| Dataset | #Items | Disk Size | Downstream Application | #Papers Used |
| :-- | --- | --- | --- | --- |
| [MNIST](https://deepai.org/dataset/mnist) | 70K | 11MB | Classification | 29+ papers |
| [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) | 60K | 163MB | Classification | 16+ papers |  
| [SVHN](http://ufldl.stanford.edu/housenumbers/) | 600K | 400MB+ | Classification | 8+ papers | 
| [LSUN](https://www.yf.io/p/lsun) | 69M+ | 1TB+ | Classification | 1 paper |
| [ImageNet](https://www.image-net.org/) | 14M+   | 166GB | Classification | 6 papers |

### Type: Tabular
| Dataset | #Items | Disk Size | Downstream Application | #Papers Used |
| :-- | --- | --- | --- | --- |
| [Adult](https://archive.ics.uci.edu/ml/datasets/adult) | 48K+ | 10MB | Classification | 8+ papers |
| [Breast Cancer](https://archive.ics.uci.edu/ml/datasets/breast+cancer) | 569 | &lt;1MB | Classification | 2 papers |
| [Diabetes](https://archive.ics.uci.edu/ml/datasets/diabetes) | 442 | &lt;1MB | Regression | 3 papers |

### Type: Text
| Dataset | #Items | Disk Size | Downstream Application | #Papers Used |
| :-- | --- | --- | --- | --- |
| [IMDB Review](https://ai.stanford.edu/~amaas/data/sentiment/) | 50k | 66MB | Sentiment Analysis | 1 paper |
| [Reuters](https://keras.io/api/datasets/reuters/) | 11K+ | 73MB | Categorization | 1 paper |
| [Newsgroup](https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups) | 20K | 1GB+ | Categorization | 1 paper |

### Type: Sequence
| Dataset | #Items | Disk Size | Downstream Application | #Papers Used |
| :-- | --- | --- | --- | --- |
| [Epileptic Seizure](https://archive.ics.uci.edu/ml/datasets/Epileptic%2BSeizure%2BRecognition) | 11K+ | 7MB | Timeseries Classification | 1 paper |
| [Activity Recognition](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) | 10K+ | 26MB | Timeseries Classification | 1 paper | 
| [Botnet](https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT) | 72M | 3GB+ | Clustering | 1 paper |

### Type: Graph
| Dataset | #Items | Disk Size | Downstream Application | #Papers Used |
| :-- | --- | --- | --- | --- |
| [OGB](https://ogb.stanford.edu/) | 100M+ | 59MB | Classification | 2 papers |
| [Cora](https://relational.fit.cvut.cz/dataset/CORA) | 2K+ | 4.5MB | Classification | 3 papers | 
| [MovieLens](http://konect.cc/networks/) | 1B+ | 3GB+ | Recommender Systems | 1 paper |

----------
**Disclaimer**

Feel free to contact us if you have any queries or exciting news on machine unlearning. In addition, we welcome all researchers to contribute to this repository and further contribute to the knowledge of machine unlearning fields.

If you have some other related references, please feel free to create a Github issue with the paper information. We will glady update the repos according to your suggestions. (You can also create pull requests, but it might take some time for us to do the merge)

