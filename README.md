# Awesome Machine Unlearning

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![arXiv](https://img.shields.io/badge/arXiv-2209.02299-b31b1b.svg)](https://arxiv.org/abs/2209.02299)
![GitHub stars](https://img.shields.io/github/stars/tamlhp/awesome-machine-unlearning?color=yellow&label=Stars)
![visitor badge](https://visitor-badge.glitch.me/badge?page_id=tamlhp.awesome-machine-unlearning) 

A collection of academic articles, published methodology, and datasets on the subject of **machine unlearning**.

- [Awesome-Machine-Unlearning](#awesome-machine-unlearning)
  - [ A Framework of Machine Unlearning](#a-framework-of-machine-unlearning)
  - [Surveys](#existing-surveys)
  - [Model-agnostic](#model-agnostic-approaches)
  - [Model-intrinsic](#model-intrinsic-approaches)
  - [Data-Driven](#data-driven-approaches)
  - [Datasets](#datasets)
    - [Type: Image](#type-image)
    - [Type: Tabular](#type-tabular)
    - [Type: Text](#type-text)
    - [Type: Sequence](#type-sequence)
    - [Type: Graph](#type-graph)
  - [Evaluation Metrics](#evaluation-metrics)

Please read and cite our paper: [![arXiv](https://img.shields.io/badge/arXiv-2209.02299-b31b1b.svg)](https://arxiv.org/abs/2209.02299)

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
[![timeline](framework.png)](https://arxiv.org/abs/2209.02299)

----------

## Existing Surveys
| **Paper Title** | **Venue** | **Year** | 
| --------------- | ---- | ---- | 
| [An Introduction to Machine Unlearning](https://arxiv.org/abs/2209.00939) | Arxiv | 2022 |
| [Making machine learning forget](https://www.sciencedirect.com/science/article/pii/S0267364917302091) | _Annual Privacy Forum_ | 2019 |
| [“Amnesia” - A Selection of Machine Learning Models That Can Forget User Data Very Fast](https://www.semanticscholar.org/paper/%22Amnesia%22-Machine-Learning-Models-That-Can-Forget-Schelter/4e99e7af4b9f08b0a89577cd8ea92a37d4744e1e) | _CIDR_ | 2019 |
| [Humans forget, machines remember: Artificial intelligence and the Right to Be Forgotten](https://www.sciencedirect.com/science/article/pii/S0267364917302091) | _Computer Law & Security Review_ | 2018 |
| [Algorithms that remember: model inversion attacks and data protection law](https://doi.org/10.1098/rsta.2018.0083) | _Philosophical Transactions of the Royal Society A_ | 2018 |
----------

## Model-Agnostic Approaches
[![Model-Agnostic](figs/model-agnostic.png)](https://arxiv.org/abs/2209.02299)
Model-agnostic machine unlearning methodologies include unlearning processes or frameworks that are applicable for different models. In some cases, they provide theoretical guarantees for only a class of models (e.g. linear models). But we still consider them model-agnostic as their core ideas are applicable to complex models (e.g. deep neural networks) with practical results.

| **Paper Title** | **Year** | **Author** | **Venue** | **Model** | **Code** | **Type** |
| --------------- | :----: | ---- | :----: | :----: | :----: | ---- |
| [Verifiable and Provably Secure Machine Unlearning](https://arxiv.org/abs/2210.09126) | 2022 | Eisenhofer et al. | _arXiv_ | - | [Code](https://github.com/cleverhans-lab/verifiable-unlearning) |  Certified Removal Mechanisms |
| [Machine Unlearning Method Based On Projection Residual](https://arxiv.org/abs/2209.15276) | 2022 | Cao et al. | _DSAA_ | - | - |  Projection Residual Method |
| [Hard to Forget: Poisoning Attacks on Certified Machine Unlearning](https://ojs.aaai.org/index.php/AAAI/article/view/20736) | 2022 | Marchant et al. | _AAAI_ | - | [[Code]](https://github.com/ngmarchant/attack-unlearning) | Certified Removal Mechanisms |
| [Markov Chain Monte Carlo-Based Machine Unlearning: Unlearning What Needs to be Forgotten](https://dl.acm.org/doi/abs/10.1145/3488932.3517406) | 2022 | Nguyen et al. | _ASIA CCS_ | MCU | - | MCMC Unlearning  |
| [Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks using an Incompetent Teacher](https://arxiv.org/abs/2205.08096) | 2022 | Chundawat et al. | _arXiv_ | - | - | Knowledge Adaptation |
| [Adaptive Machine Unlearning](https://proceedings.neurips.cc/paper/2021/hash/87f7ee4fdb57bdfd52179947211b7ebb-Abstract.html) | 2021 | Gupta et al. | _NeurIPS_ | - | [[Code]](https://github.com/ChrisWaites/adaptive-machine-unlearning) | Differential Privacy |
| [Descent-to-Delete: Gradient-Based Methods for Machine Unlearning](https://proceedings.mlr.press/v132/neel21a.html) | 2021 | Neel et al. | _ALT_ | - | - | Certified Removal Mechanisms |
| [Machine Unlearning via Algorithmic Stability](https://proceedings.mlr.press/v134/ullah21a.html) | 2021 | Ullah et al. | _COLT_ | TV | - | Certified Removal Mechanisms |
| [Knowledge-Adaptation Priors](https://proceedings.neurips.cc/paper/2021/hash/a4380923dd651c195b1631af7c829187-Abstract.html) | 2021 | Khan and Swaroop | _NeurIPS_ | K-prior | [[Code]](https://github.com/team-approx-bayes/kpriors) | Knowledge Adaptation |
| [PrIU: A Provenance-Based Approach for Incrementally Updating Regression Models](https://dl.acm.org/doi/abs/10.1145/3318464.3380571) | 2020 | Wu et al. | _NeurIPS_ | PrIU | - | Knowledge Adaptation |
| [Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks](https://arxiv.org/abs/1911.04933) | 2020 | Golatkar et al. | _CVPR_ | - | - | Certified Removal Mechanisms |
| [Certified Data Removal from Machine Learning Models](https://proceedings.mlr.press/v119/guo20c.html) | 2020 | Guo et al. | _ICML_ | - | - | Certified Removal Mechanisms |
| [A Novel Online Incremental and Decremental Learning Algorithm Based on Variable Support Vector Machine](https://link.springer.com/article/10.1007/s10586-018-1772-4) | 2019 | Chen et al. | _Cluster Computing_ | - | - | Decremental Learning  |
| [Making AI Forget You: Data Deletion in Machine Learning](https://papers.nips.cc/paper/2019/hash/cb79f8fa58b91d3af6c9c991f63962d3-Abstract.html) | 2019 | Ginart et al. | _NeurIPS_ | - | - | Decremental Learning  |
| [Understanding Black-box Predictions via Influence Functions](https://proceedings.mlr.press/v70/koh17a.html) | 2017 | Koh et al. | _ICML_ | - | [[Code]](https://github.com/kohpangwei/influence-release) | Certified Removal Mechanisms |
| [Towards Making Systems Forget with Machine Unlearning](https://dl.acm.org/doi/10.1109/SP.2015.35) | 2015 | Cao et al. | _S&P_ | - | - | Statistical Query Learning  |
| [Incremental and decremental training for linear classification](https://dl.acm.org/doi/10.1145/2623330.2623661) | 2014 | Tsai et al. | _KDD_ | - | [[Code]](https://www.csie.ntu.edu.tw/~cjlin/papers/ws/) | Decremental Learning  |
| [Multiple Incremental Decremental Learning of Support Vector Machines](https://dl.acm.org/doi/10.5555/2984093.2984196) | 2009 | Karasuyama et al. | _NIPS_ | - | - | Decremental Learning  |
| [Incremental and Decremental Learning for Linear Support Vector Machines](https://dl.acm.org/doi/10.5555/1776814.1776838) | 2007 | Romero et al. | _ICANN_ | - | - | Decremental Learning  |
| [Decremental Learning Algorithms for Nonlinear Langrangian and Least Squares Support Vector Machines](https://www.semanticscholar.org/paper/Decremental-Learning-Algorithms-for-Nonlinear-and-Duan-Li/312c677f0882d0dfd60bfd77346588f52aefd10f) | 2007 | Duan et al. | _OSB_ | - | - | Decremental Learning  |
| [Multicategory Incremental Proximal Support Vector Classifiers](https://link.springer.com/chapter/10.1007/978-3-540-45224-9_54) | 2003 | Tveit et al. | _KES_ | - | - | Decremental Learning  |
| [Incremental and Decremental Proximal Support Vector Classification using Decay Coefficients](https://link.springer.com/chapter/10.1007/978-3-540-45228-7_42) | 2003 | Tveit et al. | _DaWak_ | - | - | Decremental Learning  |
| [Incremental and Decremental Support Vector Machine Learning](https://dl.acm.org/doi/10.5555/3008751.3008808) | 2000 | Cauwenberg et al. | _NeurIPS_ | - | - | Decremental Learning  |
----------

## Model-Intrinsic Approaches
[![Model-Intrinsic](figs/model-intrinsic.png)](https://arxiv.org/abs/2209.02299)
The model-intrinsic approaches include unlearning methods designed for a specific type of models. Although they are model-intrinsic, their applications are not necessarily narrow, as many ML models can share the same type.
| **Paper Title** | **Year** | **Author** | **Venue** | **Model** | **Code** | **Unlearning For** |
| --------------- | :----: | ---- | :----: | :----: | :----: | ---- |
| [Machine Unlearning for Image Retrieval: A Generative Scrubbing Approach](https://dl.acm.org/doi/abs/10.1145/3503161.3548378) | 2022 | Zhang et al. | _MM_ | - | - | DNN-based Models |
| [Machine Unlearning: Linear Filtration for Logit-based Classifiers](https://link.springer.com/article/10.1007/s10994-022-06178-9) | 2022 | Baumhauer et al. | _Machine Learning_ | normalizing filtration | - | Softmax classifiers |
| [Deep Unlearning via Randomized Conditionally Independent Hessians](https://openaccess.thecvf.com/content/CVPR2022/html/Mehta_Deep_Unlearning_via_Randomized_Conditionally_Independent_Hessians_CVPR_2022_paper.html) | 2022 | Mehta et al. | _CVPR_ | L-CODEC | [[Code]](https://github.com/vsingh-group/LCODEC-deep-unlearning) | DNN-based Models |
| [Variational Bayesian unlearning](https://dl.acm.org/doi/abs/10.5555/3495724.3497068) | 2022 | Nguyen et al. | _NeurIPS_ | VI | - | Bayesian Models  |
| [Revisiting Machine Learning Training Process for Enhanced Data Privacy](https://dl.acm.org/doi/abs/10.1145/3474124.3474208) | 2021 | Goyal et al. | _IC3_ | - | - | DNN-based Models  |
| [Knowledge Removal in Sampling-based Bayesian Inference](https://openreview.net/forum?id=dTqOcTUOQO) | 2021 | Fu et al. | _ICLR_ | - | [[Code]](https://github.com/fshp971/mcmc-unlearning) | Bayesian Models  |
| [Mixed-Privacy Forgetting in Deep Networks](https://openaccess.thecvf.com/content/CVPR2021/html/Golatkar_Mixed-Privacy_Forgetting_in_Deep_Networks_CVPR_2021_paper.html) | 2021 | Golatkar et al. | _CVPR_ | - | - | DNN-based Models  |
| [HedgeCut: Maintaining Randomised Trees for Low-Latency Machine Unlearning](https://dl.acm.org/doi/abs/10.1145/3448016.3457239) | 2021 | Schelter et al. | _SIGMOD_ | HedgeCut | [[Code]](https://github.com/schelterlabs/hedgecut) | Tree-based Models  |
| [A Unified PAC-Bayesian Framework for Machine Unlearning via Information Risk Minimization](https://ieeexplore.ieee.org/abstract/document/9596170) | 2021 | Jose et al. | _MLSP_ | PAC-Bayesian| - | Bayesian Models  |
| [DeepObliviate: A Powerful Charm for Erasing Data Residual Memory in Deep Neural Networks](https://arxiv.org/abs/2105.06209) | 2021 | He et al. | _arXiv_ | DEEPOBLIVIATE | - | DNN-based Models  |
| [Bayesian Inference Forgetting](https://arxiv.org/abs/2101.06417) | 2021 | Fu et al. | _arXiv_ | BIF | [[Code]](https://github.com/fshp971/BIF) | Bayesian Models  |
| [Approximate Data Deletion from Machine Learning Models](https://proceedings.mlr.press/v130/izzo21a.html) | 2021 | Izzo et al. | _AISTATS_ | PRU | [[Code]](https://github.com/zleizzo/datadeletion) | Linear Models |
| [Online Forgetting Process for Linear Regression Models](https://proceedings.mlr.press/v130/li21a.html) | 2021 | Li et al. | _AISTATS_ | FIFD-OLS | - | Linear Models  |
| [Forgetting Outside the Box: Scrubbing Deep Networks of Information Accessible from Input-Output Observations](https://link.springer.com/chapter/10.1007/978-3-030-58526-6_23) | 2020 | Golatkar et al. | _ECCV_ | - | - | DNN-based Models  |
| [Influence Functions in Deep Learning Are Fragile](https://www.semanticscholar.org/paper/Influence-Functions-in-Deep-Learning-Are-Fragile-Basu-Pope/098076a2c90e42c81b843bf339446427c2ff02ed) | 2020 | Basu et al. | _arXiv_ | - | - | DNN-based Models  |
| [Deep Autoencoding Topic Model With Scalable Hybrid Bayesian Inference](https://ieeexplore.ieee.org/document/9121755) | 2020 | Zhang et al. | _IEEE_ | DATM | - | Bayesian Models  |
| [Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks](https://arxiv.org/abs/1911.04933) | 2020 | Golatkar et al. | _CVPR_ | - | - | DNN-based Models  |
| [Uncertainty in Neural Networks: Approximately Bayesian Ensembling](https://proceedings.mlr.press/v108/pearce20a.html) | 2020 | Pearce et al. | _AISTATS_ | - | [[Code]](https://teapearce.github.io/portfolio/github_io_1_ens/) | Bayesian Models  |
| [Certified Data Removal from Machine Learning Models](https://proceedings.mlr.press/v119/guo20c.html) | 2020 | Guo et al. | _ICML_ | - | - | DNN-based Models  |
| [DeltaGrad: Rapid retraining of machine learning models](https://proceedings.mlr.press/v119/wu20b.html) | 2020 | Wu et al. | _ICML_ | DeltaGrad | [[Code]](https://github.com/thuwuyinjun/DeltaGrad) | DNN-based Models  |
| [Making AI Forget You: Data Deletion in Machine Learning](https://papers.nips.cc/paper/2019/hash/cb79f8fa58b91d3af6c9c991f63962d3-Abstract.html) | 2019 | Ginart et al. | _NeurIPS_ | - | - | Linear Models  |
| [Bayesian Neural Networks with Weight Sharing Using Dirichlet Processes](https://ieeexplore.ieee.org/document/8566011) | 2018 | Roth et al. | _IEEE_ | DP | [[Code]](https://github.com/wroth8/dp-bnn) | Bayesian Models  |

----------

## Data-Driven Approaches
[![Data-Driven](figs/data-driven.png)](https://arxiv.org/abs/2209.02299)
The approaches fallen into this category use data partition, data augmentation and data influence to speed up the retraining process.
| **Paper Title** | **Year** | **Author** | **Venue** | **Model** | **Code** | **Type** |
| --------------- | :----: | ---- | :----: | :----: | :----: | ---- |
| [PUMA: Performance Unchanged Model Augmentation for Training Data Removal](https://ojs.aaai.org/index.php/AAAI/article/view/20846) | 2022 | Wu et al. | _AAAI_ | PUMA | - | Data Influence |
| [Certifiable Unlearning Pipelines for Logistic Regression: An Experimental Study](https://www.mdpi.com/2504-4990/4/3/28) | 2022 | Mahadevan and Mathioudakis | _MAKE_ | - | [[Code]](https://version.helsinki.fi/mahadeva/unlearning-experiments) | Data Influence |
| [Zero-Shot Machine Unlearning](https://arxiv.org/abs/2201.05629) | 2022 | Chundawat et al. | _arXiv_ | - | - | Data Influence |
| [GRAPHEDITOR: An Efficient Graph Representation Learning and Unlearning Approach](https://congweilin.github.io/CongWeilin.io/files/GraphEditor.pdf) | 2022 | Cong and Mahdavi | - | GRAPHEDITOR | [[Code]](https://anonymous.4open.science/r/GraphEditor-NeurIPS22-856E/README.md) | Data Influence |
| [Learning to Refit for Convex Learning Problems](https://arxiv.org/abs/2111.12545) | 2021 | Zeng et al. | _arXiv_ | OPTLEARN | - | Data Influence |
| [Fast Yet Effective Machine Unlearning](https://arxiv.org/abs/2111.08947) | 2021 | Ayush et al. | _arXiv_ | - | - | Data Augmentation |
| [SSSE: Efficiently Erasing Samples from Trained Machine Learning Models](https://openreview.net/forum?id=GRMKEx3kEo) | 2021 | Peste et al. | _NeurIPS_ | SSSE | - | Data Influence |
| [Coded Machine Unlearning](https://ieeexplore.ieee.org/document/9458237) | 2021 | Aldaghri et al. | _IEEE_ | - | - | Data Partitioning |
| [Machine Unlearning](https://ieeexplore.ieee.org/document/9519428) | 2021 | Bourtoule et al. | _IEEE_ | SISA | [[Code]](https://github.com/cleverhans-lab/machine-unlearning) | Data Partitioning |
| [How Does Data Augmentation Affect Privacy in Machine Learning?](https://ojs.aaai.org/index.php/AAAI/article/view/17284/) | 2021 | Yu et al. | _AAAI_ | - | [[Code]](https://github.com/dayu11/MI_with_DA) | Data Augmentation |
| [Amnesiac Machine Learning](https://ojs.aaai.org/index.php/AAAI/article/view/17371) | 2021 | Graves et al. | _AAAI_ | AmnesiacML | [[Code]](https://github.com/lmgraves/AmnesiacML) | Data Influence |
| [Unlearnable Examples: Making Personal Data Unexploitable](https://arxiv.org/abs/2101.04898) | 2021 | Huang et al. | _ICLR_ | - | [[Code]](https://github.com/HanxunH/Unlearnable-Examples) | Data Augmentation |
| [Descent-to-Delete: Gradient-Based Methods for Machine Unlearning](https://proceedings.mlr.press/v132/neel21a.html) | 2021 | Neel et al. | _ALT_ | - | - | Data Influence |
| [Fawkes: Protecting Privacy against Unauthorized Deep Learning Models](https://dl.acm.org/doi/abs/10.5555/3489212.3489302) | 2020 | Shan et al. | _USENIX Sec. Sym._ | Fawkes | [[Code]](https://github.com/Shawn-Shan/fawkes) | Data Augmentation |
| [PrIU: A Provenance-Based Approach for Incrementally Updating Regression Models](https://dl.acm.org/doi/abs/10.1145/3318464.3380571) | 2020 | Wu et al. | _SIGMOD_ | PrIU/PrIU-opt | - | Data Influence |
| [DeltaGrad: Rapid retraining of machine learning models](https://proceedings.mlr.press/v119/wu20b.html) | 2020 | Wu et al. | _ICML_ | DeltaGrad | [[Code]](https://github.com/thuwuyinjun/DeltaGrad) | Data Influence |


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
## Evaluation Metrics
| Metrics | Formula/Description | Usage |
| ---- | ---- | ---- |
| Accuracy | Accuracy on unlearned model on forget set and retrain set | Evaluating the predictive performance of unlearned model |
| Completeness | The overlapping (e.g. Jaccard distance) of output space between the retrained and the unlearned model | Evaluating the indistinguishability between model outputs |
| Unlearn time | The amount of time of unlearning request | Evaluating the unlearning efficiency |
| Relearn Time | The epochs number required for the unlearned model to reach the accuracy of source model | Evaluating the unlearning efficiency (relearn with some data sample) |
| Layer-wise Distance | The weight difference between original model and retrain model | Evaluate the indistinguishability between model parameters |
| Activation Distance | An average of the L2-distance between the unlearned model and retrained model’s predicted probabilities on the forget set | Evaluating the indistinguishability between model outputs | 
| JS-Divergence | Jensen-Shannon divergence between the predictions of the unlearned and retrained model | Evaluating the indistinguishability between model outputs |
| Membership Inference Attack | Recall (#detected items / #forget items) | Verify the influence of forget data on the unlearned model |
| ZRF score | $\mathcal{ZFR} = 1 - \frac{1}{nf}\sum_{i=0}^{n_f} \mathcal{JS}(M(x_i), T_d(x_i))$ | The unlearned model should not intentionally give wrong output $\(\mathcal{ZFR} = 0\)$ or random output $\(\mathcal{ZFR} = 1\)$ on the forget item |
| Anamnesis Index (AIN) | $AIN = \frac{r_t (M_u, M_{orig}, \alpha)}{r_t (M_s, M_{orig}, \alpha)}$ | Zero-shot machine unlearning | 
| Epistemic Uncertainty | if $\mbox{i(w;D) > 0}$, then $\mbox{efficacy}(w;D) = \frac{1}{i(w; D)}$;<br />otherwise $\mbox{efficacy}(w;D) = \infty$ | How much information the model exposes |
| Model Inversion Attack | Visualization | Qualitative verifications and evaluations | 


----------
**Disclaimer**

Feel free to contact us if you have any queries or exciting news on machine unlearning. In addition, we welcome all researchers to contribute to this repository and further contribute to the knowledge of machine unlearning fields.

If you have some other related references, please feel free to create a Github issue with the paper information. We will glady update the repos according to your suggestions. (You can also create pull requests, but it might take some time for us to do the merge)

