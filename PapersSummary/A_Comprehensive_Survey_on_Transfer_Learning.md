# A Comprehensive Survey on Transfer Learning

**Link:** https://arxiv.org/abs/1911.02685

### Key ideas

It is important to make a difference between transfer learning and other related work such as Semi-Supervised Learning, Mulit-View Learning and Multi-Task Learning.

**Semi-Supervised Learning:** In Semi-Supervised Learning, both the labeled and unlabeled data are drawn from the same distribution. However, in Transfer Learning, the distributions of the source and target domains are usually different.

**Multi-View Learning:** Briefly speaking, Multi-View Learning describes an object from different views (e.g. a video from the point of view of the frames and from the sound one). This approach is used in some Transfer Learning strategies.

**Multi-Task Learning:** The goal of Multi-Task Learning consists in to jointly learn a group of related tasks. The main difference between transfer learning and multi- task learning is that the former transfer the knowledge contained in the related domains, while the latter transfer the knowledge via simultaneously learning some related tasks.

#### Definitions

**Domain:** A domain D is composed of two parts, i.e., a feature space X and a marginal distribution P(X). In other words, D = {X,P(X)}. And the symbol X denotes an instance set, which is defined as X = {x|xi ∈ X, i = 1,··· ,n}.
Note that the marginal distribution P (X ) is generally an invisible component, and it is hard to obtain its explicit formulation.

**Task:** A task T consists of a label space Y and a decision function f, i.e., T = {Y,f}. The decision function f is an implicit one, which is expected to be learned from the sample data.

Some machine learning models actually output the prediicted conditional distributions of instances. In this case, f(xj) = {P(yk|xj)|yk ∈ Y,k = 1,··· ,|Y|}.

**Transfer Learning:** Given some/an observation(s) corresponding to mS ∈ N+ source domain(s) and task(s) (i.e., {(DSi , TSi )|i = 1, · · · , mS }), and some/an observation(s) about mT ∈ N+ target domain(s) and task(s) (i.e., {(DTj , TTj )|j = 1, · · · , mT }), transfer learning utilizes the knowledge implied in the source domain(s) to improve the performance of the learned decision functions fTj (j = 1, · · · , mT ) on the target domain(s).

#### Categorization

There are many different categorizations for Transfer Learning.

A first categorization consists in:

**Transductive:** Label information comes from source domain.
**Inductive:** Label information from the target domain is available.
**Unsupervised Transfer Learning:** Label information unknown for both the domain and target domains.

Another categorization present in the literature is:

**Instance-based:** Instance weighting strategy.
**Feature-based:** Transform original features to create new feature representations.

- **Asymmetric:** Transform source features to fit those in the target domain.
- **Symmetric:** Attempt to find a common latent feature space in order to transform both the source and target domain to into it.

**Parameter-based:** Approaches the transfer of learning in the model/parameters level.
**Relational-based:** Focus on problems with relational domains.

#### Transfer Learning interpretations

- **Data-based interpretations:** Transfer the knowledge via the adjustment and transformaiton of the data. The main objective is to reduce the difference of the distributions of the data. There are mainly two data-based strategies:

  - **Instance Weighting Strategy**: The idea consists in assigning weights to the source-domain instances. Such weights is could be calculated dividing the marginal probabilities of each dimension from the target-domain over the source-domain.
    However, this could be very expensive so some approximations exist to deal that such calculation such as: [Kernel Mean Matching (KMM)](https://papers.nips.cc/paper/3075-correcting-sample-selection-bias-by-unlabeled-data.pdf), [Kullback-Leibler Importance Estimation Procedure (KLIEP)](https://www.ism.ac.jp/editsec/aism/pdf/060_4_0699.pdf), [2-Stage Weighting Framework for Multi-Source Domain Adaptation (2SW-MDA)](https://papers.nips.cc/paper/4195-a-two-stage-weighting-framework-for-multi-source-domain-adaptation).
    Instead of calculating the approximation of the weights directly, it has been tried to calculate the weights iteratively: [TrAdaBoost](http://www.cs.ust.hk/~qyang/Docs/2007/tradaboost.pdf), [Multi-Source TrAdaBoost (MsTrAdaBoost)](https://ieeexplore.ieee.org/document/5539857).
    There is even some work trying to find these weights in a heuristic way: [Instance Weighting for Domain Adaptation in NLP](http://sifaka.cs.uiuc.edu/czhai/pub/acl07.pdf)
  - **Feature Mapping Strategy**: Feature-based approaches transform each original feature into a new feature representation for transfer learning. The objectives of constructing a new feature representation include minimizing the marginal and the conditional distribution difference, preserving the properties or the potential structures of the data, and finding the correspondence between features. The operations of feature transformation can be divided into three types:
    - **Feature augmentation**: Specially used in symmetric feature-based approaches. The objective consist in transforming the feature space into a feature space in a higher dimension.
    - **Feature reduction**:
      - **Feature mapping**: The goal is to extract features in the same way it is done in traditionaly Machine Learning using mapping-based methods such as PCA. However, this methods focus on data variance and not on the distributions difference.
      - **Feature clustering**: Feature clustering aims to find a more abstract feature representation of the original features.
      - **Feature selection**: Feature selection is another kind of operation for feature reduction, which is used to extract the pivot features. The pivot features are the ones that behave in the same way in different domains. Due to the stability of these features, they can be used as the bridge to transfer the knowledge.
      - **Feature encoding**: Feature encoding attempts to encode the instances into abstracter features by encoding them. Autoencoders are widely used. However, because of the computational cost, there is some work that tries to used linear Autoencoders.
    - **Feature alignment**: Note that feature augmentation and feature reduction mainly focus on the explicit features in a feature space. In contrast, in addition to the explicit features, feature alignment also focuses on some implicit features such as the statistic features and the spectral features.

- **Model-based interpretations:** Transfer learning approaches can also be interpreted from the model perspective. It is important noting that a Transfer Learnign model may consist of sub-modules such as encoder, classifiers, etc.
  - **Model Control Strategy:** This strategy consist in applying regularizers to the model in order to transfer the knowledge already contained in the pre-obtained source models.
  - **Parameter Control Strategy:** This strategy is focused on the parameters of the models as in these parameters, the knowledge of the model is represented.
    - **Parameter Sharing:** An intuitive way of controlling the parameters is to directly share the parameters of the source learner to the target learner. Parameter sharing is widely employed especially in the network-based approaches. For example, if we have a neural network for the source task, we can freeze (or say, share) most of its layers and only finetune the last few layers to produce a target network
    - **Parameter Restriction:** Another parameter-control-type strategy is to restrict the parameters. Different from the parameter sharing strategy that enforces the models share some parameters, parame- ter restriction strategy only requires the parameters of the source and the target models to be similar.
  - **Model Ensemble Strategy:** This strategy aims to combine a number of weak classifiers to make the final predictions.
  - **Deep Learning Technique:** Many researchers utilize the deep learning techniques to construct transfer learning models. Deep Learning approaches can be further divided in two types.
    - **Traditional Deep Learning:** This approach is based on the use of Autoencoders.
    - **Adversarial Deep Learning:** This approach is based on the use of GANs in which the discriminator tries to detect whether a instance comes from the source or target domain.
