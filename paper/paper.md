---
title: 'UnlockNN: Uncertainty quantification for neural network models of chemical systems'
tags:
  - Python
  - graph neural networks
  - uncertainty quantification
  - machine learning
  - material science
  - chemistry
authors:
  - name: Alexander Moriarty
    orcid: 0000-0001-7525-1419
    affiliation: 1
  - name: Kazuki Morita
    orcid: 0000-0002-2558-6963
    affiliation: 1
  - name: Keith Butler
    orcid: 0000-0001-5432-5597
    affiliation: 2
  - name: Aron Walsh
    orcid: 0000-0001-5460-7033
    affiliation: "1, 3"
affiliations:
  - name: Department of Materials, Imperial College London, London, UK
    index: 1
  - name: SciML, STFC Scientific Computing Division, Rutherford Appleton Laboratories, UK
    index: 2
  - name: Department of Materials Science and Engineering, Yonsei University, Seoul, Korea
    index: 3
date: 2 September 2021
bibliography: paper.bib
---

## Summary

Machine learning models have gained interest from materials researchers for
their ability to predict materials' properties accurately and faster than
first-principles calculations based on physical laws, particularly for complex
systems with many possible configurations
[@butlerMachineLearningMolecular2018;@ramprasadMachineLearningMaterials2017;@xueAcceleratedSearchMaterials2016].
In particular, neural networks (NNs) have been shown to achieve predictive
accuracies within the threshold of "chemical accuracy"
[@chenGraphNetworksUniversal2019;@schuttSchNetDeepLearning2018;@faberPredictionErrorsMolecular2017].

However, most current implementations of NNs for materials property prediction
lack _uncertainty quantification_, a measure of the confidence of a prediction.
This is especially detrimental to a machine learning model, as its reliability
is contingent upon the existence of "similar" materials in the training data
set. To the end user, there is no easy way to tell whether this is the case.

## Statement of need

UnlockNN provides an API to add uncertainty quantification to Keras-based models
and comes packaged with a specific implementation for compatibility with MEGNet
[@chenGraphNetworksUniversal2019], which is a graph NN implementation for
materials property prediction that has achieved state-of-the-art accuracy on
many benchmark tasks [@dunnBenchmarkingMaterialsProperty2020].

This uncertainty quantification is achieved by supplanting the output layer of
the model with a variational Gaussian process (VGP)
[@dillonTensorFlowDistributions2017;@hensmanGaussianProcessesBig2013]: a
modification of a Gaussian process (GP) that allows for scalability to large
data sets. Whilst a typical GP requires the entire training data set to be
stored in memory and used for inference (an example of _instance-based_
learning), the VGP infers a smaller set of inducing index points. The locations
of these inducing index points are optimized during training to minimise the
Kullback-Leibler divergence between the GP based on _all_ training data and the
VGP.

Once created, the probabilistic model must be trained in order to optimize the
locations of the VGP's inducing index points and its kernel parameters. However,
the number of training iterations required is typically only a small fraction of
the training iterations needed to train the base NN it is modifying.

The primary interface for unlockNN is the `model` module, which contains an
extensible `ProbNN` class for adding uncertainty quantification to arbitrary
Keras models. It also contains a `MEGNetProbModel` class for adding uncertainty
quantification to MEGNet.

Neural network-fed VGPs share a similar principle to the convolution-fed
Gaussian processes formulated by @tranMethodsComparingUncertainty2020. UnlockNN
also implements tools for calculating the performance metrics suggested by
@tranMethodsComparingUncertainty2020, including sharpness and calibration error,
via its `metrics` module.

## Acknowledgements

This project was proposed by [Keith Butler](https://github.com/keeeto) and has
benefited hugely from his support, as well as the support of [Aron
Walsh](https://wmd-group.github.io/) and [Kazuki
Morita](https://github.com/KazMorita). Thanks also to the Royal Society for
funding this project and the Science and Technology Facilities Council for
providing access to their computing cluster, which was used extensively for
testing during development.

## References
