---
title: 'UnlockGNN: Uncertainty quantification for graph neural network models of chemical systems'
tags:
  - Python
  - graph neural networks
  - uncertainty quantification
  - machine learning
  - materials
  - MEGNet
authors:
  - name: Alexander Moriarty
    orcid: 0000-0001-7525-1419
    affiliation: 1
  - name: Keith Butler
    orcid: 0000-0001-5432-5597
    affiliation: 2
  - name: Aron Walsh
    orcid: 0000-0001-5460-7033
    affiliation: "1, 3"
  - name: Kazuki Morita
    orcid: 0000-0002-2558-6963
    affiliation: 1
affiliations:
  - name: Department of Materials, Imperial College London, London, UK
    index: 1
  - name: SciML, STFC Scientific Computing Division, Rutherford Appleton Laboratories, UK
    index: 2
  - name: Department of Materials Science and Engineering, Yonsei University, Seoul, Korea
    index: 3
date: 29 September 2020
bibliography: paper.bib
---

## Summary

Machine learning models have gained interest from materials researchers for their ability to predict materials'
properties accurately and faster than first-principles calculations based on physical laws, particularly
for complex systems with many possible configurations [@butler_machine_2018;@ramprasad_machine_2017;@xue_accelerated_2016].

Graph neural networks (GNNs), a subset of the graph networks proposed by @battaglia_relational_2018, provide a data-driven
approach for learning materials' properties based on crystal or molecular _structure_.
This approach is both general and interpretable [@xie_crystal_2018;@chen_graph_2019], which addresses a major criticism
of machine learning techniques as "black boxes" [@schmidt_recent_2019].

However, current implementations of GNNs lack uncertainty quantification, a measure of the confidence of a prediction.
This is especially detrimental to a data-driven model, as its reliability is contingent upon the existence of "similar"
materials in the training data set. To the end user, there is no easy way to tell whether this is the case.

## Statement of need

`UnlockGNN` contains utilities for training a neural network-fed Gaussian process as an uncertainty quantifier.
The framework enables the training of a precursor GNN, which functions as a representation learning algorithm.
A layer of the GNN can then be selected to serve as the input (index points) for a Gaussian process.
The model can be saved and reloaded in a bundled format and used to perform predictions and confidence intervals
on unseen structures.

Graph network-fed Gaussian processes share a similar principle to the convolution-fed Gaussian processes
formulated by @tran_methods_2020 and `UnlockGNN` provides tools for calculating the performance metrics
suggested by them, including sharpness and calibration error.

`UnlockGNN` was designed for use with `Keras` [@chollet2015keras] implementations of graph neural networks.
Expanded functionality is provided for `MEGNet` [@chen_graph_2019], a high performing, modular architecture for
graph network-based modelling of materials.

## Acknowledgements

This project was proposed by [Keith Butler](https://github.com/keeeto) and has benefitted hugely from his support,
as well as the support of [Aron Walsh](https://wmd-group.github.io/) and [Kazuki Morita](https://github.com/KazMorita).
This project was supported by the Royal Society.

## References
