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
properties accurately and faster than first-principles calculations based on physical laws [@ramprasad_machine_2017].

## Statement of need

Graph neural networks (GNNs) are powerful tools for performing materials property prediciton based on structural information.
They offer a cheaper alternative to DFT models and are therefore promising for high throughput screening of materials.
However, current implementations of GNNs lack uncertainty quantifiers for regression problems.
Knowledge of the certainty in an estimate is particularly important for data-driven predictive models,
as the reliability of a prediction depends on the existence of functionally similar structures in the
training dataset, which cannot be readily determined.

`UnlockGNN` contains utilities for training a neural network-fed Gaussian process as an uncertainty quantifier.
The framework enables the training of a precursor GNN, which functions as a representation learning algorithm.
A layer of the GNN can then be selected to serve as the input (index points) for a Gaussian process.
The model can be saved and reloaded in a bundled format and used to perform predictions and confidence intervals
on unseen structures.
