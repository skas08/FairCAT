# FairCAT: A Graph Generator for Benchmarking Fairness-Aware Graph Neural Networks

FairCAT is an extension of a graph generator GenCAT. In contrast to GenCAT, FairCAT supports sensitive attributes, which makes it an appropriate graph generator for benchmarking fairness-aware Graph Neural Networks (GNNs).

FairCAT allows users to explicitly control separate degree distributions per binary sensitive attributes. Moreover, FairCAT extends the degree distribution types that user can choose from. Aside from power-law distribution, the generator allows for normal and uniform degree distribution types. FairCAT allows users to enforce specifically-defined correlations between sensitive and non-sensitive attributes.

## Requirements
- Python 3.11.9
- numpy == 1.26.4
- powerlaw == 1.5
- scipy == 1.15.2

(for experiments)
- pandas == 2.2.3
- seaborn == 0.13.2
- psutil == 7.0.0
- matplotlib == 3.10.1
- scikit-learn == 1.7.2


## Reproduction of experiments
The repository includes the FairCAT model and demo, which shows an example graph generation. 

"Experiments" folder includes code for the experiments. The aim of these experiments is to assess the ability of FairCAT to reproduce user-specified degree distributions and attribute correlations, inspect how user inputs affect generation and output, evaluate scalability, and analyze reproducibility of real-world graphs.

"Benchmarking-datasets" includes code for generating datasets for benchmarking GNNs, generated FairCAT datasets that mimic real-life datasets, and examples of slurm files used in HPC for training GNNs.
