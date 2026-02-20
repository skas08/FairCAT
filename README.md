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

### Experiments used for testing FaircAT

"Experiments" folder includes code for the experiments. The aim of these experiments is to assess the ability of FairCAT to reproduce user-specified degree distributions and attribute correlations, inspect how user inputs affect generation and output, evaluate scalability, and analyze reproducibility of real-world graphs.

- "topology_validation.ipynb": includes most code to experiments in Chapter 5. First, it measures Community-related statistics (Section 5.6.2). Second, it shows MAPE experiments (5.2 Target degrees) for various sensitive group balance and degree distribution types. Third, it includes code for the Impact of User-Specified Inputs (Section 5.4) that investigates how attribute distribution and maximum degree constraint affect time, memory (and MAPE). Lastly, it includes an example scaling experiment, which was done on HPC instead of local jupyter notebook (Section 5.5).
- "scaling_faircat.py" was run on HPC to test scaling of FairCAT (Section 5.5).
- "stats_utils.py" includes functions definitions that are used in Section 5.6.2. It consists of inter- and intra-connection density calculation, the size of LLC calculation, and characteristic path length.
- "scaling_norm.py" and "scaling_bern.py" are used for Section 5.3 Target Correlation section. They test normal and Bernoulli distributions, respectivelly.
- "scaling_max_deg.py" measures the impact of the maximum degree contraing (Section 5.4.2)
- "Pokec-dataset.ipynb" and "German-credit.ipynb" first analyze characteristics of the real-world Pokec_n and German Credit datasets, respectivelly. They measure the connectivity statistics of original and FairCAT-generated graphs reproducing the original ones. Later, it compares degree distribution and correlations between original and generated graphs.

### Benchmarking GNNs

"Benchmarking-datasets" includes code for generating datasets for benchmarking GNNs, generated FairCAT datasets that mimic real-life datasets, and examples of slurm files used in HPC for training GNNs.

The user can use the dataset by first generating or downloading the datasets that want the GNN to be trained on. They must clone FairGraphBase and paste the datasets in the data folder of FairGraphBase repo. Then they can run FairGraphBase on HPC by running the slurm files that are provided here. If users want another setting, for example they want FairGNN SAGE model on faircat_balancing_strong_imbalance, they must replace the terms in slurm file: FairGNN instead of Vanilla, SAGE instead of GCN, and strong_imbalance instead of balanced.

The FairGraphBase repository: Sasaki, Y. (2025). FairGraphBase. GitHub. https://github.com/yuya-s/MUSUBI-FairGraphBase.

- "FairGraphBase_modified" folder includes a dataset file that was modified to accept FairCAT-generated graphs. The user who wants FairGraphBase to load a FairCAT-generated graph should replace MUSUBI-FairGraphBase/utils/dataset with this file.
- "balancing_tests" folder includes "balancing-save.py" which was used for generating datasets used for balance comparisons in Section 6.2. The other file, "faircat_balancing_balanced_vanilla_gcn.slurm" is a Slurm file that is used for running FairGraphBase on balancing datasets. The other slurm files were a combination of vanilla, fairgnn and nifty GNNs and gcn and sage encoders. The only changes in the rest of Slurm files were the differences in fairness-aware GNN, encoder setting, and dataset setting. The balancing datasets are called faircat_balancing_balanced, faircat_balancing_mild_imbalance, and
  faircat_balancing_strong_imbalance.
- "correlations_tests" refer to Section 6.3 Impact of strength of correlations between sensitive and nonsensitive
  attributes on downstream GNN learning. It includes a code for generating datasets "faircat_correlations_low", "faircat_correlations_medium", and "faircat_correlations_high". Again, it includes the Slurm file for Vanilla GCN model that is trained on high correlations graph.
- "scaling_tests" refer to code used for Section 6.4 Impact of increasing graph size on downstream GNN learning. The generated graphs are increasing in size: Small (number of nodes=2^15), Medium (2^20), Large (2^23). Again, there is a Slurm file for Vanilla GCN model for small graph.
- "pokec_n_faircat" is used for Pokec-n reproduction.
- "german_faircat" is used for German Credit reproduction.

## Datasets

FairCAT-generated datasets that are used for benchmarking GNNs can be found on: Skardova, S. (2026). FairCAT-generated datasets for benchmarking fairness-aware GNNs [Data set]. Zenodo. https://doi.org/10.5281/zenodo.18716538.

Real-life Pokec-n dataset is available here: https://github.com/yuya-s/MUSUBI-FairGraphBase/tree/main/data/pokec_n

Real-life German Credit is available here: https://github.com/chirag-agarwall/nifty/tree/main/dataset/german

## Dataset statistics
To support reproducibility, we show the Statistics table that show properties of the datasets used for benchmarking GNNs.
### Balancing Experiment
| Statistic | Balanced | Mild | Strong |
| :--- | :--- | :--- | :--- |
| Nodes in s=0 ($n_0$) | 32,768 (50.00%) | 45,875 (70.00%) | 58,982 (90.00%) |
| Nodes in s=1 ($n_1$) | 32,768 (50.00%) | 19,660 (30.00%) | 6,553 (10.00%) |
| Total Degree s=0 ($\Theta_0$) | 1,238,184 | 1,748,662 | 2,236,557 |
| Total Degree s=1 ($\Theta_1$) | 1,237,040 | 749,488 | 241,567 |
| Avg. Degree s=0 ($\bar{\theta}_0$) | 37.79 | 38.12 | 37.92 |
| Avg. Degree s=1 ($\bar{\theta}_1$) | 37.75 | 38.12 | 36.86 |
| Class Dist. s=0, C=0 | 22,876 (69.81%) | 32,176 (70.14%) | 41,377 (70.15%) |
| Class Dist. s=0, C=1 | 9,892 (30.19%) | 13,699 (29.86%) | 17,605 (29.85%) |
| Class Dist. s=1, C=0 | 9,806 (29.93%) | 5,906 (30.04%) | 2,023 (30.87%) |
| Class Dist. s=1, C=1 | 22,962 (70.07%) | 13,754 (69.96%) | 4,530 (69.13%) |

---

### Target Correlation Strength Experiment
| Statistic | Low | Medium | High |
| :--- | :--- | :--- | :--- |
| Nodes in s=0 ($n_0$) | 32,768 (50.00%) | 32,768 (50.00%) | 32,768 (50.00%) |
| Nodes in s=1 ($n_1$) | 32,768 (50.00%) | 32,768 (50.00%) | 32,768 (50.00%) |
| Total Degree s=0 ($\Theta_0$) | 1,245,387 | 1,242,854 | 1,248,365 |
| Total Degree s=1 ($\Theta_1$) | 1,226,607 | 1,261,938 | 1,228,137 |
| Avg. Degree s=0 ($\bar{\theta}_0$) | 38.01 | 37.93 | 38.10 |
| Avg. Degree s=1 ($\bar{\theta}_1$) | 37.43 | 38.51 | 37.48 |
| s=0, C=0 | 22,959 (70.07%) | 22,886 (69.84%) | 22,830 (69.67%) |
| s=0, C=1 | 9,809 (29.93%) | 9,882 (30.16%) | 9,938 (30.33%) |
| s=1, C=0 | 9,815 (29.95%) | 9,798 (29.90%) | 9,911 (30.25%) |
| s=1, C=1 | 22,953 (70.05%) | 22,970 (70.10%) | 22,857 (69.75%) |

---

### Scaling Experiment
| Statistic | Small | Medium | Large |
| :--- | :--- | :--- | :--- |
| Nodes in s=0 ($n_0$) | 512 (50.00%) | 16,384 (50.00%) | 131,072 (50.00%) |
| Nodes in s=1 ($n_1$) | 512 (50.00%) | 16,384 (50.00%) | 131,072 (50.00%) |
| Total Degree s=0 ($\Theta_0$) | 31,790 | 1,069,863 | 8,599,263 |
| Total Degree s=1 ($\Theta_1$) | 31,252 | 1,074,241 | 8,559,167 |
| Avg. Degree s=0 ($\bar{\theta}_0$) | 62.09 | 65.30 | 65.61 |
| Avg. Degree s=1 ($\bar{\theta}_1$) | 61.04 | 65.57 | 65.30 |
| s=0, C=0 | 356 (69.53%) | 11,498 (70.18%) | 91,808 (70.04%) |
| s=0, C=1 | 156 (30.47%) | 4,886 (29.82%) | 39,264 (29.96%) |
| s=1, C=0 | 167 (32.62%) | 4,848 (29.59%) | 39,372 (30.04%) |
| s=1, C=1 | 345 (67.38%) | 11,536 (70.41%) | 91,700 (69.96%) |

---

### Real vs FairCAT-Generated Datasets Comparison Experiments
| Statistic | Pokec-n (Original) | Pokec-n (FairCAT) | German (Original) | German (FairCAT) |
| :--- | :--- | :--- | :--- | :--- |
| Nodes in s=0 ($n_0$) | 4,040 (65.32%) | 4,040 (65.32%) | 690 (69.00%) | 690 (69.00%) |
| Nodes in s=1 ($n_1$) | 2,145 (34.68%) | 2,145 (34.68%) | 310 (31.00%) | 310 (31.00%) |
| Total Degree s=0 ($\Theta_0$) | 19,259 | 19,686 | 17,484 | 19,785 |
| Total Degree s=1 ($\Theta_1$) | 11,383 | 11,636 | 7,486 | 7,537 |
| Avg. Degree s=0 ($\bar{\theta}_0$) | 4.77 | 4.87 | 25.34 | 28.67 |
| Avg. Degree s=1 ($\bar{\theta}_1$) | 5.31 | 5.42 | 24.15 | 24.31 |
| s=0, C=0 | 2,288 (56.63%) | 2,258 (55.89%) | 191 (27.68%) | 198 (28.70%) |
| s=0, C=1 | 1,752 (43.37%) | 1,782 (44.11%) | 499 (72.32%) | 492 (71.30%) |
| s=1, C=0 | 1,144 (53.33%) | 1,129 (52.63%) | 109 (35.16%) | 118 (38.06%) |
| s=1, C=1 | 1,001 (46.67%) | 1,016 (47.37%) | 201 (64.84%) | 192 (61.94%) |
