# Invariant regressions on point clouds for cosmology

This document describes the overall organization of this project.

Main types of directories:
 - Data
 - Code
 - Notebooks
 - Output


The main data directories are:
 - `data`: the datasets provided by Sebastian (sp?)
   * `halos.pkl`: the smallest dataset, contains 79 halos of similar mass
      - This file can be obtained from https://github.com/captain-pool/covhalo/halos.pkl, although it involves messing with github large files.
   * `halos_points_clouds.pkl.gz`: the largest dataset, contains 1,000 halos, sampled across masses. The distribution of masses in this set follows a power law.
      - This data set is available at: https://drive.google.com/file/d/1BFa3NVV8aoUTf94TYJnBEKhIQom9JTQv/view?usp=sharing
   * `halos_points_clouds_extended.pkl.gz`: the dataset that contains additional labels to the stellar mass. The additional properties in this dataset are: 'HalfMassRadr', 'StellarMetallicity', and 'StarFormRate'. This data set only contains 500 halos due to memory issues.
      - This data set is available at: https://drive.google.com/file/d/1kdGtuySDDkdzMW2BvwNvLXFvKBvXJoIf/view?usp=sharing
 - `generated_data`: intermediate data produced in the process of modeling with `halos.pkl`.
 - `generated_data_big`: intermediate data produced in the process of modeling with `halos_points_clouds.pkl.gz`
 - `generated_data_extended`: intermediate data produced in the process of modeling with `halos_points_clouds_extended.pkl.gz`

The sorts of generated datasets are as follows:
 - `features_and_targets`: The features (such as mass and concentration) and targets (such as stellar mass) of the dataset. I separated this out since it's expensive to load the whole dataset, so we are well-served by loading just the features without looking at the full point clouds. Features are `SubhaloC200` (concentration) and `Group_M_Crit200` (mass). 
 - `kmeans_subsampled_n{}_s{}.npz`: The $k$-means subsampled dataset. The number following `n` is how many trials of $k$-means that were run (this is the second axis of the resulting `numpy` array). The number following `s` is how many clusters were used (this is the third axis of the resulting array).
 - `train_indices.txt` and `test_indices.txt`: these are text files with the index of the train/test split on each line. They can be read in with `np.loadtxt`.

The `src` directory contains all `.py` files. I'll pare down the content I upload to github and describe the subdirectories below:
 - `gw`

The `notebooks` directory contains all notebooks. There is currently no organization within this directory.

Output:
 - `plots`: Where generated plots and figures live.