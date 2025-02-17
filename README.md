### Running Instruction

You need Conda to prepare the environment. In case you don't have Conda, it can be downloaded from: https://docs.anaconda.com/miniconda/install/#quick-command-line-install


#### Prepare Environment:
```bash
$ conda env create -n "gromov" -f environment.yml
$ conda activate gromov
```

#### Running Notebook:

Launch Jupyter Notebook from the project root directory, and use a browser to visit the jupyter notebook.

```bash
$ juypter notebook --ip=0.0.0.0
```


Now from the jupyter notebook UI navigate to [notebooks/] and run notebook [run_pipeline_with_subsample.ipynb](notebooks/run_pipeline_with_subsample.ipynb)


#### Data
Place [halo_pointclouds_extended.pkl.gz](https://drive.google.com/file/d/1EKXV2h4Lzk_XS-xG0qVC7_HryP81NhFi/view?usp=sharing) And [halo_pointclouds.pkl.gz](https://drive.google.com/file/d/1oyjjtloTuMEsAipcEDAIQfyAQ1idG29X/view?usp=sharing) in the `data/` folder in the project root. (Do not extract the .gz file to .pkl)


## NOTE:
Update the `CUDA_VISIBLE_DEVICES` based on the number of GPUs available on the system.
