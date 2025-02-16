### Running Instruction

Prepare Environment:
```bash
$ conda env create -n "gromov" -f environment.yml
$ conda activate gromov
```

### Running Notebook:

Launch Jupyter Notebook from the project root directory, and use a browser to visit the jupyter notebook.

```bash
$ juypter notebook --ip=0.0.0.0
```


Now navigate to and runn notebook [notebooks/run_pipeline_with_subsample.ipynb](notebooks/run_pipeline_with_subsample.ipynb)

##### NOTE:
Update the `CUDA_VISIBLE_DEVICES` based on the number of GPUs available on the system.
