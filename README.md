# SSR: Semi-supervised Soft Rasterizer for single-view 2D to 3D Reconstruction
## Accepted at ICCV2021 Workshop [[Paper]](https://arxiv.org/abs/2108.09593)

<p align="center" width="100%">
<img width="75%" src="scripts/highlights2.png">
</p>

### 0. Download the Dataset

```
bash scripts/download_dataset.sh
```

### 1. Install requirements

`pip install -r requirements.txt` 

and the `SoftRas` module

```
git clone https://github.com/ShichenLiu/SoftRas
python setup.py install
```

### 2. Train and Validate

```python
python trainval.py -e ubl -sb ../results -d $DATA -r 1
```

Argument Descriptions:
```
-e  [Experiment group to run like 'vae' (the rest of the experiment groups are in exp_configs/main_exps.py)] 
-sb [Directory where the experiments are saved]
-r  [Flag for whether to reset the experiments]
-d  [Directory where the datasets are aved]
```

### 3. Visualize the Results

Follow these steps to visualize plots. Open `results.ipynb`, run the first cell to get a dashboard like in the gif below, click on the "plots" tab, then click on "Display plots" or the "images" tab and then click on "Display Images". Parameters of the plots can be adjusted in the dashboard for custom visualizations.

<p align="center" width="100%">
<img width="100%" src="https://raw.githubusercontent.com/haven-ai/haven-ai/master/docs/vis.gif">
</p>


## Cite

```
@article{laradji2021ssr,
  title={SSR: Semi-supervised Soft Rasterizer for single-view 2D to 3D Reconstruction},
  author={Laradji, Issam and Rodr{\'\i}guez, Pau and Vazquez, David and Nowrouzezahrai, Derek},
  journal={arXiv preprint arXiv:2108.09593},
  year={2021}
}
```
