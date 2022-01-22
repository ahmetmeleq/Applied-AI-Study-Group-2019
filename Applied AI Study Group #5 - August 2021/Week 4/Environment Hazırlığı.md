```
conda create -n inzva python=3.8
conda activate inzva

conda install pandas
pip install jupyterlab
pip install scikit-learn
pip install imbalanced-learn
pip install lightgbm
pip install optuna
pip install ax-platform
pip install mlflow
pip install nltk
pip install plotnine
pip install 'plotnine[all]'
pip install tensorboardX
pip install 'ray[default]'
pip install "aioredis<2.0.0"
```

Then,

On Linux:

```
pip install https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-2.0.0.dev0-cp38-cp38-manylinux2014_x86_64.whl
```

On Windows:

```
pip install https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-2.0.0.dev0-cp38-cp38-win_amd64.whl
```

On MacOS:

```
pip install https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-2.0.0.dev0-cp38-cp38-macosx_10_13_x86_64.whl
```

Download the following datasets (Unzip if necessary):

* https://www.kaggle.com/ozlerhakan/spam-or-not-spam-dataset
* https://www.kaggle.com/c/santander-customer-transaction-prediction/
* https://www.kaggle.com/jackdaoud/marketing-data
