## Applied AI Study Group Week 5

Welcome to Week 5 of Applied AI Study Group! This week's content:

- Going over classical machine learning terms
- Data Processing: Missing Data and Class Imbalance
- Data Processing 2: Feature Selection and Filtering
- Hyperparameter Search and UI Platforms
- AutoML: AI for everyone at its simplicity

We won't be focusing on statistical or mathematical grounds of these techniques. Instead, our target is primarily the application of AI.

### Installation

The material of this week has a lot of prerequired libraries. Hence we strongly recommend installing them via either a virtual or a conda environment.

Always run these commands when you are in the Week 5 (this) path.

First, you create an environment:

```sh
# For conda environment:
conda create -n inzva-applied-ai-week5

# For virtual environment:
virtualenv env
```

Every time you work on notebooks, you need to activate it:

```sh
# For conda environment:
conda activate inzva-applied-ai-week5

# For virtual environment on Linux/MacOS:
source env/bin/activate

# For virtual environment on Windows (PowerShell):
.\env\Scripts\activate.ps1
```

You can verify your Python version:

```sh
python -V
# Example output:
# Python 3.8.5
```

After you activated your environment, install the required libraries:

```sh
pip install -r requirements.txt
```

This might take a while.

Please note that you might be required to install standalone ray wheels: [https://docs.ray.io/en/latest/installation.html](https://docs.ray.io/en/latest/installation.html)

### Clean-up

Later, if you want to remove conda environment to free up space in your computer, type:

```sh
# For conda environment:
conda env remove -n inzva-applied-ai-week5

# For virtual environment on Linux/MacOS:
rm -r env

# For virtual environment on Windows (PowerShell):
rd -r env
```

### Troubleshooting

#### 1. _Virtualenv won't be activated in Windows._

The cause of this problem is PowerShell's execution policy. Latest versions do not raise an error. However, if you are facing this issue, just open a PowerShell console in Administrator mode (right click > Run as Administrator) and then type this:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned
```

Press `A` and `Enter` to allow it. Then restart your PowerShell in normal mode. Now, it will work.

#### 2. _My virtualenv is used to work, but now it doesn't._

The best solution is to setup a new one just to be sure everything is working clearly. Conda environments are encouraged for its system-wide handle.

One reason might be that Conda environments are path-independent whereas virtualenv's are path-dependent. If you move your working folder somewhere else or change a name of a directory, it will stop working.
