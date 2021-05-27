# Intelligent Machine Reliability
Code written for research during my Master's degree. 

Two major contributions are publicly available in this repository:
1. GVFOD - An outlier detection model for time series databases. Using general value functions (GVFs) and Reinforcement Learning, anomalous or outlying behavior can be found.
2. A simulator for a robot arm with a single degree of freedom. It includes framework for parallelized (multi-node) data-driven system identification. A PID controller compatible with scipy IVP solvers is also implemented. 

## Usage Instructions
These instructions are for me to guide development
### To use GVFOD
* To install this package
```shell script
cd .../GVFOD  # Navigate to the directory of this file
pip install .  # Installs this package
```

* To use this package, import the GVFOD class as follows

```python
from gvfod import GVFOD
train, test = ...  # put your data here

kwargs = {}  # See documentation for GVFOD parameters
model = GVFOD(
    contamination=0.05,
    **kwargs
) 
model.fit(train)
y = model.predict(test)

... # and so on
```


### Replicate GVFOD paper experiments
This requires the robot arm dataset to be provided. Extract the data into the directory `.../GVFOD/data/robot_arm_1`. The file containing normal data will be `.../GVFOD/data/robot_arm_1/Normal_adjusted_150.csv`

Install the package as [above](#to-use-GVFOD). 

Then, in the shell:
* Make the dataset from the raw data. This caches the cleaned data in `.../GVFOD/data/pickles/robotarm_{type}.pkl` where `type` is one of `{normal, loose_l2, loose_l1, tight, sandy, highT}`
    ```shell script 
    cd GVFOD
    python exp/make_robotarm_dataset.py data/robotarm data/pickles    
    ```

* Parameter tuning / search
    ```shell script 
    python -m GVFOD.gvfod_param_search --runs 400 
    ```
  This will place all the search information in the folder `.../GVFOD/exp/exp_gvfod/tuning_results/`
  
  Since search is stochastic, and results will differ between seed, operating system, python, and dependency versions, the best parameters are hand-copied into `.../GVFOD/exp/exp_gvfod/exp_train_size_settings.py`

* Experiments 
    ```shell script
    python -m GVFOD.gvfod_exp
    ```
    Results will be placed into `.../GVFOD/exp/exp_gvfod/results`

* Visualization
    ```shell script
    python -m GVFOD.gvfod_vis  
    ```
    Figures will be placed into `.../GVFOD/exp/exp_gvfod/figures`
### Replicate simulator paper experiments
In progress