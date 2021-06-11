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
    git clone https://github.com/andywong36/GVFOD.git
    cd GVFOD  # Navigate to the directory of this file
    pip install .  # Installs this package
    ```
    If the intention is to run the experiments, use the following install instead:
    ```shell script
    pip install -e .[exp]
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
This requires the robot arm dataset to be provided. Extract the data into the directory `data/robotarm`. The file containing normal data will be `data/robotarm/Normal_adjusted_150.csv`

Install the package as [above](#to-use-GVFOD). 

Then, in the shell:
* Make the dataset from the raw data. This caches the cleaned data in `data/pickles/robotarm_{type}.pkl` where `type` is one of `{normal, loose_l2, loose_l1, tight, sandy, highT}`
    ```shell script 
    cd GVFOD
    python exp/make_robotarm_dataset.py data/robotarm data/pickles    
    ```

* Parameter tuning / search
    ```shell script 
    python -m exp.exp_gvfod.tuning gvfod_exp > exp/exp_gvfod/logs/exp_gvfod.log 2>&1
    ```
  This will place all the search information in the folder `exp/exp_gvfod/tuning_results/`. The experiment name `gvfod_exp` needs to be replaced for every algorithm that is to be tested. The experiments and their names are all defined in `exp/exp_gvfod/tuning_settings.py`.
  
  Since search is stochastic, and results will differ between random state, operating system, python and dependency versions, the best parameters are hand-copied into `.../GVFOD/exp/exp_gvfod/exp_train_size_settings.py`

* Experiments 
    
    The flags `--no-default-param` and `--default-param` refer to optimized (hyperopt) and hand-selected parameters 
    respectively. `--tuning-data` and `--no-tuning-data` refer to the half of the data that was used. 
    ```shell script
    python -m exp.exp_gvfod.train_size --no-default-param --tuning-data exp/exp_gvfod/results 
    python -m exp.exp_gvfod.train_size --no-default-param --no-tuning-data exp/exp_gvfod/results
    python -m exp.exp_gvfod.train_size --default-param --tuning-data exp/exp_gvfod/results
    python -m exp.exp_gvfod.train_size --default-param --no-tuning-data exp/exp_gvfod/results
    ```
    Results will be placed into `exp/exp_gvfod/results`. These commands generate the results in the order of the plots 
    in the order found in the paper.  

* Visualization
    ```shell script
    python -m exp.exp_gvfod.vis_train_size exp/exp_gvfod/results  
    ```
    Figures will be placed into the same folder
* Runtime estimates
    ``` shell script
    python -m exp.exp_gvfod.runtime test --lag 0 --no-default-param --no-tuning-data exp/exp_gvfod/runtime_results
    python -m exp.exp_gvfod.runtime summary exp/exp_gvfod/runtime_results
    ```
* Statistical tests
    
    The statistical results are calculated using a paired sample t-test, for one training size, and for one (of the four) experiments. 
    ```shell script
    python -m exp.exp_gvfod.t_test --train 2000 exp/exp_gvfod/results/train_size_delay_0_default_False_trainloss_True.json
    ```

### Replicate simulator paper experiments
In progress