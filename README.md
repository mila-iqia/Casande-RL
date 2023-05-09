# Casande

This is the repository containing the codebase of the CASANDE agent as well as the baselines introduced in the paper [Towards Trustworthy Automatic Diagnosis Systems by Emulating Doctors' Reasoning with Deep Reinforcement Learning](https://arxiv.org/abs/2210.07198) published at NeurIPS 2022.

## Introduction

Implementation of an RL-based medical conversational agent. The agent uses deep Q-learning based Rainbow to learn to interact with a patient and it inquires about potential symptoms and antecedents that the patient may be suffering from. The agent produces a distribution over possible pathologies at the end of interaction.


For a quick walk-thorugh on how to get started, skip to [Installation](#installation) section.


## Repo Structure
```
.
├── Baselines
├── chloe
│   └── agents
│   └── evaluator
│   └── models
│   └── plotting
│   └── preprocessing
│   └── pretraining
│   └── simulator
│   └── trainer
│   └── utils
│   └── eval.py
│   └── eval_fly.py
│   └── main_rl.py
│   └── pretrain.py
├── data
├── run_configs
├── scripts
├── notebooks
├── tests
├── Dockerfile
├── LICENCE
├── README.md
├── app.yml
├── job.yml
├── poetry.lock
├── pyproject.toml
└── setup.cfg
```

## Installation

### Dev environnement
We use [poetry](https://github.com/python-poetry/poetry) to manage our dependencies.

To setup the project on your machine, do the following:

1. [install poetry](https://github.com/python-poetry/poetry#installation)
2. `git clone https://github.com/mila-iqia/Casande-RL.git`
3. `cd Casande-RL` 
4. `poetry install`


## Training

To reproduce results in the paper, download the data from [link](https://figshare.com/articles/dataset/DDXPlus_Dataset/20043374) and put it in `./data`.

### Training Model Configuration

The training pipeline in this project relies on the [rlpyt](https://rlpyt.readthedocs.io/en/latest/index.html) framework from [BAIR](https://bair.berkeley.edu/blog/2019/09/24/rlpyt/). The pipeline is made in such a way that it is easy to configure rlpyt's related concepts using config files. 

An example of such a configuration file can be found at [`./run_configs/config1.yaml`](run_configs/config1.yaml).

As noticed, it is possible to specify the following elements:
   - **optimizer**: the optimizer to be used
   - **architecture**: the model architecture to be used
   - **exp_name**: the experiment name
   - **n_steps**: the number of training steps
   - **n_envs**: the number of environments to simultaneously use during training
   - **eval_n_envs**: the number of environments to use during validation/evaluation phase
   - **eval_max_steps**: the maximum number of steps during each evaluation phase
   - **eval_max_trajectories**: the maximum number of trajectories during each evaluation phase
   - **max_decorrelation_steps**: the number of steps to perform before effectively collecting samples for training
   - **log_interval_steps**: the number of steps between two successive logging phases.
   - **runner**: the rlpyt runner to be used
   - **sampler**: the rlpyt sampler to be used
   - **algo**: the algorithm to be used
   - **agent**: the type of agent to be used
   - **simulator_params**: the parameters needed to instantiate the simulator
   - **reward_config**: the configuration of the reward function to be used
   - **architecture_params**: the parameters of the selected architecture
   - **optimizer_params**: the parameters of the selected optimizer
   - **sampler_params**: the parameters of the selected sampler
   - **algo_params**: the parameters of the selected algorithm
   - **agent_params**: the parameters of the selected agent
   - **runner_params**: the parameters of the selected runner. It additionnally allows to define **custom metrics** that one wants to monitored. A custom metric is defined as a aggregation of individual metrics, and therefore, wieight of such individual metrics are provided
   - **eval_metrics**: the list of metrics of interest to be monitored. Those metric can be viewed as individual metrics computed on each trajectory.
   - **perf_metric**: the main metric that will guide early stopping. It can be one of the **custom metrics** or **eval_metrics** defined above.
   - **perf_window_size**: the window size along which the performance metric is aggregated
   - **patience**: early stopping patience

Please refer to the [rlpyt documentation](https://rlpyt.readthedocs.io/en/latest/index.html) for more details regarding rlpyt related concepts.


### Data Sharing
Because most RL training algorithms will need to run simultaneously several instances of the simulator, it could be a good idea to load the data required to instantiate the simulator **once for all** in a shared memory and provide the memory address to the different simulator instances. This has the advantage of reducing the startup time of a training process.

To this end, the script `./chloe/utils/data_to_shared_mem.py` has been created and it relies on [Apache Arrow](https://arrow.apache.org/docs/python/).

First, proceed by defining a space in the RAM where the data will be stored. Run the following command in a separate shell where 6GB is reserved for this purpose (a different name can be used in place of `/tmp/plasma` and the size of the allowed memory could be modified as well):
```
poetry run plasma_store -m 6000000000 -s /tmp/plasma
```

Next, in a different shell (do not close the first shell), run the following command:
```
poetry run python chloe/utils/data_to_shared_mem.py --data <path_to_train_data> --eval_data <path_to_validate_data> --data_prefix training --eval_data_prefix validate --shared_data_socket "/tmp/plasma" 
```
For instance, to put the downloaded data  present in `./data` into the plasma store, run the following command in cmdline:
```
poetry run python chloe/utils/data_to_shared_mem.py --data ./data/release_train_patients.zip --eval_data ./data/release_validate_patients.zip --data_prefix training --eval_data_prefix validate --shared_data_socket "/tmp/plasma" 
```

### Running the code for training

Running scripts have been put in place depending whether or not the plasma store for sharing data is used:
   - When using plasma store: [`./scripts/run_train.sh`](scripts/run_train.sh).
   - When not using plasma store: [`./scripts/run_train_no_sharing.sh`](scripts/run_train_no_sharing.sh).

All these files contain three variables that could be adjusted accordingly:
   - `config_base`: the folder where the configuration files are located
   - `output_base`: the folder where the trained agents will be saved

   - `data_base`: the folder where to get the data from

To train an agent, run the following command:
```
bash [path-to-bash-run-script] [data-dir] [yaml-model-config-file] [plasma_store_id] [cuda-id] [num-workers] [prefix] [path-to-train-data] [path-to-eval-data]
```
- `path-to-bash-run-script`: Bash file containing the run instructions and some default paths (`run_train.sh`, `run_train_no_sharing.sh`, ).
- `data-dir`: the folder where to retrieve the dataset
- `yaml-model-config-file`: Name of the config file containing config for training the agent, reward, and simulator.
- `plasma_store_id`: Id of the plasma store to be used (e.g., `/tmp/plasma`). **only used for the `run_train.sh` script.**
- `cuda-id`: id of the gpu to be used in the run.
- `num-workers`: Number of workers to be used.
- `prefix`: It gets added to the output path. This is currently used to separate the output for training and evaluation into separate folders.
- `path-to-train-data`: Path to the training patients. 
- `path-to-eval-data`: Path to the validation patients.


Assuming plasma store is used and the downloaded data be put in `./data`, the code for training the agent whose performance has been reported in the paper is:

```
bash ./scripts/run_train.sh "./data" config1.yaml "/tmp/plasma" 0 4
```

After the training, the trained agent is located in the folder `./output/config1/run_0` under the name `best.pkl`.

More generally, the trained agent is under `$output_base/$config$prefix/run_0/best.pkl`.


The model may take 1-3 days to converge.

## Evaluation

Once the training process is over, it is time to evaluate the trained agent.

As for the training pipeline, we provide scripts for running the evaluation:
   - [`./scripts/run_eval.sh`](scripts/run_eval.sh).


The file contains three variables that could be adjusted accordingly:
   - `output_base`: the folder where the trained agents is saved
   - `data_base`: the folder where to get the data from

To evaluate the agent, run the following command:
```
bash [path-to-bash-run-script] [data-dir] [yaml-model-config-file] [cuda-id] [path-to-eval-data] [path-to-output-model-dir]
```

- `path-to-bash-run-script`: Bash file containing the run instructions and some default paths (`run_eval.sh`, `run_eval_no_sharing.sh`, ).
- `data-dir`: the folder where to retrieve the dataset
- `yaml-model-config-file`: Name of the config file containing config for training the agent, reward, and simulator.
- `cuda-id`: id of the gpu to be used in the run.
- `path-to-eval-data`: Path to the eval dataset.
- `path-to-output-model-dir`: Path to the directory of the model to be evaluated (without the `run_0` suffix)



To reproduce the results in the paper, do the follwing:

Depending on which version of the DDXPLUS dataset you are usig, rename either config1_eng.yaml or config1_fr.yaml to config1.yaml, and then run the following 2 commands:

```
bash ./scripts/run_train.sh "./data" config1.yaml "/tmp/plasma" 0 4

bash ./scripts/run_eval.sh "./data" cfg.yml 0  "release_test_patients.zip" <path-where-model-where-saved>
```


We used the `AUCTraj` metric as a proxy to quantify the quality of a trajectory to mimic the exploration-confirmation paradigm during our evaluation. While not perfect, it tends to capture the Area under the curve of the graph which, for each differential diagnosis prediction `p_t` made during a trajectory at time `t`, plots a point `(x, y)` where `x = 1 - exp(-KLDIV(p_t, p_0))` is a dissimilarity measure between `p_t` and `p_0` (i.e., how far is the current prediction with respect to the first prediction) and `y = exp(-KLDIV(p_t, gt_diff))` is a similarity measure between `p_t` and the ground-truth differential `gt_diff`. Properly quantifying the quality of a trajectory under the exploration-confirmation paradigm is still an open-question.

The resulting metrics are located in `./ouput/<path-where-model-where-saved>/run_0/best_eval/BatchMetrics.json`.

Please note that the `DSHM` metric used in the paper is called `DSF1` in the code.

## Printing Interactions

The folder [`notebooks`](notebooks/) contains a notebook which illustrates how to write down the trajectories followed by a given agent when interacting with a patient as well as wirting down the patient.
