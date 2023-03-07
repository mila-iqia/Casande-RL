import os

import mlflow
import orion.client.cli as cli
from rlpyt.utils.collections import AttrDict

from chloe.main_rl import run
from chloe.pretrain import run as pretrain_run


class TestAlgo(object):
    def system_setup(self, tmpdir):
        """get the simulator data
        """
        sample_symptoms = [
            "{",
            '    "sy1": {',
            '        "name": "sy1",',
            '        "type-donnes": "C",',
            '        "default_value": 1,',
            '        "possible-values": [1, 2, 3]',
            "    },",
            '    "sy2": {',
            '        "name": "sy2",',
            '        "type-donnes": "M",',
            '        "default_value": "A",',
            '        "possible-values": ["A", "B", "C", "D", "E"]',
            "    },",
            '    "sy3": {',
            '        "name": "sy3"',
            "    },",
            '    "sy4": {',
            '        "name": "sy4"',
            "    },",
            '    "sy5": {',
            '        "name": "sy5"',
            "    },",
            '    "sy6": {',
            '        "name": "sy6"',
            "    },",
            '    "sy7": {',
            '        "name": "sy7"',
            "    },",
            '    "sy8": {',
            '        "name": "sy8"',
            "    },",
            '    "sy9": {',
            '        "name": "sy9"',
            "    }",
            "}",
        ]
        sample_conditions = [
            "{",
            '    "cond1": {',
            '        "condition_name": "First5",',
            '        "severity": 1,',
            '        "symptoms": {',
            '            "sy1": {',
            '                "probability": 53',
            "            },",
            '            "sy2": {',
            '                "probability": 35',
            "            },",
            '            "sy3": {',
            '                "probability": 28',
            "            },",
            '            "sy4": {',
            '                "probability": 28',
            "            },",
            '            "sy5": {',
            '                "probability": 23',
            "            }",
            "        }",
            "    },",
            '    "cond2": {',
            '        "condition_name": "Mult3",',
            '        "severity": 2,',
            '        "symptoms": {',
            '            "sy3": {',
            '                "probability": 65',
            "            },",
            '            "sy6": {',
            '                "probability": 32',
            "            },",
            '            "sy9": {',
            '                "probability": 29',
            "            }",
            "        }",
            "    },",
            '    "cond3": {',
            '        "condition_name": "Prime",',
            '        "severity": 3,',
            '        "symptoms": {',
            '            "sy7": {',
            '                "probability": 84',
            "            },",
            '            "sy5": {',
            '                "probability": 84',
            "            },",
            '            "sy3": {',
            '                "probability": 73',
            "            },",
            '            "sy2": {',
            '                "probability": 62',
            "            },",
            '            "sy1": {',
            '                "probability": 44',
            "            }",
            "        }",
            "    },",
            '    "cond4": {',
            '        "condition_name": "Odd",',
            '        "severity": 4,',
            '        "symptoms": {',
            '            "sy1": {',
            '                "probability": 81',
            "            },",
            '            "sy3": {',
            '                "probability": 72',
            "            },",
            '            "sy5": {',
            '                "probability": 72',
            "            },",
            '            "sy7": {',
            '                "probability": 54',
            "            },",
            '            "sy9": {',
            '                "probability": 54',
            "            }",
            "        }",
            "    },",
            '    "cond5": {',
            '        "condition_name": "Even",',
            '        "severity": 5,',
            '        "symptoms": {',
            '            "sy2": {',
            '                "probability": 83',
            "            },",
            '            "sy4": {',
            '                "probability": 69',
            "            },",
            '            "sy6": {',
            '                "probability": 58',
            "            },",
            '            "sy8": {',
            '                "probability": 51',
            "            }",
            "        }",
            "    }",
            "}",
        ]
        header1 = "PATIENT,GENDER,RACE,ETHNICITY,AGE_BEGIN"
        header2 = "AGE_END,PATHOLOGY,NUM_SYMPTOMS,SYMPTOMS"
        sample_patients = [
            header1 + "," + header2,
            "id1,M,white,nonhispanic,9,9,First5,3,sy2_@_A:38;sy4:47;sy5:31;sy2_@_B:38",
            "id2,M,white,nonhispanic,0,0,First5,5,sy1_@_1:3;sy2_@_E:8;sy3:8;sy4:7;sy5:3",
            "id3,M,white,nonhispanic,1,1,Even,3,sy6:30;sy2_@_C:42;sy2_@_D:42;sy8:35",
            "id4,M,white,nonhispanic,4,4,Odd,3,sy7:39;sy3:42;sy9:28",
            "id5,M,white,nonhispanic,8,8,Mult3,2,sy6:34;sy9:37",
            "id6,M,white,nonhispanic,9,9,Prime,3,sy1_@_3:32;sy2_@_B:38;sy5:31",
        ]

        # tmp_file symptoms
        symptoms = tmpdir.join("symptoms.json")
        sample_data_symptom = "\n".join(sample_symptoms)
        symptoms.write(sample_data_symptom)
        filename_symptom = os.path.join(tmpdir, "symptoms.json")

        # tmp_file conditions
        conditions = tmpdir.join("conditions.json")
        sample_data_condition = "\n".join(sample_conditions)
        conditions.write(sample_data_condition)
        filename_condition = os.path.join(tmpdir, "conditions.json")

        # tmp_file samples
        patients = tmpdir.join("patients.csv")
        sample_data_patient = "\n".join(sample_patients)
        patients.write(sample_data_patient)
        filename_patients = os.path.join(tmpdir, "patients.csv")

        return filename_symptom, filename_condition, filename_patients

    def system_setup_differential(self, tmpdir):
        """get the simulator data with differential data
        """
        sample_symptoms = [
            "{",
            '    "sy1": {',
            '        "name": "sy1",',
            '        "type-donnes": "C",',
            '        "default_value": 1,',
            '        "possible-values": [1, 2, 3]',
            "    },",
            '    "sy2": {',
            '        "name": "sy2",',
            '        "type-donnes": "M",',
            '        "default_value": "A",',
            '        "possible-values": ["A", "B", "C", "D", "E"]',
            "    },",
            '    "sy3": {',
            '        "name": "sy3"',
            "    },",
            '    "sy4": {',
            '        "name": "sy4"',
            "    },",
            '    "sy5": {',
            '        "name": "sy5"',
            "    },",
            '    "sy6": {',
            '        "name": "sy6"',
            "    },",
            '    "sy7": {',
            '        "name": "sy7"',
            "    },",
            '    "sy8": {',
            '        "name": "sy8"',
            "    },",
            '    "sy9": {',
            '        "name": "sy9"',
            "    }",
            "}",
        ]
        sample_conditions = [
            "{",
            '    "cond1": {',
            '        "condition_name": "First5",',
            '        "severity": 1,',
            '        "symptoms": {',
            '            "sy1": {',
            '                "probability": 53',
            "            },",
            '            "sy2": {',
            '                "probability": 35',
            "            },",
            '            "sy3": {',
            '                "probability": 28',
            "            },",
            '            "sy4": {',
            '                "probability": 28',
            "            },",
            '            "sy5": {',
            '                "probability": 23',
            "            }",
            "        }",
            "    },",
            '    "cond2": {',
            '        "condition_name": "Mult3",',
            '        "severity": 2,',
            '        "symptoms": {',
            '            "sy3": {',
            '                "probability": 65',
            "            },",
            '            "sy6": {',
            '                "probability": 32',
            "            },",
            '            "sy9": {',
            '                "probability": 29',
            "            }",
            "        }",
            "    },",
            '    "cond3": {',
            '        "condition_name": "Prime",',
            '        "severity": 3,',
            '        "symptoms": {',
            '            "sy7": {',
            '                "probability": 84',
            "            },",
            '            "sy5": {',
            '                "probability": 84',
            "            },",
            '            "sy3": {',
            '                "probability": 73',
            "            },",
            '            "sy2": {',
            '                "probability": 62',
            "            },",
            '            "sy1": {',
            '                "probability": 44',
            "            }",
            "        }",
            "    },",
            '    "cond4": {',
            '        "condition_name": "Odd",',
            '        "severity": 4,',
            '        "symptoms": {',
            '            "sy1": {',
            '                "probability": 81',
            "            },",
            '            "sy3": {',
            '                "probability": 72',
            "            },",
            '            "sy5": {',
            '                "probability": 72',
            "            },",
            '            "sy7": {',
            '                "probability": 54',
            "            },",
            '            "sy9": {',
            '                "probability": 54',
            "            }",
            "        }",
            "    },",
            '    "cond5": {',
            '        "condition_name": "Even",',
            '        "severity": 5,',
            '        "symptoms": {',
            '            "sy2": {',
            '                "probability": 83',
            "            },",
            '            "sy4": {',
            '                "probability": 69',
            "            },",
            '            "sy6": {',
            '                "probability": 58',
            "            },",
            '            "sy8": {',
            '                "probability": 51',
            "            }",
            "        }",
            "    }",
            "}",
        ]
        header1 = "PATIENT,GENDER,RACE,ETHNICITY,AGE_BEGIN"
        header2 = "AGE_END,PATHOLOGY,NUM_SYMPTOMS,SYMPTOMS"
        header3 = "DIFFERNTIAL_DIAGNOSIS,FOUND_GT_PATHOLOGY"
        data = [
            "id1,M,white,nonhispanic,9,9,First5,3,sy2_@_A:38;sy4:47;sy5:31;sy2_@_B:38",
            "id2,M,white,nonhispanic,0,0,First5,5,sy1_@_1:3;sy2_@_E:8;sy3:8;sy4:7;sy5:3",
            "id3,M,white,nonhispanic,1,1,Even,3,sy6:30;sy2_@_C:42;sy2_@_D:42;sy8:35",
            "id4,M,white,nonhispanic,4,4,Odd,3,sy7:39;sy3:42;sy9:28",
            "id5,M,white,nonhispanic,8,8,Mult3,2,sy6:34;sy9:37",
            "id6,M,white,nonhispanic,9,9,Prime,3,sy1_@_3:32;sy2_@_B:38;sy5:31",
        ]
        diff = [
            "First5:1:2;Mult3:8:9;Prime:1:1,True",
            ",False",
            "Odd:9:0;Prime:1:2,False",
            "Even:1:1;Odd:4:5;First5:4:3,True",
            "Mult3:3:5;Even:9:8,True",
            "Prime:3:9,True",
        ]
        final_data = [data[i] + "," + diff[i] for i in range(len(data))]
        sample_patients = [header1 + "," + header2 + "," + header3] + final_data

        # tmp_file symptoms
        symptoms = tmpdir.join("symptoms.json")
        sample_data_symptom = "\n".join(sample_symptoms)
        symptoms.write(sample_data_symptom)
        filename_symptom = os.path.join(tmpdir, "symptoms.json")

        # tmp_file conditions
        conditions = tmpdir.join("conditions.json")
        sample_data_condition = "\n".join(sample_conditions)
        conditions.write(sample_data_condition)
        filename_condition = os.path.join(tmpdir, "conditions.json")

        # tmp_file samples
        patients = tmpdir.join("patients.csv")
        sample_data_patient = "\n".join(sample_patients)
        patients.write(sample_data_patient)
        filename_patients = os.path.join(tmpdir, "patients.csv")

        return filename_symptom, filename_condition, filename_patients

    def system_setup_hierarchical_differential(self, tmpdir):
        """get the simulator data with differential data and hierarchical symptoms.
        """
        sample_symptoms = [
            "{",
            '    "sy1": {',
            '        "name": "sy1",',
            '        "type-donnes": "C",',
            '        "code_question": "sy5",',
            '        "default_value": 1,',
            '        "possible-values": [1, 2, 3]',
            "    },",
            '    "sy2": {',
            '        "name": "sy2",',
            '        "type-donnes": "M",',
            '        "default_value": "A",',
            '        "possible-values": ["A", "B", "C", "D", "E"]',
            "    },",
            '    "sy3": {',
            '        "name": "sy3"',
            "    },",
            '    "sy4": {',
            '        "name": "sy4"',
            "    },",
            '    "sy5": {',
            '        "name": "sy5"',
            "    },",
            '    "sy6": {',
            '        "name": "sy6"',
            "    },",
            '    "sy7": {',
            '        "name": "sy7"',
            "    },",
            '    "sy8": {',
            '        "name": "sy8"',
            "    },",
            '    "sy9": {',
            '        "name": "sy9"',
            "    }",
            "}",
        ]
        sample_conditions = [
            "{",
            '    "cond1": {',
            '        "condition_name": "First5",',
            '        "symptoms": {',
            '            "sy1": {',
            '                "probability": 53',
            "            },",
            '            "sy2": {',
            '                "probability": 35',
            "            },",
            '            "sy3": {',
            '                "probability": 28',
            "            },",
            '            "sy4": {',
            '                "probability": 28',
            "            },",
            '            "sy5": {',
            '                "probability": 23',
            "            }",
            "        }",
            "    },",
            '    "cond2": {',
            '        "condition_name": "Mult3",',
            '        "symptoms": {',
            '            "sy3": {',
            '                "probability": 65',
            "            },",
            '            "sy6": {',
            '                "probability": 32',
            "            },",
            '            "sy9": {',
            '                "probability": 29',
            "            }",
            "        }",
            "    },",
            '    "cond3": {',
            '        "condition_name": "Prime",',
            '        "symptoms": {',
            '            "sy7": {',
            '                "probability": 84',
            "            },",
            '            "sy5": {',
            '                "probability": 84',
            "            },",
            '            "sy3": {',
            '                "probability": 73',
            "            },",
            '            "sy2": {',
            '                "probability": 62',
            "            },",
            '            "sy1": {',
            '                "probability": 44',
            "            }",
            "        }",
            "    },",
            '    "cond4": {',
            '        "condition_name": "Odd",',
            '        "symptoms": {',
            '            "sy1": {',
            '                "probability": 81',
            "            },",
            '            "sy3": {',
            '                "probability": 72',
            "            },",
            '            "sy5": {',
            '                "probability": 72',
            "            },",
            '            "sy7": {',
            '                "probability": 54',
            "            },",
            '            "sy9": {',
            '                "probability": 54',
            "            }",
            "        }",
            "    },",
            '    "cond5": {',
            '        "condition_name": "Even",',
            '        "symptoms": {',
            '            "sy2": {',
            '                "probability": 83',
            "            },",
            '            "sy4": {',
            '                "probability": 69',
            "            },",
            '            "sy6": {',
            '                "probability": 58',
            "            },",
            '            "sy8": {',
            '                "probability": 51',
            "            }",
            "        }",
            "    }",
            "}",
        ]
        header1 = "PATIENT,GENDER,RACE,ETHNICITY,AGE_BEGIN"
        header2 = "AGE_END,PATHOLOGY,NUM_SYMPTOMS,SYMPTOMS"
        header3 = "DIFFERNTIAL_DIAGNOSIS,FOUND_GT_PATHOLOGY"
        data = [
            "id1,M,white,nonhispanic,9,9,First5,3,sy2_@_A:38;sy4:47;sy5:31;sy2_@_B:38",
            "id2,M,white,nonhispanic,0,0,First5,5,sy1_@_1:3;sy2_@_E:8;sy3:8;sy4:7;sy5:3",
            "id3,M,white,nonhispanic,1,1,Even,3,sy6:30;sy2_@_C:42;sy2_@_D:42;sy8:35",
            "id4,M,white,nonhispanic,4,4,Odd,3,sy7:39;sy3:42;sy9:28",
            "id5,M,white,nonhispanic,8,8,Mult3,2,sy6:34;sy9:37",
            "id6,M,white,nonhispanic,9,9,Prime,3,sy1_@_3:32;sy2_@_B:38;sy5:31",
        ]
        diff = [
            "First5:1:2;Mult3:8:9;Prime:1:1,True",
            ",False",
            "Odd:9:0;Prime:1:2,False",
            "Even:1:1;Odd:4:5;First5:4:3,True",
            "Mult3:3:5;Even:9:8,True",
            "Prime:3:9,True",
        ]
        final_data = [data[i] + "," + diff[i] for i in range(len(data))]
        sample_patients = [header1 + "," + header2 + "," + header3] + final_data

        # tmp_file symptoms
        symptoms = tmpdir.join("symptoms.json")
        sample_data_symptom = "\n".join(sample_symptoms)
        symptoms.write(sample_data_symptom)
        filename_symptom = os.path.join(tmpdir, "symptoms.json")

        # tmp_file conditions
        conditions = tmpdir.join("conditions.json")
        sample_data_condition = "\n".join(sample_conditions)
        conditions.write(sample_data_condition)
        filename_condition = os.path.join(tmpdir, "conditions.json")

        # tmp_file samples
        patients = tmpdir.join("patients.csv")
        sample_data_patient = "\n".join(sample_patients)
        patients.write(sample_data_patient)
        filename_patients = os.path.join(tmpdir, "patients.csv")

        return filename_symptom, filename_condition, filename_patients

    def get_args(self, tmpdir, data, eval_data=None):
        return AttrDict(
            {
                "data": data,
                "eval_data": data if eval_data is None else eval_data,
                "output": tmpdir,
                "run_ID": 0,
                "datetime_suffix": False,
                "no_replace_if_present": False,
                "start_from_scratch": True,
                "cpu_list": None,
                "num_torch_threads": None,
                "n_gpus": 0,
                "cuda_idx": None,
                "n_workers": 2,
                "shared_data_socket": None,
            }
        )

    def get_pretrain_args(self, tmpdir, data, eval_data=None):
        return AttrDict(
            {
                "data": data,
                "eval_data": data if eval_data is None else eval_data,
                "output": tmpdir,
                "datetime_suffix": False,
                "no_replace_if_present": False,
                "no_data_corrupt": False,
                "num_epochs": 3,
                "batch_size": 2,
                "n_workers": 0,
                "patience": 2,
                "valid_percentage": 0.5,
                "lr": 1e-3,
                "metric": "dist_accuracy",
                "topk": 1,
                "seed": 1234,
                "cuda_idx": None,
                "shared_data_socket": None,
            }
        )

    def get_params(self, symptom_file, condition_file, architecture, agent, algo):
        sim_params = {
            "symptom_filepath": symptom_file,
            "condition_filepath": condition_file,
            "max_turns": 5,
            "action_type": 0,
            "include_turns_in_state": True,
            "stop_if_repeated_question": False,
            "include_race_in_state": True,
            "include_ethnicity_in_state": True,
            "is_reward_relevancy_patient_specific": True,
            "use_differential_diagnosis": True,
        }

        reward_config = {
            "reward_on_repeated_action": 0,
            "reward_on_missing_diagnosis": -5,
            "reward_on_correct_diagnosis": 10,
            "reward_on_intermediary_turns": -1,
            "reward_on_relevant_symptom_inquiry": 0,
            "reward_on_irrelevant_symptom_inquiry": 0,
        }

        sampler_params = {}

        architecture_params = {
            "input_size": 17,
            "hidden_sizes": [20, 20],
            "output_size": 14,
            "dueling": True,
            "dueling_fc_sizes": [20],
            "num_symptoms": 9,
            "freeze_one_hot_encoding": True,
            "embedding_dict": {1: 8, 2: 2, 3: 5, 4: 2},
            "not_inquired_value": 0,
            "mask_inquired_symptoms": True,
            "include_turns_in_state": True,
            "use_turn_just_for_masking": False,
            "min_turns_ratio_for_decision": 0.4,
            "n_atoms": 5,
        }

        if "mixed" in architecture.lower():
            architecture_params["pi_hidden_sizes"] = [20]

        if "rebuild" in architecture.lower():
            architecture_params["reb_size"] = 15
            architecture_params["reb_hidden_sizes"] = [20]

        if "mixreb" in architecture.lower():
            architecture_params["reb_size"] = 15
            architecture_params["reb_hidden_sizes"] = [20]
            architecture_params["pi_hidden_sizes"] = [20]

        if "r2d1" in architecture.lower():
            architecture_params["lstm_size"] = 15

        agent_params = {}
        if "cat" in agent.lower():
            agent_params["n_atoms"] = 5

        algo_params = {
            "discount": 0.99,
            "batch_size": 3,
            "replay_ratio": 32,
            "replay_size": 200,
            "learning_rate": 0.0000625,
            "target_update_interval": 1,
            "eps_steps": 10,
            "n_step_return": 2,
            "double_dqn": True,
            "prioritized_replay": True,
            "min_steps_learn": 4,
            "ReplayBufferCls": "PrioritizedReplayBuffer",
        }
        if "cat" in algo.lower():
            algo_params["V_min"] = -20
            algo_params["V_max"] = 15

        if "r2d1" in algo.lower():
            algo_params["ReplayBufferCls"] = "PrioritizedSequenceReplayBuffer"
            algo_params["warmup_T"] = 0
            algo_params["batch_T"] = 3
            algo_params["store_rnn_state_interval"] = 2
            algo_params.pop("batch_size")
            sim_params["max_turns"] = 8
            sampler_params["batch_T"] = 2

        if "mixed" in algo.lower():
            added_params = {
                "replay_intermediate_data_flag": True,
                "separate_classifier_optimizer": True,
                "pretrain_flag": True,
                "pretrain_epochs": 2,
                "pretrain_batch_size": 2,
                "pretrain_perf_metric": "accuracy",
                "pretrain_loss_func": "cross_entropy",
                "reward_shaping_flag": True,
                "reward_shaping_back_propagate_flag": False,
                "reward_shaping_func": "cross_entropy",
                "clf_reward_flag": True,
                "clf_reward_func": "cross_entropy",
                "clf_loss_flag": True,
                "clf_loss_complete_data_flag": False,
                "clf_loss_only_at_end_episode_flag": True,
                "clf_loss_func": "cross_entropy",
            }
            algo_params.update(added_params)
            if "mixreb" in architecture.lower():
                added_params = {
                    "feature_rebuild_loss_flag": True,
                    "feature_rebuild_loss_min": None,
                    "feature_rebuild_loss_max": None,
                    "feature_rebuild_loss_coef": 1.0,
                    "feature_rebuild_loss_func": "bce",
                    "feature_rebuild_loss_kwargs": None,
                }
                algo_params.update(added_params)

        if "rebuild" in algo.lower():
            added_params = {
                "feature_rebuild_loss_flag": True,
                "feature_rebuild_loss_min": None,
                "feature_rebuild_loss_max": None,
                "feature_rebuild_loss_coef": 1.0,
                "feature_rebuild_loss_func": "bce",
                "feature_rebuild_loss_kwargs": None,
            }
            algo_params.update(added_params)

        runner_params = {
            "topk": 3,
            "eval_coeffs": [0.4, 0.15, 0.15, 0.3],
            "traj_auxiliary_reward_flag": True,
        }

        hyper_params = {
            "exp_name": f"test_{algo}_{agent}",
            "optimizer": "adam",
            "perf_window_size": 1,
            "log_interval_steps": 2,
            "n_steps": 20,
            "n_envs": 1,
            "max_decorrelation_steps": 0,
            "eval_n_envs": 0,
            "eval_max_steps": 60,
            "eval_max_trajectories": 2,
            "patience": 20,
            "runner": "MinibatchRl",
            "sampler": "SerialSampler",
            "algo": algo,
            "agent": agent,
            "architecture": architecture,
            "simulator_params": sim_params,
            "reward_config": reward_config,
            "architecture_params": architecture_params,
            "algo_params": algo_params,
            "optimizer_params": {"eps": 0.00015},
            "sampler_params": sampler_params,
            "runner_params": runner_params,
            "agent_params": agent_params,
            "eval_metrics": ["accuracy", "f1"],
            "perf_metric": "f1",
        }
        return hyper_params

    def test_dqn_model_algo(self, tmpdir):
        out = self.system_setup(tmpdir)
        symptom, condition, patients = out
        args = self.get_args(tmpdir, patients)
        arch = "baseline_dqn_model"
        agent = "dqnagent"
        algo = "dqn"
        params = self.get_params(symptom, condition, arch, agent, algo)
        cli._HAS_REPORTED_RESULTS = False
        mlflow.start_run()
        run(args, params)
        mlflow.end_run()
        assert True

    def test_cat_dqn_model_algo(self, tmpdir):
        out = self.system_setup(tmpdir)
        symptom, condition, patients = out
        args = self.get_args(tmpdir, patients)
        arch = "baseline_cat_dqn_model"
        agent = "catdqnagent"
        algo = "categoricaldqn"
        params = self.get_params(symptom, condition, arch, agent, algo)
        cli._HAS_REPORTED_RESULTS = False
        mlflow.start_run()
        run(args, params)
        mlflow.end_run()
        assert True

    def test_mixed_dqn_model_algo(self, tmpdir):
        out = self.system_setup(tmpdir)
        symptom, condition, patients = out
        args = self.get_args(tmpdir, patients)
        arch = "mixed_dqn_model"
        agent = "mixed_dqnagent"
        algo = "mixed_dqn"
        params = self.get_params(symptom, condition, arch, agent, algo)
        cli._HAS_REPORTED_RESULTS = False
        mlflow.start_run()
        run(args, params)
        mlflow.end_run()
        assert True

    def test_rebuild_dqn_model_algo(self, tmpdir):
        out = self.system_setup(tmpdir)
        symptom, condition, patients = out
        args = self.get_args(tmpdir, patients)
        arch = "rebuild_dqn_model"
        agent = "rebuild_dqnagent"
        algo = "rebuild_dqn"
        params = self.get_params(symptom, condition, arch, agent, algo)
        cli._HAS_REPORTED_RESULTS = False
        mlflow.start_run()
        run(args, params)
        mlflow.end_run()
        assert True

    def test_mixed_rebuild_cat_dqn_model_algo(self, tmpdir):
        out = self.system_setup(tmpdir)
        symptom, condition, patients = out
        args = self.get_args(tmpdir, patients)
        arch = "mixreb_cat_dqn_model"
        agent = "mixed_catdqnagent"
        algo = "mixed_categoricaldqn"
        params = self.get_params(symptom, condition, arch, agent, algo)
        cli._HAS_REPORTED_RESULTS = False
        mlflow.start_run()
        run(args, params)
        mlflow.end_run()
        assert True

    def test_mixed_cat_dqn_model_algo(self, tmpdir):
        out = self.system_setup(tmpdir)
        symptom, condition, patients = out
        args = self.get_args(tmpdir, patients)
        arch = "mixed_cat_dqn_model"
        agent = "mixed_catdqnagent"
        algo = "mixed_categoricaldqn"
        params = self.get_params(symptom, condition, arch, agent, algo)
        cli._HAS_REPORTED_RESULTS = False
        mlflow.start_run()
        run(args, params)
        mlflow.end_run()
        assert True

    def test_mixed_dqn_model_algo_no_stop_action(self, tmpdir):
        out = self.system_setup(tmpdir)
        symptom, condition, patients = out
        args = self.get_args(tmpdir, patients)
        arch = "mixed_dqn_model"
        agent = "mixed_dqnagent"
        algo = "mixed_dqn"
        params = self.get_params(symptom, condition, arch, agent, algo)
        params["architecture_params"]["use_stop_action"] = False
        if not (params["algo_params"].get("clf_reward_kwargs", None)):
            params["algo_params"]["clf_reward_kwargs"] = {}
        params["algo_params"]["clf_reward_kwargs"]["exit_loss_coeff"] = 0.0
        cli._HAS_REPORTED_RESULTS = False
        mlflow.start_run()
        run(args, params)
        mlflow.end_run()
        assert True

    def test_mixed_cat_dqn_model_algo_no_stop_action(self, tmpdir):
        out = self.system_setup(tmpdir)
        symptom, condition, patients = out
        args = self.get_args(tmpdir, patients)
        arch = "mixed_cat_dqn_model"
        agent = "mixed_catdqnagent"
        algo = "mixed_categoricaldqn"
        params = self.get_params(symptom, condition, arch, agent, algo)
        params["architecture_params"]["use_stop_action"] = False
        if not (params["algo_params"].get("clf_reward_kwargs", None)):
            params["algo_params"]["clf_reward_kwargs"] = {}
        params["algo_params"]["clf_reward_kwargs"]["exit_loss_coeff"] = 0.0
        cli._HAS_REPORTED_RESULTS = False
        mlflow.start_run()
        run(args, params)
        mlflow.end_run()
        assert True

    def test_mixed_dqn_model_algo_with_traj_aux_reward_log(self, tmpdir):
        out = self.system_setup(tmpdir)
        symptom, condition, patients = out
        args = self.get_args(tmpdir, patients)
        arch = "mixed_dqn_model"
        agent = "mixed_dqnagent"
        algo = "mixed_dqn"
        params = self.get_params(symptom, condition, arch, agent, algo)
        params["runner_params"]["traj_auxiliary_reward_flag"] = True
        cli._HAS_REPORTED_RESULTS = False
        mlflow.start_run()
        run(args, params)
        mlflow.end_run()
        assert True

    def test_mixed_dqn_model_algo_with_diff_traj_aux_reward_log(self, tmpdir):
        out = self.system_setup_differential(tmpdir)
        symptom, condition, patients = out
        args = self.get_args(tmpdir, patients)
        arch = "mixed_dqn_model"
        agent = "mixed_dqnagent"
        algo = "mixed_dqn"
        params = self.get_params(symptom, condition, arch, agent, algo)
        params["runner_params"]["traj_auxiliary_reward_flag"] = True
        cli._HAS_REPORTED_RESULTS = False
        mlflow.start_run()
        run(args, params)
        mlflow.end_run()
        assert True

    def test_mixed_dqn_model_hierarchy_algo_with_diff_traj_aux_reward_log(self, tmpdir):
        out = self.system_setup_hierarchical_differential(tmpdir)
        symptom, condition, patients = out
        args = self.get_args(tmpdir, patients)
        arch = "mixed_dqn_model"
        agent = "mixed_dqnagent"
        algo = "mixed_dqn"
        params = self.get_params(symptom, condition, arch, agent, algo)
        params["runner_params"]["traj_auxiliary_reward_flag"] = True
        cli._HAS_REPORTED_RESULTS = False
        mlflow.start_run()
        run(args, params)
        mlflow.end_run()
        assert True

    def test_mixed_dqn_model_pretrain(self, tmpdir):
        out = self.system_setup(tmpdir)
        symptom, condition, patients = out
        args = self.get_pretrain_args(tmpdir, patients)
        arch = "mixed_dqn_model"
        agent = "mixed_dqnagent"
        algo = "mixed_dqn"
        params = self.get_params(symptom, condition, arch, agent, algo)
        params["runner_params"]["traj_auxiliary_reward_flag"] = False
        cli._HAS_REPORTED_RESULTS = False
        mlflow.start_run()
        pretrain_run(args, params)
        mlflow.end_run()
        assert True

    def test_mixed_dqn_model_differential_pretrain(self, tmpdir):
        out = self.system_setup_differential(tmpdir)
        symptom, condition, patients = out
        args = self.get_pretrain_args(tmpdir, patients)
        arch = "mixed_dqn_model"
        agent = "mixed_dqnagent"
        algo = "mixed_dqn"
        params = self.get_params(symptom, condition, arch, agent, algo)
        params["runner_params"]["traj_auxiliary_reward_flag"] = False
        cli._HAS_REPORTED_RESULTS = False
        mlflow.start_run()
        pretrain_run(args, params)
        mlflow.end_run()
        assert True

    def test_mixed_cat_dqn_model_algo_with_diff_traj_aux_reward_log(self, tmpdir):
        out = self.system_setup_differential(tmpdir)
        symptom, condition, patients = out
        args = self.get_args(tmpdir, patients)
        arch = "mixed_cat_dqn_model"
        agent = "mixed_catdqnagent"
        algo = "mixed_categoricaldqn"
        params = self.get_params(symptom, condition, arch, agent, algo)
        params["runner_params"]["traj_auxiliary_reward_flag"] = True
        cli._HAS_REPORTED_RESULTS = False
        mlflow.start_run()
        run(args, params)
        mlflow.end_run()
        assert True

    def test_r2d1_dqn_model_algo(self, tmpdir):
        out = self.system_setup(tmpdir)
        symptom, condition, patients = out
        args = self.get_args(tmpdir, patients)
        arch = "baseline_r2d1_dqn_model"
        agent = "r2d1agent"
        algo = "r2d1"
        params = self.get_params(symptom, condition, arch, agent, algo)
        cli._HAS_REPORTED_RESULTS = False
        mlflow.start_run()
        run(args, params)
        mlflow.end_run()
        assert True

    def test_mixed_r2d1_dqn_model_algo(self, tmpdir):
        out = self.system_setup(tmpdir)
        symptom, condition, patients = out
        args = self.get_args(tmpdir, patients)
        arch = "mixed_r2d1_dqn_model"
        agent = "mixed_r2d1agent"
        algo = "mixed_r2d1"
        params = self.get_params(symptom, condition, arch, agent, algo)
        params["algo_params"]["replay_intermediate_data_flag"] = False
        cli._HAS_REPORTED_RESULTS = False
        mlflow.start_run()
        run(args, params)
        mlflow.end_run()
        assert True

    def test_mixed_r2d1_dqn_model_algo_with_diff_traj_aux_reward_log(self, tmpdir):
        out = self.system_setup_differential(tmpdir)
        symptom, condition, patients = out
        args = self.get_args(tmpdir, patients)
        arch = "mixed_r2d1_dqn_model"
        agent = "mixed_r2d1agent"
        algo = "mixed_r2d1"
        params = self.get_params(symptom, condition, arch, agent, algo)
        params["algo_params"]["replay_intermediate_data_flag"] = False
        params["runner_params"]["traj_auxiliary_reward_flag"] = True
        cli._HAS_REPORTED_RESULTS = False
        mlflow.start_run()
        run(args, params)
        mlflow.end_run()
        assert True

    def test_mixed_r2d1_dqn_model_pretrain(self, tmpdir):
        out = self.system_setup(tmpdir)
        symptom, condition, patients = out
        args = self.get_pretrain_args(tmpdir, patients)
        arch = "mixed_r2d1_dqn_model"
        agent = "mixed_r2d1agent"
        algo = "mixed_r2d1"
        params = self.get_params(symptom, condition, arch, agent, algo)
        params["algo_params"]["replay_intermediate_data_flag"] = False
        params["runner_params"]["traj_auxiliary_reward_flag"] = False
        cli._HAS_REPORTED_RESULTS = False
        mlflow.start_run()
        pretrain_run(args, params)
        mlflow.end_run()
        assert True

    def test_mixed_r2d1_dqn_model_differential_pretrain(self, tmpdir):
        out = self.system_setup_differential(tmpdir)
        symptom, condition, patients = out
        args = self.get_pretrain_args(tmpdir, patients)
        arch = "mixed_r2d1_dqn_model"
        agent = "mixed_r2d1agent"
        algo = "mixed_r2d1"
        params = self.get_params(symptom, condition, arch, agent, algo)
        params["algo_params"]["replay_intermediate_data_flag"] = False
        params["runner_params"]["traj_auxiliary_reward_flag"] = False
        cli._HAS_REPORTED_RESULTS = False
        mlflow.start_run()
        pretrain_run(args, params)
        mlflow.end_run()
        assert True

    def test_severity_mixed_cat_dqn_model_algo(self, tmpdir):
        out = self.system_setup(tmpdir)
        symptom, condition, patients = out
        args = self.get_args(tmpdir, patients)
        arch = "mixed_cat_dqn_model"
        agent = "mixed_catdqnagent"
        algo = "mixed_categoricaldqn"
        params = self.get_params(symptom, condition, arch, agent, algo)
        tmpF = "sigmoid_modulated_cross_entropy_and_entropy_neg_reward"
        params["algo_params"]["clf_reward_func"] = tmpF
        params["algo_params"]["clf_reward_kwargs"] = dict(
            max_turns=5,
            ent_weight=0.0,
            alpha=50,
            initial_penalty=0.0,
            ce_max_value=10.0,
            ent_max_value=None,
            penalty_alpha=50,
            use_severity_as_weight=False,
            ce_weight=1.0,
            sev_in_weight=0.5,
            sev_out_weight=0.5,
        )
        tmpF2 = "ce_ent_sent_reshaping"
        params["algo_params"]["reward_shaping_func"] = tmpF2
        params["algo_params"]["reward_shaping_kwargs"] = dict(
            max_turns=5,
            min_map_val=-2.0,
            max_map_val=2.0,
            ce_alpha=2,
            ent_alpha=5,
            js_alpha=9,
            tv_alpha=5,
            sev_ent_alpha=8,
            sev_ent_alpha_b=0.5,
            ent_weight=0.0,
            ce_weight=1.0,
            js_weight=20.0,
            tv_weight=0.0,
            sev_ent_weight=0.0,
            sev_tv_weight=0.0,
            sev_js_weight=0.0,
            sev_in_weight=0.5,
            sev_out_weight=0.5,
            normalize_sev_dist=False,
            link_div_with_negative_evidence=False,
            bounds_dict=dict(js_min=0, js_max=0.15, ce_min=-2.0, ce_max=2.0,),
        )
        cli._HAS_REPORTED_RESULTS = False
        mlflow.start_run()
        run(args, params)
        mlflow.end_run()
        assert True
