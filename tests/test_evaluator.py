import os

import mlflow
import orion.client.cli as cli
from rlpyt.utils.collections import AttrDict

from chloe.eval import run as eval_run
from chloe.main_rl import run


class TestEvaluator(object):
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

    def get_eval_args(self, tmpdir, data, model_path):
        return AttrDict(
            {
                "data": data,
                "model_path": model_path,
                "output": tmpdir,
                "datetime_suffix": False,
                "no_replace_if_present": False,
                "max_trajectories": None,
                "seed": None,
                "n_envs": 1,
                "sharing_prefix": "testing",
                "cuda_idx": None,
                "shared_data_socket": None,
                "max_generation": 5,
                "topk": 3,
                "eval_coeffs": None,
                "sample_indices_flag": True,
                "compute_metric_flag": True,
            }
        )

    def get_params(self, symptom_file, condition_file, architecture, agent, algo):
        sim_params = {
            "symptom_filepath": symptom_file,
            "condition_filepath": condition_file,
            "max_turns": 5,
            "action_type": 0,
            "include_turns_in_state": False,
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

        architecture_params = {
            "input_size": 17,
            "hidden_sizes": [20, 20],
            "output_size": 14,
            "dueling": True,
            "dueling_fc_sizes": [20],
            "num_symptoms": 9,
            "freeze_one_hot_encoding": True,
            "embedding_dict": {0: 8, 1: 2, 2: 5, 3: 2},
            "not_inquired_value": 0,
            "mask_inquired_symptoms": True,
            "include_turns_in_state": False,
            "use_turn_just_for_masking": False,
            "min_turns_ratio_for_decision": None,
            "n_atoms": 5,
        }

        if "mixed" in architecture.lower():
            architecture_params["pi_hidden_sizes"] = [20]

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
            "n_step_return": 3,
            "double_dqn": True,
            "prioritized_replay": True,
            "min_steps_learn": 4,
            "ReplayBufferCls": "PrioritizedReplayBuffer",
        }
        if "cat" in algo.lower():
            algo_params["V_min"] = -20
            algo_params["V_max"] = 15

        if "mixed" in algo.lower():
            added_params = {
                "replay_intermediate_data_flag": True,
                "separate_classifier_optimizer": True,
                "pretrain_flag": True,
                "pretrain_epochs": 1,
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
            "sampler_params": {},
            "runner_params": runner_params,
            "agent_params": agent_params,
            "eval_metrics": ["accuracy", "f1"],
            "perf_metric": "f1",
        }
        return hyper_params

    def test_eval_dqn_model_algo(self, tmpdir):
        out = self.system_setup(tmpdir)
        symptom, condition, patients = out
        args = self.get_args(tmpdir, patients)
        arch = "baseline_dqn_model"
        agent = "dqnagent"
        algo = "dqn"
        params = self.get_params(symptom, condition, arch, agent, algo)
        cli._HAS_REPORTED_RESULTS = False
        mlflow.start_run()
        uid = mlflow.active_run().info.run_id
        exp_name = params["exp_name"]
        pref = f"{exp_name}/{uid}/"
        run(args, params)
        mlflow.end_run()
        model_path = os.path.join(tmpdir, f"{pref}run_{args.run_ID}/params.pkl")
        eval_args = self.get_eval_args(tmpdir, patients, model_path)
        eval_run(eval_args, params)
        assert True

    def test_end_eval_dqn_model_algo(self, tmpdir):
        out = self.system_setup(tmpdir)
        symptom, condition, patients = out
        args = self.get_args(tmpdir, patients)
        args["end_training_eval_data"] = patients
        arch = "baseline_dqn_model"
        agent = "dqnagent"
        algo = "dqn"
        params = self.get_params(symptom, condition, arch, agent, algo)
        cli._HAS_REPORTED_RESULTS = False
        mlflow.start_run()
        run(args, params)
        mlflow.end_run()
        assert True

    def test_end_eval_dqn_model_algo_with_differential(self, tmpdir):
        out = self.system_setup_differential(tmpdir)
        symptom, condition, patients = out
        args = self.get_args(tmpdir, patients)
        args["end_training_eval_data"] = patients
        arch = "baseline_dqn_model"
        agent = "dqnagent"
        algo = "dqn"
        params = self.get_params(symptom, condition, arch, agent, algo)
        cli._HAS_REPORTED_RESULTS = False
        mlflow.start_run()
        run(args, params)
        mlflow.end_run()
        assert True

    def test_end_eval_mixed_dqn_model_algo(self, tmpdir):
        out = self.system_setup(tmpdir)
        symptom, condition, patients = out
        args = self.get_args(tmpdir, patients)
        args["end_training_eval_data"] = patients
        arch = "mixed_dqn_model"
        agent = "mixed_dqnagent"
        algo = "mixed_dqn"
        params = self.get_params(symptom, condition, arch, agent, algo)
        cli._HAS_REPORTED_RESULTS = False
        mlflow.start_run()
        run(args, params)
        mlflow.end_run()
        assert True

    def test_end_eval_mixed_dqn_model_algo_with_differential(self, tmpdir):
        out = self.system_setup_differential(tmpdir)
        symptom, condition, patients = out
        args = self.get_args(tmpdir, patients)
        args["end_training_eval_data"] = patients
        arch = "mixed_dqn_model"
        agent = "mixed_dqnagent"
        algo = "mixed_dqn"
        params = self.get_params(symptom, condition, arch, agent, algo)
        cli._HAS_REPORTED_RESULTS = False
        mlflow.start_run()
        run(args, params)
        mlflow.end_run()
        assert True

    def test_eval_dqn_model_algo_with_traj_aux_reward_log(self, tmpdir):
        out = self.system_setup(tmpdir)
        symptom, condition, patients = out
        args = self.get_args(tmpdir, patients)
        arch = "baseline_dqn_model"
        agent = "dqnagent"
        algo = "dqn"
        params = self.get_params(symptom, condition, arch, agent, algo)
        params["runner_params"]["traj_auxiliary_reward_flag"] = True
        cli._HAS_REPORTED_RESULTS = False
        mlflow.start_run()
        uid = mlflow.active_run().info.run_id
        exp_name = params["exp_name"]
        pref = f"{exp_name}/{uid}/"
        run(args, params)
        mlflow.end_run()
        model_path = os.path.join(tmpdir, f"{pref}run_{args.run_ID}/params.pkl")
        eval_args = self.get_eval_args(tmpdir, patients, model_path)
        eval_run(eval_args, params)
        assert True

    def test_eval_mixed_cat_dqn_model_algo(self, tmpdir):
        out = self.system_setup(tmpdir)
        symptom, condition, patients = out
        args = self.get_args(tmpdir, patients)
        arch = "mixed_cat_dqn_model"
        agent = "mixed_catdqnagent"
        algo = "mixed_categoricaldqn"
        params = self.get_params(symptom, condition, arch, agent, algo)
        cli._HAS_REPORTED_RESULTS = False
        mlflow.start_run()
        uid = mlflow.active_run().info.run_id
        exp_name = params["exp_name"]
        pref = f"{exp_name}/{uid}/"
        run(args, params)
        mlflow.end_run()
        model_path = os.path.join(tmpdir, f"{pref}run_{args.run_ID}/params.pkl")
        eval_args = self.get_eval_args(tmpdir, patients, model_path)
        eval_run(eval_args, params)
        assert True

    def test_eval_mixed_dqn_model_algo_with_traj_aux_reward_log(self, tmpdir):
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
        uid = mlflow.active_run().info.run_id
        exp_name = params["exp_name"]
        pref = f"{exp_name}/{uid}/"
        run(args, params)
        mlflow.end_run()
        model_path = os.path.join(tmpdir, f"{pref}run_{args.run_ID}/params.pkl")
        eval_args = self.get_eval_args(tmpdir, patients, model_path)
        eval_run(eval_args, params)
        assert True

    def test_eval_mixed_dqn_model_algo_with_diff_traj_aux_reward_log(self, tmpdir):
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
        uid = mlflow.active_run().info.run_id
        exp_name = params["exp_name"]
        pref = f"{exp_name}/{uid}/"
        run(args, params)
        mlflow.end_run()
        model_path = os.path.join(tmpdir, f"{pref}run_{args.run_ID}/params.pkl")
        eval_args = self.get_eval_args(tmpdir, patients, model_path)
        eval_run(eval_args, params)
        assert True
