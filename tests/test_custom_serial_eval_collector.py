import os

import torch
from rlpyt.utils.collections import AttrDict

from chloe.evaluator.custom_serial_eval_collector import CustomSerialEvalCollector
from chloe.simulator.simulator import PatientInteractionSimulator
from chloe.utils.train_utils import SimPaTrajEvalInfo


class TestCustomSerialEvalCollector(object):
    """Implements tests for TestCustomSerialEvalCollector"""

    def setup_colllector(self, tmpdir):
        self.setup_env(tmpdir)
        self.eval_collector = CustomSerialEvalCollector(
            envs=self.envs,
            agent=None,
            TrajInfoCls=SimPaTrajEvalInfo,
            max_T=10,
            max_trajectories=2,
            out_path=tmpdir,
            max_generation=1,
            topk=2,
        )

    def setup_env(self, tmpdir):
        sample_symptoms = [
            "{",
            '    "sy1": {',
            '        "name": "sy1"',
            "    },",
            '    "sy2": {',
            '        "name": "sy2"',
            "    },",
            '    "sy3": {',
            '        "name": "sy3"',
            "    },",
            '    "sy6": {',
            '        "name": "sy6"',
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
            '        "urgence": "1",',
            '        "symptoms": {',
            '            "sy1": {',
            '                "probability": 53',
            "            },",
            '            "sy2": {',
            '                "probability": 35',
            "            },",
            '            "sy3": {',
            '                "probability": 28',
            "            }",
            "        }",
            "    },",
            '    "cond2": {',
            '        "condition_name": "Mult3",',
            '        "urgence": "3",',
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
            "    }",
            "}",
        ]
        header1 = "PATIENT,GENDER,RACE,ETHNICITY,AGE_BEGIN"
        header2 = "AGE_END,PATHOLOGY,NUM_SYMPTOMS,SYMPTOMS"
        sample_patients = [
            header1 + "," + header2,
            "id1,M,white,nonhispanic,9,9,First5,3,sy1;sy2;sy3",
            # "id2,F,white,nonhispanic,0,0,Mult3,2,sy3;sy6",
        ]
        # tmp_file symptoms
        filename_symptom = os.path.join(tmpdir, "symptoms.json")
        sample_data_symptom = "\n".join(sample_symptoms)
        with open(filename_symptom, "w") as file:
            file.write(sample_data_symptom)

        # tmp_file conditions
        filename_condition = os.path.join(tmpdir, "conditions.json")
        sample_data_condition = "\n".join(sample_conditions)
        with open(filename_condition, "w") as file:
            file.write(sample_data_condition)

        # tmp_file samples
        filename_patients = os.path.join(tmpdir, "patients.csv")
        sample_data_patient = "\n".join(sample_patients)
        with open(filename_patients, "w") as file:
            file.write(sample_data_patient)

        self.envs = [
            PatientInteractionSimulator(
                filename_patients, filename_symptom, filename_condition
            )
        ]

    def test_generate_trajectory(self, tmpdir):
        # import pdb;pdb.set_trace()
        self.setup_colllector(tmpdir)
        trajectories = [[2, 4, 1, 3, 5]]
        traj_frames = [[0, 1, 0, 0, 0, 1, 1, -1, -1]]
        traj_patho_predictions = [AttrDict(dict(dist_info=torch.tensor([-2.0, -1.0])))]
        patho_counts = {}
        for trajectory, frame, patho_prediction in zip(
            trajectories, traj_frames, traj_patho_predictions
        ):
            self.envs[0].reset()
            self.envs[0].frame = frame
            # import pdb;pdb.set_trace()
            self.envs[0].ordered_actions = trajectory
            self.eval_collector.generate_trajectory(
                self.envs[0].target_pathology,
                patho_counts,
                self.envs[0],
                [patho_prediction],
                0,
            )
            curr_pathology = self.envs[0].target_pathology
            render_node = tmpdir.join(f"trajs/{curr_pathology}.txt")
            content_render = render_node.read()
            alist = content_render.split("\n")
            topk = 2
            assert "DIFFERENTIAL DIAGNOSIS:" in alist[-(topk + 2)]
            assert "Mult3" in alist[-(topk + 1)]
            assert "First5" in alist[-topk]

        assert True
