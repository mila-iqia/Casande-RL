import collections
import os
import random

import numpy as np

from chloe.simulator.simulator import PatientInteractionSimulator
from chloe.utils.sim_utils import (
    encode_age,
    encode_ethnicity,
    encode_race,
    encode_sex,
    get_symptoms_with_multiple_answers,
    load_and_check_data,
    load_csv,
    only_contain_derivated_symptoms,
    preprocess_differential,
    preprocess_symptoms,
    stringify_differential,
)


class TestSimulator(object):
    def test_preprocess_symptoms(self):
        target = ["Nasal congestion", "Sore throat", "Vomiting"]

        symptoms_st = "Nasal congestion:30;Sore throat:42;Vomiting:35"
        result = preprocess_symptoms(symptoms_st)
        assert target == result

        symptoms_st = "Sore throat:42;Vomiting:35;Nasal congestion:30"
        result = preprocess_symptoms(symptoms_st)
        assert target == result

        symptoms_st = "Vomiting:35;Sore throat:42;Nasal congestion:30"
        result = preprocess_symptoms(symptoms_st)
        assert target == result

    def test_only_contain_derivated_symptoms(self):
        target = ["Nasal_@_congestion", "Sore_@_throat", "douleurxx_intensity_@_1"]
        assert only_contain_derivated_symptoms(target)

        target2 = ["Nasal_@_congestion", "Sore throat", "douleurxx_intensity_@_8"]
        assert not only_contain_derivated_symptoms(target2)

    def test_get_symptoms_with_multiple_answers(self):
        target = [
            "pain_body_@_head",
            "pain_body_@_hand",
            "douleurxx_intensity_@_5",
            "fever",
            "pain_irrad_@_leg",
            "pain_irrad_@_finger",
            "pain_irrad_@_chest",
            "skin_couleur_@_red",
        ]
        result = set(get_symptoms_with_multiple_answers(target))
        expected_result = set(["pain_body", "pain_irrad"])
        assert result == expected_result

    def test_encode_age(self):
        assert encode_age(0) == 0
        assert encode_age(3) == 1
        assert encode_age(10) == 2
        assert encode_age(25) == 3
        assert encode_age(40) == 4
        assert encode_age(50) == 5
        assert encode_age(70) == 6
        assert encode_age(80) == 7
        assert encode_age(100) == 7

    def test_encode_race(self):
        assert encode_race("white") == 0
        assert encode_race("black") == 1
        assert encode_race("asian") == 2
        assert encode_race("native") == 3
        assert encode_race("other") == 4
        assert encode_race("blue") == 4

    def test_encode_ethnicity(self):
        assert encode_ethnicity("hispanic") == 0
        assert encode_ethnicity("nonhispanic") == 1
        try:
            encode_ethnicity("something_weird")
            assert False
        except Exception:
            assert True

    def test_encode_sex(self):
        assert encode_sex("M") == 0
        assert encode_sex("F") == 1
        try:
            encode_sex("something_weird")
            assert False
        except Exception:
            assert True

    def test_load_and_check_data(self, tmpdir):
        sample_symptoms = [
            "{",
            '    "abdominal-distention": {',
            '        "name": "Abdominal distention",',
            '        "hash": "02c0f56b1824ed7adaa714f76f69da0c77f60c85fab7"',
            "    },",
            '    "abdominal-pain": {',
            '        "name": "Abdominal pain",',
            '        "hash": "8f2c2f06a10a80de2d46ff07627b5d6cd6a19cb4ebc2"',
            "    },",
            '    "abnormal-appearing-skin": {',
            '        "name": "Abnormal appearing skin",',
            '        "hash": "5c94f181def3220a5ab03fdcc58150df5cacfb3c18c4"',
            "    },",
            '    "abnormal-appearing-tongue": {',
            '        "name": "Abnormal appearing tongue",',
            '        "hash": "98f6fac90b90f659c2a0ae623cd913c6f88d298dbc98"',
            "    },",
            '    "abnormal-appetite": {',
            '        "name": "Abnormal appetite",',
            '        "hash": "223a0f6bcd513658da2134797bd70803dcc02851093b"',
            "    },",
            '    "abnormal-body-temperature": {',
            '        "name": "Abnormal body temperature",',
            '        "hash": "fba63724ff660b4d1ab8eec0ba07a1984cbcdc2ec181"',
            "    }",
            "}",
        ]
        symptoms = tmpdir.join("symptoms.json")
        sample_data = "\n".join(sample_symptoms)
        symptoms.write(sample_data)
        filename_symptom = os.path.join(tmpdir, "symptoms.json")
        provided_data = [
            "Abdominal pain",
            "Abnormal appearing tongue",
            "Abdominal distention",
        ]

        index_2_key, _, _ = load_and_check_data(filename_symptom, provided_data, "name")

        keys = [
            "abnormal-body-temperature",
            "abnormal-appetite",
            "abnormal-appearing-tongue",
            "abnormal-appearing-skin",
            "abdominal-pain",
            "abdominal-distention",
        ]

        assert index_2_key == sorted(keys)

        wrong_provided_data = ["Abdominal pain", "toto", "tata"]
        try:
            load_and_check_data(filename_symptom, wrong_provided_data, "name")
            assert False
        except Exception:
            assert True

    def test_load_csv(self, tmpdir):
        header1 = "PATIENT,GENDER,RACE,ETHNICITY,AGE_BEGIN"
        header2 = "AGE_END,PATHOLOGY,NUM_SYMPTOMS,SYMPTOMS"
        sample_patients = [
            header1 + "," + header2,
            "id1,M,white,nonhispanic,9,9,Adrenal cancer,0,",
            "id2,M,white,nonhispanic,0,0,Embolism,5,sy1:32;sy2:38;sy3:48;sy4:47;sy5:31",
            "id3,M,white,nonhispanic,1,1,Strep throat,4,sy6:30;sy1:42;sy3:35;sy4:20",
            "id4,M,white,nonhispanic,4,4,Seborrheic dermatitis,3,sy7:39;sy8:42;sy9:28",
            "id5,M,white,nonhispanic,8,8,Adrenal cancer,2,sy2:34;sy4:37",
            "id6,M,white,nonhispanic,9,9,Lupus,0,",
        ]

        unique_sy = ["sy" + str(i) for i in range(1, 10)]
        unique_pa = [
            "Adrenal cancer",
            "Embolism",
            "Seborrheic dermatitis",
            "Strep throat",
        ]
        patients = tmpdir.join("patients.csv")
        sample_data = "\n".join(sample_patients)
        patients.write(sample_data)
        filename_patients = os.path.join(tmpdir, "patients.csv")

        (
            df,
            unique_symptoms,
            unique_pathologies,
            pathology_symptoms,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = load_csv(filename_patients)
        pathology_symptoms = {
            unique_pathologies[a]: set(
                [unique_symptoms[b] for b in pathology_symptoms[a]]
            )
            for a in pathology_symptoms.keys()
        }

        assert len(df) == 4
        assert sorted(unique_symptoms) == unique_sy
        assert sorted(unique_pathologies) == sorted(unique_pa)
        assert sorted(pathology_symptoms.keys()) == sorted(unique_pa)
        assert pathology_symptoms["Adrenal cancer"] == set(["sy2", "sy4"])
        assert pathology_symptoms["Embolism"] == set(
            ["sy1", "sy2", "sy5", "sy4", "sy3"]
        )
        assert pathology_symptoms["Seborrheic dermatitis"] == set(["sy7", "sy9", "sy8"])
        assert pathology_symptoms["Strep throat"] == set(["sy1", "sy6", "sy4", "sy3"])

    def test_simulator(self, tmpdir):
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
            "id1,M,white,nonhispanic,9,9,First5,3,sy2:38;sy4:47;sy5:31",
            "id2,M,white,nonhispanic,0,0,First5,5,sy1:32;sy2:38;sy3:48;sy4:47;sy5:31",
            "id3,M,white,nonhispanic,1,1,Even,3,sy6:30;sy2:42;sy8:35",
            "id4,M,white,nonhispanic,4,4,Odd,3,sy7:39;sy3:42;sy9:28",
            "id5,M,white,nonhispanic,8,8,Mult3,2,sy6:34;sy9:37",
            "id6,M,white,nonhispanic,9,9,Prime,3,sy1:32;sy2:38;sy5:31",
        ]
        non_consistent_sample_patients = [
            header1 + "," + header2,
            "id1,M,white,nonhispanic,9,9,First5,3,sy6:38;sy7:47;sy8:31",
            "id3,M,white,nonhispanic,1,1,Even,3,sy7:30;sy1:42;sy9:35",
            "id4,M,white,nonhispanic,4,4,Odd,3,sy8:39;sy3:42;sy9:28",
            "id5,M,white,nonhispanic,8,8,Mult3,2,sy2:34;sy1:37",
            "id6,M,white,nonhispanic,9,9,Prime,3,sy1:32;sy2:38;sy9:31",
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

        # tmp_file samples
        non_con_patients = tmpdir.join("non_con_patients.csv")
        non_con_sample_data_patient = "\n".join(non_consistent_sample_patients)
        non_con_patients.write(non_con_sample_data_patient)
        non_con_filename_patients = os.path.join(tmpdir, "non_con_patients.csv")

        try:
            env = PatientInteractionSimulator(
                non_con_filename_patients,
                filename_symptom,
                filename_condition,
                max_turns=10,
            )
            assert False
        except Exception:
            assert True

        # render file
        render_node = tmpdir.join("render.txt")
        filename_render = os.path.join(tmpdir, "render.txt")

        env = PatientInteractionSimulator(
            filename_patients,
            filename_symptom,
            filename_condition,
            max_turns=10,
            action_type=0,
        )

        obs = env.reset()
        assert len(obs) == 13
        num_turns = 0
        while True:
            action = random.randint(0, 13)
            obs, rewards, done, info = env.step(action)
            num_turns += 1
            if done:
                env.render(mode="all", filename=filename_render)
                break

        content_render = render_node.read()
        alist = content_render.split("\n")
        assert len(alist) > 9 + num_turns

        # action_type : 1 (tuple)
        # render file
        render_node = tmpdir.join("render1.txt")
        filename_render = os.path.join(tmpdir, "render1.txt")

        env = PatientInteractionSimulator(
            filename_patients,
            filename_symptom,
            filename_condition,
            max_turns=10,
            action_type=1,
        )
        num_symps = 9
        num_pathos = 5

        obs = env.reset()
        assert len(obs) == 13
        num_turns = 0
        while True:
            act_type = random.choices([0, 1], weights=[0.8, 0.2])[0]
            act_symp = random.randint(0, num_symps - 1)
            act_path = random.randint(0, num_pathos - 1)
            action = [act_type, act_symp, act_path]
            obs, rewards, done, info = env.step(action)
            num_turns += 1
            if done:
                env.render(mode="all", filename=filename_render)
                break

        content_render = render_node.read()
        alist = content_render.split("\n")
        assert len(alist) > 9 + num_turns

        # action_type : 1 (tuple)
        # include_turns_in_state : True (tuple)
        # render file
        render_node = tmpdir.join("render1.txt")
        filename_render = os.path.join(tmpdir, "render1.txt")

        env = PatientInteractionSimulator(
            filename_patients,
            filename_symptom,
            filename_condition,
            max_turns=10,
            action_type=1,
            include_turns_in_state=True,
        )
        num_symps = 9
        num_pathos = 5

        obs = env.reset()
        assert len(obs) == 14
        num_turns = 0
        while True:
            act_type = random.choices([0, 1], weights=[0.8, 0.2])[0]
            act_symp = random.randint(0, num_symps - 1)
            act_path = random.randint(0, num_pathos - 1)
            action = [act_type, act_symp, act_path]
            obs, rewards, done, info = env.step(action)
            num_turns += 1
            if done:
                env.render(mode="all", filename=filename_render)
                break

        content_render = render_node.read()
        alist = content_render.split("\n")
        assert len(alist) > 9 + num_turns

        assert True

    def test_simulator_categorical(self, tmpdir):
        """Tests the simulator ability to handle categorical/multi-choice symptom data.
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
        non_consistent_sample_patients = [
            header1 + "," + header2,
            "id1,M,white,nonhispanic,9,9,First5,3,sy6:38;sy7:47;sy8:31",
            "id3,M,white,nonhispanic,1,1,Even,3,sy7:30;sy1_@_1:42;sy9:35",
            "id4,M,white,nonhispanic,4,4,Odd,3,sy8:39;sy3:42;sy9:28",
            "id5,M,white,nonhispanic,8,8,Mult3,2,sy2_@_C:34;sy1_@_2:37",
            "id6,M,white,nonhispanic,9,9,Prime,3,sy1_@_1:32;sy2_@_B:38;sy9:31",
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

        # tmp_file samples
        non_con_patients = tmpdir.join("non_con_patients.csv")
        non_con_sample_data_patient = "\n".join(non_consistent_sample_patients)
        non_con_patients.write(non_con_sample_data_patient)
        non_con_filename_patients = os.path.join(tmpdir, "non_con_patients.csv")

        try:
            env = PatientInteractionSimulator(
                non_con_filename_patients,
                filename_symptom,
                filename_condition,
                max_turns=10,
            )
            assert False
        except Exception:
            assert True

        # action_type : 0 (int)
        # categorical and multi-choice symptom
        # render file
        render_node = tmpdir.join("render.txt")
        filename_render = os.path.join(tmpdir, "render.txt")

        env = PatientInteractionSimulator(
            filename_patients,
            filename_symptom,
            filename_condition,
            max_turns=10,
            action_type=0,
        )

        sympt, val = env.get_symptom_and_value("sy1_@_3")
        assert sympt == "sy1"
        assert val == 3

        sympt, val = env.get_symptom_and_value("sy2_@_B")
        assert sympt == "sy2"
        assert val == "B"

        obs = env.reset()
        assert len(obs) == 17
        num_turns = 0
        while True:
            action = random.randint(0, 13)
            obs, rewards, done, info = env.step(action)
            num_turns += 1
            if done:
                env.render(mode="all", filename=filename_render)
                break

        content_render = render_node.read()
        alist = content_render.split("\n")
        assert len(alist) > 9 + num_turns

        # action_type : 1 (tuple)
        # categorical and multi-choice symptom
        # render file
        render_node = tmpdir.join("render1.txt")
        filename_render = os.path.join(tmpdir, "render1.txt")

        env = PatientInteractionSimulator(
            filename_patients,
            filename_symptom,
            filename_condition,
            max_turns=10,
            action_type=1,
        )
        num_symps = 9
        num_pathos = 5

        obs = env.reset()
        assert len(obs) == 17
        num_turns = 0
        while True:
            act_type = random.choices([0, 1], weights=[0.8, 0.2])[0]
            act_symp = random.randint(0, num_symps - 1)
            act_path = random.randint(0, num_pathos - 1)
            action = [act_type, act_symp, act_path]
            obs, rewards, done, info = env.step(action)
            num_turns += 1
            if done:
                env.render(mode="all", filename=filename_render)
                break

        content_render = render_node.read()
        alist = content_render.split("\n")
        assert len(alist) > 9 + num_turns

        # action_type : 1 (tuple)
        # categorical and multi-choice symptom
        # include_turns_in_state : True (tuple)
        # render file
        render_node = tmpdir.join("render2.txt")
        filename_render = os.path.join(tmpdir, "render2.txt")

        env = PatientInteractionSimulator(
            filename_patients,
            filename_symptom,
            filename_condition,
            max_turns=10,
            action_type=1,
            include_turns_in_state=True,
        )
        num_symps = 9
        num_pathos = 5

        obs = env.reset()
        assert len(obs) == 18
        num_turns = 0
        while True:
            act_type = random.choices([0, 1], weights=[0.8, 0.2])[0]
            act_symp = random.randint(0, num_symps - 1)
            act_path = random.randint(0, num_pathos - 1)
            action = [act_type, act_symp, act_path]
            obs, rewards, done, info = env.step(action)
            num_turns += 1
            if done:
                env.render(mode="all", filename=filename_render)
                break

        content_render = render_node.read()
        alist = content_render.split("\n")
        assert len(alist) > 9 + num_turns

    def test_simulator_demographic_feature_inclusion(self, tmpdir):
        """Tests the simulator ability to handle demo feature inclusion/exclusion.
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

        # action_type : 0 (int)
        # include_race_in_state
        # render file
        render_node = tmpdir.join("render3.txt")
        filename_render = os.path.join(tmpdir, "render3.txt")

        env = PatientInteractionSimulator(
            filename_patients,
            filename_symptom,
            filename_condition,
            max_turns=10,
            action_type=0,
            include_race_in_state=False,
        )

        obs = env.reset()
        assert len(obs) == 16
        num_turns = 0
        while True:
            action = random.randint(0, 13)
            obs, rewards, done, info = env.step(action)
            num_turns += 1
            if done:
                env.render(mode="all", filename=filename_render)
                break

        content_render = render_node.read()
        alist = content_render.split("\n")
        assert len(alist) > 9 + num_turns
        assert "not included" in alist[3]

        # action_type : 0 (int)
        # include_ethnicity_in_state
        # render file
        render_node = tmpdir.join("render4.txt")
        filename_render = os.path.join(tmpdir, "render4.txt")

        env = PatientInteractionSimulator(
            filename_patients,
            filename_symptom,
            filename_condition,
            max_turns=10,
            action_type=0,
            include_ethnicity_in_state=False,
        )

        obs = env.reset()
        assert len(obs) == 16
        num_turns = 0
        while True:
            action = random.randint(0, 13)
            obs, rewards, done, info = env.step(action)
            num_turns += 1
            if done:
                env.render(mode="all", filename=filename_render)
                break

        content_render = render_node.read()
        alist = content_render.split("\n")
        assert len(alist) > 9 + num_turns
        assert "not included" in alist[4]

        # action_type : 0 (int)
        # include_race_in_state
        # include_ethnicity_in_state
        # render file
        render_node = tmpdir.join("render5.txt")
        filename_render = os.path.join(tmpdir, "render5.txt")

        env = PatientInteractionSimulator(
            filename_patients,
            filename_symptom,
            filename_condition,
            max_turns=10,
            action_type=0,
            include_race_in_state=False,
            include_ethnicity_in_state=False,
        )

        obs = env.reset()
        assert len(obs) == 15
        num_turns = 0
        while True:
            action = random.randint(0, 13)
            obs, rewards, done, info = env.step(action)
            num_turns += 1
            if done:
                env.render(mode="all", filename=filename_render)
                break

        content_render = render_node.read()
        alist = content_render.split("\n")
        assert len(alist) > 9 + num_turns
        assert "not included" in alist[3]
        assert "not included" in alist[4]

        assert True

    def test_simulator_hierarchical(self, tmpdir):
        """Tests the simulator ability to handle differential printing.
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
        sample_patients = [
            header1 + "," + header2,
            "id3,M,white,nonhispanic,1,1,Even,3,sy6:30;sy2_@_C:42;sy2_@_D:42;sy8:35",
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

        # action_type : 0 (int)
        # differential

        env = PatientInteractionSimulator(
            filename_patients,
            filename_symptom,
            filename_condition,
            max_turns=10,
            action_type=0,
        )

        # num_pathos = 5
        # pathos_names = ["First5", "Mult3", "Prime", "Odd", "Even"]
        obs = env.reset()
        assert len(obs) == 17
        action = env.symptom_name_2_index["sy5"]
        obs, _, _, _ = env.step(action)
        sy1_index = env.symptom_name_2_index["sy1"]
        sy1_beg, sy1_end = env.symptom_to_obs_mapping[sy1_index]
        data = obs[sy1_beg:sy1_end]
        assert np.all(data == 1 / 3)  # default value
        assert True

    def test_simulator_differential(self, tmpdir):
        """Tests the simulator ability to handle differential printing.
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

        # action_type : 0 (int)
        # differential
        # render file
        render_node = tmpdir.join("render6.txt")
        filename_render = os.path.join(tmpdir, "render6.txt")

        env = PatientInteractionSimulator(
            filename_patients,
            filename_symptom,
            filename_condition,
            max_turns=10,
            action_type=0,
        )

        num_pathos = 5
        pathos_names = ["First5", "Mult3", "Prime", "Odd", "Even"]
        obs = env.reset()
        assert len(obs) == 17
        num_turns = 0
        preds = np.array([random.random() for _ in range(num_pathos)])
        argsort_indices = sorted(range(len(preds)), key=preds.__getitem__, reverse=True)
        topk = random.randint(1, num_pathos)
        while True:
            action = random.randint(0, 13)
            obs, rewards, done, info = env.step(action)
            num_turns += 1
            if done:
                env.render(
                    mode="all",
                    filename=filename_render,
                    patho_predictions=preds,
                    num=topk,
                )
                break

        content_render = render_node.read()
        alist = content_render.split("\n")
        assert len(alist) > 9 + num_turns + topk + 1
        assert "DIFFERENTIAL DIAGNOSIS:" in alist[-(topk + 2)]
        for i in range(1, topk + 1):
            idx = argsort_indices[i - 1]
            assert pathos_names[idx] in alist[-(topk + 2 - i)]

        assert True

    def test_simulator_differential_load(self, tmpdir):
        """Tests the simulator ability to handle differential printing.
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

        # test load (filtering on FOUND_GT_PATHOLOGY)
        out = load_csv(filename_patients)
        df = out[0]
        assert len(df) == 4

        # action_type : 0 (int)
        # differential
        # render file
        render_node = tmpdir.join("render10.txt")
        filename_render = os.path.join(tmpdir, "render10.txt")

        env = PatientInteractionSimulator(
            filename_patients,
            filename_symptom,
            filename_condition,
            max_turns=10,
            action_type=0,
        )

        # assertion on differential computation
        assert env.max_differential_len == 3
        differential_dict = collections.OrderedDict()
        differential_dict[2] = [0.25, 5.0]
        differential_dict[0] = [4.0, 1.0]
        ind, probas = env._compute_differential_probs(differential_dict)
        assert np.all(ind == np.array([0, 2, -1], dtype=int))
        assert np.all(probas == np.array([0.8, 0.2, 0], dtype=np.float32))

        # assertion on get_data at index
        x, (id_patho, diff_ind, diff_probas) = env.get_data_at_index(2, False)
        s_ind = np.argsort(diff_probas, axis=-1)
        diff_probas = diff_probas[s_ind[::-1]]
        diff_ind = diff_ind[s_ind[::-1]]
        d = [2, 0, 0, 1] + [1 / 3] + [1, -1, -1, -1, -1] + [-1, -1, -1, 1, -1, -1, 1]
        s = 0.75 + 0.9
        assert np.all(x == np.array(d, dtype=np.float32)), f"{x} - {d}"
        assert id_patho == 1
        assert np.all(diff_ind == np.array([4, 1, -1], dtype=int))
        assert np.all(diff_probas == np.array([0.9 / s, 0.75 / s, 0], dtype=np.float32))

        # assertion on get_data at index with mask
        mask = [1, 1, 1, 1, 0, 0, 1, 1, 0]
        xm, _ = env.get_data_at_index(2, True, mask)
        dm = [2, 0, 0, 1] + [0] + [0, 0, 0, 0, 0] + [0, 0, -1, 1, 0, 0, 1]
        assert np.all(xm == np.array(dm, dtype=np.float32))

        num_pathos = 5
        pathos_names = ["First5", "Mult3", "Prime", "Odd", "Even"]
        obs = env.reset()
        assert len(obs) == 17
        num_turns = 0
        preds = np.array([random.random() for _ in range(num_pathos)])
        argsort_indices = sorted(range(len(preds)), key=preds.__getitem__, reverse=True)
        topk = random.randint(1, num_pathos)
        while True:
            action = random.randint(0, 13)
            obs, rewards, done, info = env.step(action)
            num_turns += 1
            if done:
                env.render(
                    mode="all",
                    filename=filename_render,
                    patho_predictions=preds,
                    num=topk,
                )
                break

        content_render = render_node.read()
        alist = content_render.split("\n")
        assert len(alist) > 9 + num_turns + topk + 1
        assert "DIFFERENTIAL DIAGNOSIS:" in alist[-(topk + 2)]
        for i in range(1, topk + 1):
            idx = argsort_indices[i - 1]
            assert pathos_names[idx] in alist[-(topk + 2 - i)]

        assert True

    def compute_differential_probs(
        self, differential, modeled_pathos, include_other=False
    ):
        other_proba = 0.0
        sum_model_proba = 0.0
        result = []
        for diff_data in differential:
            patho = diff_data[0]
            sommeOR = diff_data[1]
            proba = sommeOR / (1.0 + sommeOR)
            if patho in modeled_pathos:
                sum_model_proba += proba
                result.append([patho, proba])
            else:
                other_proba += proba
        result = sorted(result, key=lambda x: x[1], reverse=True)
        if include_other and other_proba > 0:
            result.append(["Others", other_proba])
            sum_model_proba += other_proba
        for i in range(len(result)):
            result[i][1] = (
                result[i][1] / sum_model_proba
                if sum_model_proba != 0
                else 1.0 / len(result)
            )
        return result

    def test_stringify_differential(self):
        diff = "Laryngite aigue:0.220:1.096;Angine instable:0.205:1.189"
        res1 = self.compute_differential_probs(
            preprocess_differential(diff), ["Laryngite aigue", "Angine instable"], False
        )
        diff2 = stringify_differential(res1)
        res2 = self.compute_differential_probs(
            preprocess_differential(diff2),
            ["Laryngite aigue", "Angine instable"],
            False,
        )
        assert len(res1) == len(res2)
        for i in range(len(res1)):
            assert res1[i][0] == res2[i][0]
            assert abs(res1[i][1] - res2[i][1]) <= 1e-8

    def test_simulator_symptom_relevancy(self, tmpdir):
        """Tests the simulator ability to handle demo feature inclusion/exclusion.
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
            '        "name": "sy3",',
            '        "is_antecedent": true',
            "    },",
            '    "sy4": {',
            '        "name": "sy4"',
            "    },",
            '    "sy5": {',
            '        "name": "sy5"',
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
            '            "sy4": {',
            '                "probability": 28',
            "            },",
            '            "sy5": {',
            '                "probability": 23',
            "            }",
            "        },",
            '        "antecedents": {',
            '            "sy3": {',
            '                "probability": 28',
            "            }",
            "        }",
            "    }",
            "}",
        ]
        header1 = "PATIENT,GENDER,RACE,ETHNICITY,AGE_BEGIN"
        header2 = "AGE_END,PATHOLOGY,NUM_SYMPTOMS,SYMPTOMS"
        sample_patients = [
            header1 + "," + header2,
            "id1,M,white,nonhispanic,9,9,First5,3,sy2_@_A:8;sy1_@_1:4;sy5:9;sy2_@_B:7"
            ";sy3:2",
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

        # action_type : 0 (int)
        # is_reward_relevancy_patient_specific

        env = PatientInteractionSimulator(
            filename_patients,
            filename_symptom,
            filename_condition,
            max_turns=10,
            action_type=0,
            is_reward_relevancy_patient_specific=False,
        )

        env.reset()
        assert env._is_direct_symptom_relevancy(0, 0, env.target_symptoms)
        assert env._is_direct_symptom_relevancy(0, 1, env.target_symptoms)
        assert env._is_direct_symptom_relevancy(0, 2, env.target_symptoms)
        assert env._is_direct_symptom_relevancy(0, 3, env.target_symptoms)
        assert env._is_direct_symptom_relevancy(0, 4, env.target_symptoms)
        out = env._get_relevant_number_of_symptoms_and_antecedent_from_patient(
            env.target_pathology_index,
            env.target_symptoms,
            env.is_reward_relevancy_patient_specific,
        )
        num_symp, num_atcd = out
        assert [num_symp, num_atcd] == [4, 1]

        env2 = PatientInteractionSimulator(
            filename_patients,
            filename_symptom,
            filename_condition,
            max_turns=10,
            action_type=0,
            is_reward_relevancy_patient_specific=True,
        )

        env2.reset()
        assert not env2._is_direct_symptom_relevancy(0, 0, env2.target_symptoms)
        assert env2._is_direct_symptom_relevancy(0, 1, env2.target_symptoms)
        assert env2._is_direct_symptom_relevancy(0, 2, env2.target_symptoms)
        assert not env2._is_direct_symptom_relevancy(0, 3, env2.target_symptoms)
        assert env2._is_direct_symptom_relevancy(0, 4, env2.target_symptoms)
        out2 = env2._get_relevant_number_of_symptoms_and_antecedent_from_patient(
            env2.target_pathology_index,
            env2.target_symptoms,
            env2.is_reward_relevancy_patient_specific,
        )
        num_symp2, num_atcd2 = out2
        assert [num_symp2, num_atcd2] == [2, 1]

    def test_simulator_linked_questions(self, tmpdir):
        """Tests the simulator ability to handle linked questions.
        """
        sample_symptoms = [
            "{",
            '    "sy1": {',
            '        "name": "sy1",',
            '        "type-donnes": "C",',
            '        "default_value": 1,',
            '        "code_question": "sy3",',
            '        "possible-values": [1, 2, 3]',
            "    },",
            '    "sy2": {',
            '        "name": "sy2",',
            '        "type-donnes": "M",',
            '        "default_value": "A",',
            '        "code_question": "sy3",',
            '        "possible-values": ["A", "B", "C", "D", "E"]',
            "    },",
            '    "sy3": {',
            '        "code_question": "sy3",',
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

        # action_type : 0 (int)

        env = PatientInteractionSimulator(
            filename_patients,
            filename_symptom,
            filename_condition,
            max_turns=10,
            action_type=0,
        )
        assert "sy3" in env.all_linked_symptoms
        assert "sy1" in env.all_linked_symptoms["sy3"]
        assert "sy2" in env.all_linked_symptoms["sy3"]
        assert len(env.all_linked_symptoms["sy3"]) == 2
        assert len(env.all_linked_symptoms) == 1

        obs = env.reset()
        assert len(obs) == 17
        sy3_idx = env.symptom_name_2_index["sy3"]
        sy1_idx = env.symptom_name_2_index["sy1"]
        sy2_idx = env.symptom_name_2_index["sy2"]

        sy3_start_idx, sy3_end_idx = env.symptom_to_obs_mapping[sy3_idx][0:2]
        sy1_start_idx, sy1_end_idx = env.symptom_to_obs_mapping[sy1_idx][0:2]
        sy2_start_idx, sy2_end_idx = env.symptom_to_obs_mapping[sy2_idx][0:2]

        tmp_obs = np.zeros(len(obs))
        tmp_obs, no_evid, imp = env._set_symptom_to_observation(
            tmp_obs, [], sy3_idx, sy3_start_idx, sy3_end_idx, False
        )
        assert tmp_obs[sy1_start_idx:sy1_end_idx].tolist() == [1 / 3]
        assert tmp_obs[sy2_start_idx:sy2_end_idx].tolist() == [1, -1, -1, -1, -1]
        assert tmp_obs[sy3_start_idx:sy3_end_idx].tolist() == [-1]
        assert no_evid and (sy1_idx in imp and sy2_idx in imp) and len(imp) == 2

        tmp_obs = np.zeros(len(obs))
        tmp_obs, no_evid, imp = env._set_symptom_to_observation(
            tmp_obs, ["sy3"], sy3_idx, sy3_start_idx, sy3_end_idx, False
        )
        assert tmp_obs[sy1_start_idx:sy1_end_idx].tolist() == [0]
        assert tmp_obs[sy2_start_idx:sy2_end_idx].tolist() == [0, 0, 0, 0, 0]
        assert tmp_obs[sy3_start_idx:sy3_end_idx].tolist() == [1]
        assert not no_evid and (imp is None or len(imp) == 0)

        tmp_obs = np.zeros(len(obs))
        tmp_obs, no_evid, imp = env._set_symptom_to_observation(
            tmp_obs, [], sy2_idx, sy2_start_idx, sy2_end_idx, False
        )
        assert tmp_obs[sy1_start_idx:sy1_end_idx].tolist() == [0]
        assert tmp_obs[sy2_start_idx:sy2_end_idx].tolist() == [1, -1, -1, -1, -1]
        assert tmp_obs[sy3_start_idx:sy3_end_idx].tolist() == [0]
        assert no_evid and (imp is None or len(imp) == 0)

        tmp_obs = np.zeros(len(obs))
        tmp_obs, no_evid, imp = env._set_symptom_to_observation(
            tmp_obs, ["sy2_@_B", "sy2_@_D"], sy2_idx, sy2_start_idx, sy2_end_idx, False
        )
        assert tmp_obs[sy1_start_idx:sy1_end_idx].tolist() == [0]
        assert tmp_obs[sy2_start_idx:sy2_end_idx].tolist() == [-1, 1, -1, 1, -1]
        assert tmp_obs[sy3_start_idx:sy3_end_idx].tolist() == [1]
        assert not no_evid and (sy3_idx in imp) and len(imp) == 1
