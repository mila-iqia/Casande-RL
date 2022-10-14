import os

import pandas as pd

from chloe.preprocessing.preprocessing_filter import filter_data


class TestPreProcessing(object):
    def test_filter(self, tmpdir):

        sample_symptoms = [
            "{",
            '    "sy1": {',
            '        "is_antecedent": true,',
            '        "name": "sy1"',
            "    },",
            '    "sy2": {',
            '        "is_antecedent": true,',
            '        "name": "sy2"',
            "    },",
            '    "sy3": {',
            '        "is_antecedent": true,',
            '        "name": "sy3"',
            "    },",
            '    "sy4": {',
            '        "is_antecedent": false,',
            '        "name": "sy4"',
            "    },",
            '    "sy5": {',
            '        "is_antecedent": false,',
            '        "name": "sy5"',
            "    },",
            '    "sy6": {',
            '        "is_antecedent": false,',
            '        "name": "sy6"',
            "    },",
            '    "sy7": {',
            '        "is_antecedent": false,',
            '        "type-donnes": "C",',
            '        "possible-values": ["a", "b", "c"],',
            '        "name": "sy7"',
            "    },",
            '    "sy8": {',
            '        "is_antecedent": true,',
            '        "name": "sy8"',
            "    },",
            '    "sy9": {',
            '        "is_antecedent": true,',
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
            "id2,M,white,nonhispanic,0,0,First5,3,sy1:32;sy2:38;sy3:48",
            "id3,M,white,nonhispanic,1,1,Even,3,sy6:30;sy2:42;sy8:35",
            "id4,M,white,nonhispanic,4,4,Odd,3,sy7_@_a:39;sy3:42;sy9:28;sy5:1",
            "id5,M,white,nonhispanic,8,8,Mult3,2,sy6:34;sy9:37",
            "id6,M,white,nonhispanic,9,9,Prime,3,sy1:32;sy2:38;sy5:31",
            "id7,M,white,nonhispanic,9,9,Prime,0,",
            "id8,M,white,nonhispanic,1,1,EvenOdd,3,sy6:30;sy2:42;sy8:35",
            "id9,M,white,nonhispanic,4,4,Prime,1,sy7_@_c:39",
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

        # tmp_file patients
        patients = tmpdir.join("patients.csv")
        sample_data_patient = "\n".join(sample_patients)
        patients.write(sample_data_patient)
        filename_patients = os.path.join(tmpdir, "patients.csv")

        filter_data(
            filename_symptom, filename_condition, filename_patients, tmpdir, "filtered"
        )

        filtered_filename_patients = os.path.join(tmpdir, "filtered_patients.zip")

        df = pd.read_csv(filtered_filename_patients, sep=",")

        assert len(df) == 5

        # row with only binary antecedent symptoms get filtered
        assert "id2" not in df["PATIENT"].values

        # rows with 0 symptoms get filtered
        assert "id7" not in df["PATIENT"].values

        # rows with only derivated symptoms get filtered
        assert "id9" not in df["PATIENT"].values

        # no defined pathology get filtered
        assert "id8" not in df["PATIENT"].values
