import sys

import numpy as np
import pandas as pd
import pytest

from chloe.plotting.generate_plots import (
    create_age_gps,
    extract_per_patho_metrics_data,
    get_age_group,
    get_patho_urgence_map,
)

if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO


class TestPlotting(object):
    """Implements tests for testing generating plots"""

    def setup(self):
        self.setup_patients_data()
        self.setup_conditions_data()
        self.setup_metrics_data()

    def setup_patients_data(self):
        patients_data = StringIO(
            """,PATIENT,GENDER,RACE,ETHNICITY,AGE_BEGIN,AGE_END,PATHOLOGY,NUM_SYMPTOMS,SYMPTOMS
                        0,1,M,white,nonhispanic,4,4,Laryngite aigue,4,a;b_@_v:15;c;d:12
                        1,2,M,white,nonhispanic,70,70,Bronchiectasies,5,a;b_@_v:15;c;d:12;e
                        2,3,F,white,nonhispanic,30,30,RGO,4,b_@_v:15;c;d:12
                        3,4,F,white,nonhispanic,52,52,Bronchiectasies,3,a;c;d:12
                        4,5,M,white,nonhispanic,12,12,Chagas,2,b_@_v:15;d:12"""
        )
        self.patients_dataframe = pd.read_csv(patients_data)

    def setup_conditions_data(self):
        self.conditions = {
            "RGO": {"condition_name": "RGO ee", "urgence": 1},
            "RGO1": {"condition_name": "RGO,ee1\r", "urgence": 2},
            "RGO2": {"condition_name": "RGO ee2\n", "urgence": 3},
        }

    def setup_metrics_data(self):
        self.metrics_data = {
            "pathos": [[], {"P1": 1, "P2": 2}],
            "per_patho": {
                "1": {"f1": 0.5, "precision": 0.6, "recall": 0.7},
                "2": {"f1": 0.6, "precision": 0.5, "recall": 0.8},
            },
        }
        self.metric_to_extract = ["f1", "precision", "recall"]

    @pytest.mark.parametrize(
        "age, output",
        [
            (4, "under 5 yo"),
            (5, "5-16yo"),
            (10, "5-16yo"),
            (16, "16-35yo"),
            (30, "16-35yo"),
            (35, "35-65yo"),
            (50, "35-65yo"),
            (65, "65 and up"),
            (70, "65 and up"),
        ],
    )
    def test_get_age_group(self, age, output):
        # Compute
        computed_age = get_age_group(age)

        # Validate
        assert output == computed_age

    @pytest.mark.parametrize(
        "age", [(-2), (250)],
    )
    def test_age_group_range(self, age):
        # Validate
        with pytest.raises(ValueError):
            _ = get_age_group(age)

    def test_create_age_gps(self):
        # Setup
        output = np.array(["under 5 yo", "65 and up", "16-35yo", "35-65yo", "5-16yo"])

        # Compute
        patients_data_wit_age_gps = create_age_gps(
            self.patients_dataframe, "AGE_GROUPS"
        )
        patient_age_gps = patients_data_wit_age_gps.AGE_GROUPS.to_numpy().flatten()

        # Validate
        assert all(output == patient_age_gps)

    def test_get_patho_urgence_map(self):
        # Setup
        output = {"RGO ee": 1, "RGO ee1 ": 1, "RGO ee2 ": 0}

        # Compute
        computed_output = get_patho_urgence_map(self.conditions)

        # Verify
        assert output == computed_output

    def test_extract_per_patho_metrics_data(self):
        # Setup
        output = {
            "pathos": ["P1", "P2"],
            "f1": [0.5, 0.6],
            "precision": [0.6, 0.5],
            "recall": [0.7, 0.8],
        }

        # Compute
        computed_output = extract_per_patho_metrics_data(
            self.metrics_data, self.metric_to_extract
        )

        # Verify
        assert output == computed_output
