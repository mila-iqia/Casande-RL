import numpy as np
import pytest
import torch
from numpy.testing import assert_array_almost_equal

from chloe.utils.clf_loss_utils import (
    cross_entropy_loss,
    sigmoid_modulated_cross_entropy_and_entropy_loss,
    sigmoid_modulated_cross_entropy_and_entropy_neg_reward,
)
from chloe.utils.reward_shaping_utils import (
    ce_ent_sent_reshaping,
    cross_entropy_reshaping,
)
from chloe.utils.scheduler_utils import sigmoid_scheduler


class TestRewardShapingUtils(object):
    """Implements tests for the reward shaping functions."""

    def setup(self):
        device = torch.device("cpu")

        self.prev_values = torch.tensor(
            [[0.7, 0.2, 0.1], [-24.1, -10.7, -1.2], [0, 1, 0], [1, 1, 1]]
        ).to(device)
        self.next_values = torch.tensor(
            [[0.2, 0.6, 0.2], [-10.7, -1.2, -24.1], [1, 0, 0], [1, 1, 1]]
        ).to(device)
        self.target_pathos = torch.tensor([1, 0, 1, 2]).to(device)
        self.severity = torch.tensor([1, 2, 4]).to(device)
        self.evidence = torch.tensor([0, 1, 1, 1]).to(device)
        self.timestep = torch.tensor([0, 1, 7, 12]).float().to(device)
        self.discount_factor = 0.99
        self.rs_schedule_params = dict(
            ce_alpha=5,
            ent_alpha=5,
            js_alpha=5,
            tv_alpha=5,
            sev_ent_alpha=8,
            sev_ent_alpha_b=1,
            link_div_with_negative_evidence=False,
            normalize_sev_dist=True,
            severity_threshold=3,
            max_turns=30,
            min_turns=0,
            min_map_val=-5,
            max_map_val=5,
        )
        self.rs_coefficients = dict(
            ent_weight=0.0,
            ce_weight=0.0,
            js_weight=0.0,
            tv_weight=0.0,
            sev_ent_weight=0.0,
            sev_tv_weight=0.0,
            sev_js_weight=0.0,
        )
        self.rs_clamp_value = {
            "ent_min": 0,
            "ent_max": 5,
            "js_min": 0,
            "js_max": 5,
            "tv_min": 0,
            "tv_max": 5,
            "ce_min": -1,
            "ce_max": 1,
            "sev_js_min": 0,
            "sev_js_max": 5,
            "sev_tv_min": 0,
            "sev_tv_max": 5,
            "sev_ent_min": 0,
            "sev_ent_max": 5,
        }
        self.clf_schedule_params = dict(
            alpha=5,
            use_severity_as_weight=False,
            ent_weight=1.0,
            max_turns=30,
            min_turns=0,
            min_map_val=-5,
            max_map_val=5,
        )

    def test_js_divergence_reshaping(self):
        # Setup
        output = np.array([0.013, 0.417, 0.011, 0.0])

        # Compute
        coeffs = dict(self.rs_coefficients)
        coeffs["js_weight"] = 1.0
        computed_output, _ = ce_ent_sent_reshaping(
            self.prev_values,
            self.next_values,
            self.target_pathos,
            None,
            None,
            self.evidence,
            discount=self.discount_factor,
            severity=self.severity,
            timestep=self.timestep,
            **self.rs_schedule_params,
            **coeffs,
            **self.rs_clamp_value,
        )
        computed_output = computed_output.cpu().numpy().flatten()

        # Verify
        assert_array_almost_equal(output, computed_output, decimal=3)

    def test_cross_entropy_reshaping(self):
        # Setup
        output = np.array([0.4260, 1.3495e01, -0.9845, 0.0109])

        # Compute
        computed_output, _ = cross_entropy_reshaping(
            self.next_values,
            self.prev_values,
            self.target_pathos,
            None,
            None,
            self.evidence,
            self.discount_factor,
            severity=self.severity,
            timestep=self.timestep,
        )
        computed_output = computed_output.cpu().numpy().flatten()

        # Verify
        assert_array_almost_equal(output, computed_output, decimal=3)

    @pytest.mark.parametrize(
        "alpha, max_turns, min_turns, min_map_val, max_map_val, inc_dec, output",
        [
            (0, 30, 0, -5, 5, True, [0.0067, 0.0093, 0.0650, 0.2689]),
            (0, 30, 0, -5, 5, False, [0.9933, 0.9907, 0.9350, 0.7311]),
        ],
    )
    def test_get_weight(
        self, alpha, max_turns, min_turns, min_map_val, max_map_val, inc_dec, output
    ):
        # Compute
        computed_output = sigmoid_scheduler(
            self.timestep,
            alpha,
            max_turns,
            min_turns,
            min_map_val,
            max_map_val,
            not inc_dec,
        )
        computed_output = computed_output.cpu().numpy().flatten()

        # Verify
        assert_array_almost_equal(output, computed_output, decimal=3)

    def test_severe_patho_jsd_reshaping(self):
        # Setup
        output = np.array([3.382e-02, 3.157e-05, 1.553e-01, 0.000e+00])

        # Compute
        coeffs = dict(self.rs_coefficients)
        coeffs["sev_js_weight"] = 1.0
        computed_output, _ = ce_ent_sent_reshaping(
            self.prev_values,
            self.next_values,
            self.target_pathos,
            None,
            None,
            self.evidence,
            discount=self.discount_factor,
            severity=self.severity,
            timestep=self.timestep,
            **self.rs_schedule_params,
            **coeffs,
            **self.rs_clamp_value,
        )
        computed_output = computed_output.cpu().numpy().flatten()

        # Verify
        assert_array_almost_equal(output, computed_output, decimal=3)

    def test_r3_reshaping(self):
        # Setup
        output = np.array([-5.3957e-02, -7.2558e+00,  6.2476e-01,  2.6971e-03])

        # Compute
        coeffs = dict(self.rs_coefficients)
        coeffs["sev_js_weight"] = 1.0
        coeffs["js_weight"] = 1.0
        coeffs["ce_weight"] = 1.0
        computed_output, _ = ce_ent_sent_reshaping(
            self.prev_values,
            self.next_values,
            self.target_pathos,
            None,
            None,
            self.evidence,
            discount=self.discount_factor,
            severity=self.severity,
            timestep=self.timestep,
            use_severity_as_weight=True,
            **self.rs_schedule_params,
            **coeffs,
            **self.rs_clamp_value,
        )
        computed_output = computed_output.cpu().numpy().flatten()

        # Verify
        assert_array_almost_equal(output, computed_output, decimal=4)

    def test_cross_entropy_loss(self):
        # Setup
        output = np.array([0.85, 9.5, 1.551, 1.099])

        # Compute
        computed_output, _ = cross_entropy_loss(
            self.next_values,
            self.target_pathos,
            None,
            None,
            None,
            reduction="none",
            severity=self.severity,
            timestep=self.timestep,
        )
        computed_output = computed_output.cpu().numpy().flatten()

        # Verify
        assert_array_almost_equal(output, computed_output, decimal=3)

    def test_sigmoid_modulated_cross_entropy_and_entropy_loss(self):
        # Setup
        output = np.array([0.965, 5.535, 2.303, 2.158])

        # Compute
        computed_output, _ = sigmoid_modulated_cross_entropy_and_entropy_loss(
            self.next_values,
            self.target_pathos,
            None,
            None,
            None,
            reduction="none",
            severity=self.severity,
            timestep=self.timestep,
            **self.clf_schedule_params,
        )
        computed_output = computed_output.cpu().numpy().flatten()

        # Verify
        assert_array_almost_equal(output, computed_output, decimal=3)

    def test_sigmoid_modulated_cross_entropy_and_entropy_neg_reward(self):
        # Setup
        output = np.array([1.833, 9.501, 2.439, 2.099])

        # Compute
        computed_output, _ = sigmoid_modulated_cross_entropy_and_entropy_neg_reward(
            self.next_values,
            self.target_pathos,
            None,
            None,
            None,
            reduction="none",
            severity=self.severity,
            timestep=self.timestep,
            initial_penalty=0.0,
            **self.clf_schedule_params,
        )
        computed_output = computed_output.cpu().numpy().flatten()

        # Verify
        assert_array_almost_equal(output, computed_output, decimal=3)
