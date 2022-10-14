import numpy as np
import torch

from chloe.utils.tensor_utils import (
    _get_target_categorical_distributional,
    _negate_tensor,
    get_nb_pathos,
    get_nb_severe_pathos,
)


class TestTensorUtils(object):
    def test_get_nb_severe_pathos(self):
        preds = torch.from_numpy(np.array([[0.2, 0.3, 0.1, 0.4, 0.0]]))
        diff_indices = torch.from_numpy(np.array([[1, 4, 2]]))
        diff_probas = torch.from_numpy(np.array([[0.5, 0.4, 0.1]]))
        severity = torch.from_numpy(np.array([2, 1, 4, 4, 2]))
        sev_thres = 3
        _, out1, in1, gt1, _ = get_nb_severe_pathos(
            preds, None, diff_indices, diff_probas, severity, sev_thres
        )
        assert out1 == 1
        assert in1 == 1
        assert gt1 == 2
        _, out2, in2, gt2, _ = get_nb_severe_pathos(
            preds, None, diff_indices, diff_probas, severity, sev_thres, 0.3
        )
        assert out2 == 0
        assert in2 == 2
        assert gt2 == 2

    def test_get_nb_pathos(self):
        preds = torch.from_numpy(np.array([[0.2, 0.3, 0.1, 0.4, 0.0]]))
        diff_indices = torch.from_numpy(np.array([[1, 4, 2]]))
        diff_probas = torch.from_numpy(np.array([[0.5, 0.4, 0.1]]))
        _, out1, in1, gt1, _ = get_nb_pathos(
            preds, None, diff_indices, diff_probas, None, 3, severe_flag=False
        )
        assert out1 == 2
        assert in1 == 1
        assert gt1 == 3
        _, out2, in2, gt2, _ = get_nb_pathos(
            preds, None, diff_indices, diff_probas, None, 3, 0.3, severe_flag=False
        )
        assert out2 == 1
        assert in2 == 2
        assert gt2 == 2

    def test_negate_tensors(self):
        preds = np.array([[7, 8, 9], [1, 2, 3], [4, 5, 6], [3, 9, 8], [7, 5, 3]])

        assert np.all(_negate_tensor(preds) == -preds)

    def test_get_target_categorical_distributional(self):
        n_atoms = 29
        v_min, v_max = np.random.rand(2)
        if v_min > v_max:
            v_min, v_max = v_max, v_min
        dim_size = [5]
        target = np.random.rand(*dim_size) * (v_max - v_min) + v_min
        pi_src = np.random.rand(*dim_size, n_atoms)
        pi_src = pi_src / pi_src.sum(axis=-1, keepdims=True)
        lin_z = torch.linspace(v_min, v_max, n_atoms)

        target_p = _get_target_categorical_distributional(
            torch.from_numpy(target.astype(np.float32)),
            torch.from_numpy(pi_src.astype(np.float32)),
            v_min,
            v_max,
            lin_z,
        )
        computed_val = torch.tensordot(target_p, lin_z, dims=1).numpy()

        assert np.allclose(
            target, computed_val, rtol=1e-2
        ), f"{v_min} - {v_max}, {pi_src.tolist()}"
