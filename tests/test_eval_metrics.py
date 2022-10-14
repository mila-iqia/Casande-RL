import numpy as np
import torch

from chloe.utils.dist_metric import (
    dist_accuracy,
    dist_js_div,
    dist_kl_div,
    dist_ncg,
    dist_ndcg,
    dist_total_variation,
    numpy_get_pathos_inout_ratio,
    numpy_get_severe_pathos_inout_ratio,
    numpy_logsoftmax,
    numpy_softmax,
)
from chloe.utils.eval_utils import TopKAccuracy
from chloe.utils.reward_shaping_utils import cross_entropy_reshaping
from chloe.utils.tensor_utils import soft_cross_entropy


class TestEvalMetrics(object):
    def test_numpy_get_severe_pathos_inout_ratio(self):
        preds = np.array([[0.2, 0.3, 0.1, 0.4, 0.0]])
        diff_indices = np.array([[1, 4, 2]])
        diff_probas = np.array([[0.5, 0.4, 0.1]])
        severe_ind = np.array([0, 1, 4])
        out1, in1 = numpy_get_severe_pathos_inout_ratio(
            preds, None, diff_indices, diff_probas, severe_ind
        )
        assert out1 == (3 - 2 - 1) / (3 - 2)
        assert in1 == (2 - 1) / 2
        out2, in2 = numpy_get_severe_pathos_inout_ratio(
            preds, None, diff_indices, diff_probas, severe_ind, 0.3
        )
        assert out2 == (3 - 2 - 0) / (3 - 2)
        assert in2 == (2 - 2) / 2

    def test_numpy_get_pathos_inout_ratio(self):
        preds = np.array([[0.2, 0.3, 0.1, 0.4, 0.0]])
        diff_indices = np.array([[1, 4, 2]])
        diff_probas = np.array([[0.5, 0.4, 0.1]])
        out1, in1 = numpy_get_pathos_inout_ratio(
            preds, None, diff_indices, diff_probas,
        )
        assert out1 == (5 - 3 - 2) / (5 - 3)
        assert in1 == (3 - 1) / 3
        out2, in2 = numpy_get_pathos_inout_ratio(
            preds, None, diff_indices, diff_probas, 0.3
        )
        assert out2 == (5 - 2 - 1) / (5 - 2)
        assert in2 == (2 - 2) / 2

    def test_topkaccuracy(self):
        preds = np.array([[7, 8, 9], [1, 2, 3], [4, 5, 6], [3, 9, 8], [7, 5, 3]])
        gt = np.array([1, 2, 1, 0, 0])

        try:
            TopKAccuracy(0)(gt, preds)
            assert False
        except ValueError:
            assert True
        assert TopKAccuracy(1)(gt, preds) == 0.4
        assert TopKAccuracy(2)(gt, preds) == 0.8
        assert TopKAccuracy(3)(gt, preds) == 1.0
        assert TopKAccuracy(20)(gt, preds) == 1.0

    def test_dist_accuracy(self):
        preds = np.array([7, 8, 9, 5, 4])
        tgt = 0
        tgt_ind = np.array([1, 2, 4])
        tgt_proba = np.array([0.5, 0.2, 0.3])

        # sorted pred indices = [2, 1, 0, 3, 4]
        # sort tgt indices = [1, 4, 2]
        # intersect{[2], [1, 4, 2]} / min(len([2]), len([1, 2, 4]))
        assert dist_accuracy(preds, tgt, tgt_ind, tgt_proba, 1, False) == 1.0
        # intersect{[2, 1], [1, 4, 2]} / min(len([2,1]), len([1, 2, 4]))
        assert dist_accuracy(preds, tgt, tgt_ind, tgt_proba, 2, False) == 1.0
        # intersect{[2, 1, 0], [1, 4, 2]} / min(len([2,1,0]), len([1, 2, 4]))
        assert dist_accuracy(preds, tgt, tgt_ind, tgt_proba, 3, False) == 2 / 3
        # intersect{[2, 1, 0, 3], [1, 4, 2]} / min(len([2,1,0,3]), len([1, 2, 4]))
        assert dist_accuracy(preds, tgt, tgt_ind, tgt_proba, 4, False) == 2 / 3
        # intersect{[2, 1, 0, 3, 4], [1, 4, 2]} / min(len([2,1,0,3,4]), len([1, 2, 4]))
        assert dist_accuracy(preds, tgt, tgt_ind, tgt_proba, 5, False) == 1.0

        # intersect{[2], [1, 4, 2]} / min(len([2]), len([1, 2, 4]))
        assert dist_accuracy(preds, tgt, tgt_ind, tgt_proba, 1, True) == 0.0
        # intersect{[2, 1], [1, 4, 2]} / min(len([2,1]), len([1, 2, 4]))
        assert dist_accuracy(preds, tgt, tgt_ind, tgt_proba, 2, True) == 0.5
        # intersect{[2, 1, 0], [1, 4, 2]} / min(len([2,1,0]), len([1, 2, 4]))
        assert dist_accuracy(preds, tgt, tgt_ind, tgt_proba, 3, True) == 2 / 3
        # intersect{[2, 1, 0, 3], [1, 4, 2]} / min(len([2,1,0,3]), len([1, 2, 4]))
        assert dist_accuracy(preds, tgt, tgt_ind, tgt_proba, 4, True) == 2 / 3
        # intersect{[2, 1, 0, 3, 4], [1, 4, 2]} / min(len([2,1,0,3,4]), len([1, 2, 4]))
        assert dist_accuracy(preds, tgt, tgt_ind, tgt_proba, 5, True) == 1.0

    def test_dist_total_variation(self):
        pred1 = np.array([7, 8, 9, 5, 4])
        pred2 = np.array([4, 8, 9, 5, 7])
        val = dist_total_variation(pred1, pred2)

        p1 = np.exp(pred1 - np.max(pred1))
        p2 = np.exp(pred2 - np.max(pred2))

        p1 /= p1.sum()
        p2 /= p2.sum()
        assert val == np.abs(p1 - p2).sum() * 0.5

    def test_dist_ncg(self):
        preds = np.array([7, 8, 9, 5, 4])
        tgt = 0
        tgt_ind = np.array([1, 2, 4])
        tgt_proba = np.array([0.5, 0.2, 0.3])
        tgt_relevancy = 2 ** tgt_proba - 1
        rel_dict = {i: v for i, v in zip(tgt_ind, tgt_relevancy)}

        num_class = 5
        for k in range(num_class):
            if k not in rel_dict:
                rel_dict[k] = 0.0

        # sorted pred indices = [2, 1, 0, 3, 4]
        # sort tgt indices = [1, 4, 2]
        # {[2]} / [1]
        assert dist_ncg(preds, tgt, tgt_ind, tgt_proba, 1, False) == (
            rel_dict[2] / rel_dict[1]
        )
        # {[2, 1]} / [1, 4]
        assert dist_ncg(preds, tgt, tgt_ind, tgt_proba, 2, False) == (
            rel_dict[2] + rel_dict[1]
        ) / (rel_dict[1] + rel_dict[4])
        # {[2, 1, 0]} / [1, 4, 2]
        assert dist_ncg(preds, tgt, tgt_ind, tgt_proba, 3, False) == (
            rel_dict[2] + rel_dict[1] + rel_dict[0]
        ) / (rel_dict[1] + rel_dict[4] + rel_dict[2])
        # {[2, 1, 0, 3]} / [1, 4, 2]
        assert dist_ncg(preds, tgt, tgt_ind, tgt_proba, 4, False) == (
            rel_dict[2] + rel_dict[1] + rel_dict[0] + rel_dict[3]
        ) / (rel_dict[1] + rel_dict[4] + rel_dict[2])
        # [2, 1, 0, 3, 4] / [1, 4, 2]
        assert dist_ncg(preds, tgt, tgt_ind, tgt_proba, 5, False) == (
            (rel_dict[2] + rel_dict[1] + rel_dict[0] + rel_dict[3] + rel_dict[4])
            / (rel_dict[1] + rel_dict[4] + rel_dict[2])
        )

        # sort tgt indices = [1, 4, 2]
        # {[2]} / [1]
        assert dist_ncg(preds, tgt, tgt_ind, tgt_proba, 1, True) == 0.0
        # {[2, 1]} / [1, 4]
        assert dist_ncg(preds, tgt, tgt_ind, tgt_proba, 2, True) == (rel_dict[1]) / (
            rel_dict[1] + rel_dict[4]
        )
        # {[2, 1, 0]} / [1, 4, 2]
        assert dist_ncg(preds, tgt, tgt_ind, tgt_proba, 3, True) == (
            rel_dict[2] + rel_dict[1]
        ) / (rel_dict[1] + rel_dict[4] + rel_dict[2])
        # {[2, 1, 0, 3]} / [1, 4, 2]
        assert dist_ncg(preds, tgt, tgt_ind, tgt_proba, 4, True) == (
            rel_dict[2] + rel_dict[1] + rel_dict[0] + rel_dict[3]
        ) / (rel_dict[1] + rel_dict[4] + rel_dict[2])
        # [2, 1, 0, 3, 4] / [1, 4, 2]
        assert dist_ncg(preds, tgt, tgt_ind, tgt_proba, 5, True) == (
            (rel_dict[2] + rel_dict[1] + rel_dict[0] + rel_dict[3] + rel_dict[4])
            / (rel_dict[1] + rel_dict[4] + rel_dict[2])
        )

    def test_dist_ndcg(self):
        preds = np.array([7, 8, 9, 5, 4])
        tgt = 0
        tgt_ind = np.array([1, 2, 4])
        tgt_proba = np.array([0.5, 0.2, 0.3])

        assert round(dist_ndcg(preds, tgt, tgt_ind, tgt_proba, 1, False), 4) == 0.3590
        assert round(dist_ndcg(preds, tgt, tgt_ind, tgt_proba, 2, False), 4) == 0.7321
        assert round(dist_ndcg(preds, tgt, tgt_ind, tgt_proba, 3, False), 4) == 0.6463
        assert round(dist_ndcg(preds, tgt, tgt_ind, tgt_proba, 4, False), 4) == 0.6463
        assert round(dist_ndcg(preds, tgt, tgt_ind, tgt_proba, 5, False), 4) == 0.7873

        assert round(dist_ndcg(preds, tgt, tgt_ind, tgt_proba, 1, True), 4) == 0.0
        assert round(dist_ndcg(preds, tgt, tgt_ind, tgt_proba, 2, True), 4) == 0.4666
        assert round(dist_ndcg(preds, tgt, tgt_ind, tgt_proba, 3, True), 4) == 0.6463
        assert round(dist_ndcg(preds, tgt, tgt_ind, tgt_proba, 4, True), 4) == 0.6463
        assert round(dist_ndcg(preds, tgt, tgt_ind, tgt_proba, 5, True), 4) == 0.7873

    def test_numpy_softmax(self):
        a = np.random.rand(3, 8)
        t = torch.from_numpy(a)
        prob = torch.nn.functional.softmax(t, dim=1)
        prob_np = numpy_softmax(a, axis=1)
        assert np.allclose(prob_np, prob.numpy()), f"{prob_np} - {prob.numpy()}"

    def test_numpy_logsoftmax(self):
        a = np.random.rand(3, 8)
        t = torch.from_numpy(a)
        log_prob = torch.nn.functional.log_softmax(t, dim=1)
        log_prob_np = numpy_logsoftmax(a, axis=1)
        assert np.allclose(log_prob_np, log_prob.numpy()), f"{log_prob_np} - {log_prob}"

    def test_numpy_kl_div(self):
        a = np.random.rand(3, 8)
        b = np.random.rand(3, 8)
        t = torch.from_numpy(a)
        x = torch.from_numpy(b)
        t = torch.nn.functional.softmax(t, dim=1)
        x = torch.nn.functional.log_softmax(x, dim=1)
        kl_div = torch.nn.functional.kl_div(x, t, reduction="none").sum(dim=1)
        kl_div_np = dist_kl_div(a, b, axis=1)
        assert np.allclose(kl_div_np, kl_div.numpy()), f"{kl_div_np} - {kl_div.numpy()}"

    def test_numpy_js_div(self):
        a = np.random.rand(3, 8)
        b = np.random.rand(3, 8)
        x1 = torch.from_numpy(a)
        x2 = torch.from_numpy(b)
        t1 = torch.nn.functional.softmax(x1, dim=1)
        t2 = torch.nn.functional.softmax(x2, dim=1)
        t = (t1 + t2) / 2
        log_t = torch.log(t + 1.e-12)
        kl_div1 = torch.nn.functional.kl_div(log_t, t1, reduction="none").sum(dim=1)
        kl_div2 = torch.nn.functional.kl_div(log_t, t2, reduction="none").sum(dim=1)
        js_div = 0.5 * (kl_div1 + kl_div2) / np.log(2)
        js_div_np = dist_js_div(a, b, axis=1)
        assert np.allclose(js_div_np, js_div.numpy()), f"{js_div_np} - {js_div.numpy()}"

    def test_soft_cross_entropy(self):
        def softXEnt(input, target, weight=None):
            """Adapted from this pytorch discussion link.
               https://discuss.pytorch.org/t/
               soft-cross-entropy-loss-tf-has-it-does-pytorch-have-it/69501/2

               Per definition, we have (https://en.wikipedia.org/wiki/Cross_entropy):
                   CE(p,q) = -(p * log(q)).sum()
               With a provided weight per class, the computation becomes:
                   CE(p,q,w) = -(p * log(q)).sum() * (p * w).sum()
            """
            tmp_weight = 1.0 if weight is None else weight.unsqueeze(0)
            logprobs = torch.nn.functional.log_softmax(input, dim=1)
            avg_logprobs = (target * logprobs).sum(dim=1)
            avg_weight = 1.0 if weight is None else (target * tmp_weight).sum(dim=1)
            result = -(avg_logprobs * avg_weight)
            flag = weight is None
            result = result.mean() if flag else result.sum() / avg_weight.sum()
            return result

        N = 5
        C = 8
        preds = torch.rand(N, C)
        targets = torch.randint(low=0, high=C, size=(N,))
        target_indices = targets.unsqueeze(1)
        target_probas = torch.ones_like(target_indices).float()

        weight = torch.rand(C)

        sce = soft_cross_entropy(preds, target_indices, target_probas, weight=weight)
        ce = torch.nn.functional.cross_entropy(preds, targets, weight=weight)

        # compare with classical cross_entopy
        assert torch.allclose(sce, ce)

        target_indices2 = torch.cat(
            [target_indices, -1 * torch.ones_like(target_indices)], dim=1
        )
        target_probas2 = torch.cat(
            [target_probas, torch.zeros_like(target_probas)], dim=1
        )
        sce2 = soft_cross_entropy(preds, target_indices2, target_probas2, weight=weight)

        # compare with classical cross_entropy in case of padding
        assert torch.allclose(sce2, ce)

        indice_len = torch.randint(low=1, high=C + 1, size=(N,)).numpy()
        max_len = indice_len.max()
        soft_targets = np.ones((N, max_len), dtype=int) * -1
        soft_probas = np.zeros((N, max_len))
        soft_distribution = np.zeros((N, C))
        for i in range(N):
            k = indice_len[i]
            soft_targets[i, 0:k] = np.random.choice(np.arange(C), size=k, replace=False)
            soft_probas[i, 0:k] = torch.rand(size=(k,)).numpy()
            soft_probas[i] /= soft_probas[i].sum()
            for t in range(k):
                soft_distribution[i, soft_targets[i, t]] = soft_probas[i, t]
        assert np.allclose(soft_distribution.sum(axis=1), np.ones((N,)))
        assert np.allclose(soft_probas.sum(axis=1), np.ones((N,)))

        soft_targets = torch.from_numpy(soft_targets)
        soft_probas = torch.from_numpy(soft_probas)
        soft_distribution = torch.from_numpy(soft_distribution)

        sce3 = soft_cross_entropy(preds, soft_targets, soft_probas, weight=weight)
        sce4 = softXEnt(preds, soft_distribution, weight=weight)

        # generic case - compare with implementation based on a distribution expressed
        # in terms 1 single array of C elements
        assert torch.allclose(sce3, sce4)

    def test_cross_entropy_reshaping(self):
        N = 5
        C = 8
        preds1 = torch.rand(N, C)
        preds2 = torch.rand(N, C)
        targets = torch.randint(low=0, high=C, size=(N,))
        target_indices = targets.unsqueeze(1)
        target_probas = torch.ones_like(target_indices).float()
        discount = 0.99

        value, _ = cross_entropy_reshaping(
            preds2, preds1, targets, target_indices, target_probas, None, discount
        )
        value2 = -(
            discount
            * torch.nn.functional.cross_entropy(preds2, targets, reduction="none")
            - torch.nn.functional.cross_entropy(preds1, targets, reduction="none")
        )
        assert torch.allclose(value, value2)
