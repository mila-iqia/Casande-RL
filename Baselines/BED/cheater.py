import logging
import numpy as np
from random import sample
from QMR import QMR
logger = logging.getLogger(__name__)


class Cheater:
    """
    The cheater solver know all the true findings (full informaiton).
    Note that `learn` method know that known findings is full observation,
    while the `inference` method cann't differential partial observation from
    full information and treat all situations equally. For this reason, the
    `learn` method has better predictions.
    """

    def __init__(self, args):
        self.qmr = QMR({'args': args})
        self.args = args

    def _inference(self):
        for action in self.qmr.findings:
            self.qmr.step(action)
        correctness = self.qmr.inference()
        return correctness

    def _collect(self):
        n_diseases = self.args.n_diseases
        findings2disease = {}
        for _ in range(10**7):
            self.qmr.reset()
            if self.qmr.findings not in findings2disease:
                findings2disease[self.qmr.findings] = np.zeros(
                    n_diseases, dtype=np.int64)
            findings2disease[self.qmr.findings][self.qmr.disease] += 1
        return findings2disease

    def _learn(self):
        n_diseases = self.args.n_diseases

        # Learn
        findings2disease = self._collect()

        # Predict
        n_correct = [0, 0, 0]
        test_size = self.args.test_size
        for _ in range(test_size):
            self.qmr.reset()
            if self.qmr.findings in findings2disease:
                top5 = findings2disease[self.qmr.findings].argsort()[-5:][::-1]
            else:
                top5 = sample(range(n_diseases), 5)
            target = self.qmr.disease
            correctness = (target == top5[0],
                           target in top5[:3], target in top5)
            n_correct = [i + j for i, j in zip(n_correct, correctness)]
        accuracy = [i / test_size for i in n_correct]
        logger.info(f'Learn:#experiments: {test_size}; accuracy: {accuracy}.')

    def _inference(self):
        n_correct = [0, 0, 0]
        test_size = self.args.test_size
        all_findings = set(range(self.qmr.n_all_findings))
        for _ in range(test_size):
            self.qmr.reset()
            neg_findings = all_findings - set(self.qmr.findings)
            correctness = self.qmr.inference(self.qmr.findings, neg_findings)
            n_correct = [i + j for i, j in zip(n_correct, correctness)]
        accuracy = [i / test_size for i in n_correct]
        logger.info(
            f'Inference:#experiments: {test_size}; accuracy: {accuracy}.')

    def compare_distribution(self):
        findings2disease = self._collect()
        logger.info(self.qmr.disease2finding)
        logger.info(findings2disease)

        all_findings = set(range(self.qmr.n_all_findings))
        for _ in range(10):
            self.qmr.reset()
            if self.qmr.findings in findings2disease:
                logger.info(
                    f'Disease: {self.qmr.disease}, Findings: {self.qmr.findings}')
                learn_dist = findings2disease[self.qmr.findings]
                learn_dist = learn_dist / learn_dist.sum()

                neg_findings = all_findings - set(self.qmr.findings)
                inference_dist, _ = self.qmr.compute_disease_probs(
                    self.qmr.findings, neg_findings, normalize=True)
                logger.info(
                    f'Learn distribution: {learn_dist}, Inference distribution: {inference_dist.tolist()}\n')

    def run(self):
        if self.qmr.test_data is not None:
            raise NotImplementedError(
                'Cheater class is not implemented for real dataset')
        if self.args.cheater_method == 'inference':
            self._inference()
        elif self.args.cheater_method == 'learn':
            self._learn()
        elif self.args.cheater_method == 'compare':
            self.compare_distribution()
        else:
            raise NotImplementedError(
                f'Cheater method {self.cheater_method} is not implemented.')
