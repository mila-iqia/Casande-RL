
import random
from collections import defaultdict, Counter
import pickle


def one_disease_sample_1(disease2finding):
    disease = random.randrange(2)
    findings = set()
    while True:
        for finding, prob in disease2finding[disease].items():
            if random.random() <= prob:
                findings.add(finding)
        if len(findings) > 0:
            break
    findings = list(findings)
    return disease, findings


def one_disease_sample_2(disease2finding):
    findings = set()
    while True:
        disease = random.randrange(2)

        for finding, prob in disease2finding[disease].items():
            if random.random() <= prob:
                findings.add(finding)
        if len(findings) > 0:
            break
    findings = list(findings)

    return disease, findings


def run_helper(sample_func, disease2finding):
    d = defaultdict(int)
    for _ in range(10**6):
        disease, findings = sample_func(disease2finding)
        d[tuple([disease] + findings)] += 1
    print(d)
    p0_0 = (d[(0, 0)] + d[(0, 0, 1)]) / (d[(0, 0)] +
                                         d[(1, 0)] + d[(0, 0, 1)] + d[(1, 0, 1)])
    p0_1 = (d[(0, 1)] + d[0, 0, 1]) / (d[(0, 1)] +
                                       d[(1, 1)] + d[(0, 0, 1)] + d[(1, 0, 1)])
    p0_01 = d[(0, 0, 1)] / (d[(0, 0, 1)] + d[(1, 0, 1)])
    print(f'{p0_0=:.4f}, {p0_1=:.4f}, {p0_01=:.4f}')


def run():
    disease2finding = {0: {0: 0.8, 1: 0.2}, 1: {0: 0.3, 1: 0.6}}

    # Generate using
    run_helper(one_disease_sample_1, disease2finding)
    run_helper(one_disease_sample_2, disease2finding)


def get_disease2finding(data):
    # with open('../dataset/dxy_dataset/all_norm_symptoms.txt', 'r') as f:
    #     norm_symps = set(f.read().split('\n')[:-1])

    # norm_symps = pickle.load(open('../dataset/acl2018-mds/slot_set.p', 'rb'))

    disease2finding = defaultdict(list)
    for case in data:
        disease = case['disease_tag']
        for symp, b in case['goal']['implicit_inform_slots'].items():
            if b:
                disease2finding[disease].append(symp)
        for symp, b in case['goal']['explicit_inform_slots'].items():
            if b:
                disease2finding[disease].append(symp)

    disease2finding = {d: Counter(fs) for d, fs in disease2finding.items()}
    print(Counter([case['disease_tag'] for case in data]))
    return disease2finding


def build_graph():
    # path = '../dataset/dxy_dataset/dxy_dialog_data_dialog_v2.pickle'
    path = '../dataset/acl2018-mds/acl2018-mds.p'
    data = pickle.load(open(path, 'rb'))
    print(len(data['train']), len(data['test']))
    train_disease2finding = get_disease2finding(data['train'])
    test_disease2finding = get_disease2finding(data['test'])

    print(train_disease2finding)
    print(test_disease2finding)

    for d, fs in test_disease2finding.items():
        for f in fs.keys():
            if f not in train_disease2finding[d]:
                print(f'{d}: {f}')
    print('fininished')


if __name__ == '__main__':
    build_graph()
