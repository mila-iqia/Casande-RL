import numpy as np
from tqdm import tqdm
import pickle, ast, json
import copy
from operator import itemgetter
import random
from scipy.stats import entropy
import pandas as pd

eps = 0.001
gender2idx = {'M':0,'F':1}
class Age2Idx():
    def age2idx(self, age):
        if age < 1:
            return 0
        elif age < 4:
            return 1
        elif age < 14:
            return 2
        elif age < 29:
            return 3
        elif age < 44:
            return 4
        elif age < 59:
            return 5
        elif age < 74:
            return 6
        else:
            return 7
    
    def __len__(self):
        return 8

    def __getitem__(self, age):
        return self.age2idx(age)

age2idx = Age2Idx()

def _convert_to_aarlc_format(row_data, s2idx):
    init_evidence_idx = s2idx[row_data["INITIAL_EVIDENCE"]]
    all_evi_idxs = [s2idx[evi] for evi in ast.literal_eval(row_data["EVIDENCES"])]
    patient_data = {
        "d": row_data["PATHOLOGY"],
        "self_report": [init_evidence_idx],
        "acquired": list(set(all_evi_idxs) - {init_evidence_idx}),
        "test": [],
        "age": age2idx[row_data["AGE"]],
        "gender": gender2idx[row_data["SEX"]],
        "all_sym": all_evi_idxs,
    }
    return patient_data

def get_patients(args, s2idx, train=True):
    if train:
        patients_df = pd.read_csv(args.train_data_path, sep=",", engine="c")
    else:
        patients_df = pd.read_csv(args.val_data_path, sep=",", engine="c")
    patients = patients_df.apply(lambda row_data: _convert_to_aarlc_format(row_data, s2idx), axis="columns").to_list()
    patients = {idx: patient_data for idx, patient_data in enumerate(patients)}
    return patients

class environment(object):
    def __init__(self, args, train=None):
        self.args = args
        self.dataset = self.args.dataset
        self.context_size = len(age2idx) + len(gender2idx)
        self.load_meta_data(args.evi_meta_path, args.patho_meta_path)
        self.patients = get_patients(args, self.s2idx, train=args.train if train is None else train)
        self.sample_size = len(self.patients)
        self.idx = 0
        self.indexes = np.arange(self.sample_size)
        self.diag_size = len(self.d2idx)

        self.cost = np.concatenate((1 * np.ones(self.ss_size), 1 * np.ones(self.ts_size)))
        self.earn = np.concatenate((1 * np.ones(self.ss_size), 1 * np.ones(self.ts_size)))

    def load_meta_data(self, evi_meta_path, patho_meta_path):
        with open(patho_meta_path, "r") as file:
            patho_meta_data = json.load(file)
        with open(evi_meta_path, "r") as file:
            evi_meta_data = json.load(file)

        self.d2idx = {patho: idx for idx, patho in enumerate(patho_meta_data.keys())}
        self.s2idx = {evi: idx for idx, evi in enumerate(evi_meta_data.keys())}
        self.ts_size = 0
        self.ss_size = len(self.s2idx)
        self.symptom_size = self.ts_size + self.ss_size
        print(self.symptom_size)

    def reset(self):

        self.idx = 0
        np.random.shuffle(self.indexes)
        
    def initialize_state(self, batch_size):

        self.batch_size = batch_size
        self.batch_index = self.indexes[self.idx : self.idx+batch_size]
        self.idx += batch_size
        self.disease = []
        self.pos_sym = []
        self.acquired_sym = []
        
        i = 0
        init_state = np.zeros((batch_size, self.symptom_size+self.context_size))
        self.all_state = np.zeros((batch_size, self.symptom_size))
        for item in self.batch_index:
            self.disease.append(self.d2idx[self.patients[item]['d']])
            self.all_state[i, self.patients[item]['all_sym']] = 1
            init_state[i, self.patients[item]['self_report']] = 1
            init_state[i, self.patients[item]['age']+self.symptom_size] = 1
            init_state[i, self.patients[item]['gender']+self.symptom_size+len(age2idx)] = 1
            i += 1
        self.disease = np.array(self.disease)

        return init_state, self.disease

    def step(self, s, a_p, done, right_diagnosis, agent, ent_init, threshold, ent):

        s_ = copy.deepcopy(s)
        ent_ = copy.deepcopy(ent)
        s_[~done, a_p[~done]] =  self.all_state[~done, a_p[~done]] * 2 - 1

        a_d_, p_d_ = agent.choose_diagnosis(s_)

        ent_[~done] = entropy(p_d_[~done], axis = 1)
        ent_ratio = (ent-ent_) / ent_init

        diag = (ent_ < threshold[a_d_]) & (~done)
        right_diag = (a_d_ == np.array(self.disease)) & diag 

        reward_s = self.args.mu * self.reward_func(s[:,:self.symptom_size], s_[:,:self.symptom_size], diag, a_p) 
        reward_s[ent_ratio > 0] += (self.args.nu * ent_ratio[ent_ratio > 0])
        reward_s[diag] -= (self.args.mu * 1)
        reward_s[right_diag] += (self.args.mu * 2)
        reward_s[done] = 0
        
        done += diag
        right_diagnosis += right_diag
        
        return s_, reward_s, done, right_diagnosis, diag, ent_, a_d_
    
    def reward_func(self, s, s_, diag, a_p):
        
        reward = -self.cost[a_p]
        reward += np.sum(np.abs(s-s_)*self.cost, axis = 1) * 0.7
        reward += np.sum(((s_-s)>0)*self.earn, axis = 1)

        return reward

if __name__ == '__main__':
    np.random.seed(1000)
    random.seed(1000)
    env = environment(100)