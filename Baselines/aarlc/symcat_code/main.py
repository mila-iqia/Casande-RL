from env import *
from agent import *
import copy
import torch
import pickle
import time
import numpy as np
from tqdm import tqdm
from scipy.stats import entropy
import argparse

import wandb

def main():
    print("Initializing Environment and generating Patients....")
    env = environment(args, train=True)
    agent = Policy_Gradient_pair_model(state_size = env.symptom_size, disease_size = env.diag_size, context_size= env.context_size, LR = args.lr, Gamma = args.gamma)
    threshold_list = []
    best_a = 0
    if args.threshold_random_initial:
        threshold = np.random.rand(env.diag_size)
    else:
        threshold = args.threshold * np.ones(env.diag_size)


    for epoch in range(args.EPOCHS):
        env.reset()
        agent.train()
        num_batches = env.sample_size // args.batch_size
        for batch_idx in tqdm(range(num_batches), total=num_batches, desc=f"epoch {epoch}: "):
            states = []
            action_m = []
            rewards_s = []
            action_s = []
            true_d = []

            s, true_disease = env.initialize_state(args.batch_size)
            s_init = copy.deepcopy(s)
            s_final = copy.deepcopy(s)

            a_d, p_d = agent.choose_diagnosis(s)
            init_ent = entropy(p_d, axis = 1)
            
            done = (init_ent < threshold[a_d])
            right_diag = (a_d == env.disease) & done

            diag_ent = np.zeros(args.batch_size)
            finl_diag = np.zeros(args.batch_size).astype(int) - 1
            diag_ent[right_diag] = init_ent[right_diag]
            ent = init_ent

            for i in range(args.MAXSTEP):
                a_s = agent.choose_action_s(s)
                
                s_, r_s, done, right_diag, final_idx, ent_, a_d_ = env.step(s, a_s, done, right_diag, agent, init_ent, threshold, ent)
                s_final[final_idx] = s_[final_idx]
                diag_ent[right_diag] = ent_[right_diag]
                finl_diag[right_diag] = a_d_[right_diag]
                # print(max(finl_diag[right_diag]))
                # print(max(a_d_[right_diag]))
                # print(finl_diag[right_diag])
                # print(a_d_[right_diag])
                # input()
                if i == (args.MAXSTEP - 1):
                    r_s[~done] += 1

                states.append(s)
                rewards_s.append(r_s)
                action_s.append(a_s)
                true_d.append(true_disease)
                
                s = s_
                ent = ent_
                
                if all(done):
                    break
            
            diag = np.sum(done)
            s_final[~done] = s_[~done]

            all_step, ave_reward_s = agent.create_batch(states, rewards_s, action_s, true_d)
            a_d, p_d = agent.choose_diagnosis(s)
            
            t_d = (a_d == env.disease) & (~done)
            diag_ent[t_d] = entropy(p_d[t_d], axis = 1)
            finl_diag[t_d] = a_d[t_d]
            # print(max(finl_diag))
            for idx, item in enumerate(finl_diag):
                if item >= 0 and abs(threshold[item] - diag_ent[idx]) > 0.01:
                    threshold[item] = (args.lamb * threshold[item] + (1-args.lamb) * diag_ent[idx])   #update the threshold

            agent.update_param_rl()
            agent.update_param_c()
            
            accuracy = (sum(right_diag)+sum(t_d))/(args.batch_size)
            best_a = np.max([best_a, accuracy])

            ave_pos = (np.sum(s_final == 1) - np.sum(s_init == 1)) / all_step
            ave_step = all_step / args.batch_size

            threshold_list.append(threshold)

            # wandb logging
            results_dict = {"accuracy/train": accuracy, "average_pos/train": ave_pos, "average_step/train": ave_step, "average_symptom_reward/train": ave_reward_s, "epoch": epoch}
            wandb.log(results_dict)

            # print("==Epoch:", epoch+1, '\tAve. Accu:', accuracy, '\tBest Accu:', best_a, '\tAve. Pos:', ave_pos)
            # print('Threshold:', threshold[:5], '\tAve. Step:', ave_step, '\tAve. Reward Sym.:', ave_reward_s, '\n')

        if args.eval_on_train_epoch_end:
            test(agent=agent, threshold=threshold)

        agent.save_model(args)
        info = str(args.dataset) + '_' + str(args.threshold) + '_' + str(args.mu) + '_' + str(args.nu) + '_' + str(args.trail)
        with open(os.path.join(args.save_dir, f"threshold_changing_curve_{info}.pkl"), 'wb') as f:
            pickle.dump(threshold_list, f)

@torch.no_grad()
def test(agent=None, threshold=None):
    print("Initializing Environment and generating Patients....")
    env = environment(args, train=False)
    if agent is None:
        agent = Policy_Gradient_pair_model(state_size = env.symptom_size, disease_size = env.diag_size, context_size= env.context_size, LR = args.lr, Gamma = args.gamma)
        agent.load_model(args)
    agent.eval()

    if threshold is None:
        info = str(args.dataset) + '_' + str(args.threshold) + '_' + str(args.mu) + '_' + str(args.nu) + '_' + str(args.trail)
        with open(os.path.join(args.checkpoint_dir, f"threshold_changing_curve_{info}.pkl"), 'rb') as f:
            bf = pickle.load(f)
            threshold = bf[-1]

    steps_on_ave = 0
    pos_on_ave = 0
    accu_on_ave = 0

    num_batches = env.sample_size // args.batch_size
    for batch_idx in tqdm(range(num_batches), total=num_batches):
        states = []
        action_m = []
        rewards_s = []
        action_s = []
        true_d = []


        s, true_disease, = env.initialize_state(args.batch_size)
        s_init = copy.deepcopy(s)
        s_final = copy.deepcopy(s)

        a_d, p_d = agent.choose_diagnosis(s)
        init_ent = entropy(p_d, axis = 1)
        
        done = (init_ent < threshold[a_d])
        right_diag = (a_d == env.disease) & done

        diag_ent = np.zeros(args.batch_size)
        diag_ent[right_diag] = init_ent[right_diag]
        ent = init_ent

        for i in range(args.MAXSTEP):

            a_s = agent.choose_action_s(s)
            s_, r_s, done, right_diag, final_idx, ent_, a_d_ = env.step(s, a_s, done, right_diag, agent, init_ent, threshold, ent)

            s_final[final_idx] = s_[final_idx]
            # diag_ent[right_diag] = ent_[right_diag]

            if i == args.MAXSTEP - 1:
                r_s[done==False] -= 1

            states.append(s)
            rewards_s.append(r_s)
            action_s.append(a_s)
            true_d.append(true_disease)
            
            s = s_
            ent = ent_
            
            if all(done):
                break

        diag = np.sum(done)
        s_final[~done] = s_[~done]
            
        all_step, ave_reward_s = agent.create_batch(states, rewards_s, action_s, true_d)
        a_d, p_d = agent.choose_diagnosis(s)
        finl_ent = entropy(p_d, axis = 1)
        t_d = (a_d == env.disease) & (~done)
        # diag_ent[t_d] = finl_ent[t_d]
        
        ave_step = all_step / args.batch_size
        ave_pos = (np.sum(s_final == 1) - np.sum(s_init == 1)) / all_step
        accurate = (sum(right_diag) + sum(t_d)) / args.batch_size

        steps_on_ave = batch_idx / (batch_idx + 1) * steps_on_ave + 1 / (batch_idx + 1) * ave_step
        pos_on_ave = batch_idx / (batch_idx + 1) * pos_on_ave + 1 / (batch_idx + 1) * ave_pos
        accu_on_ave = batch_idx / (batch_idx + 1) * accu_on_ave + 1 / (batch_idx + 1) * accurate
        # print(steps_on_ave, pos_on_ave, accu_on_ave)
        # wandb logging
        results_dict = {"accuracy/validation": accu_on_ave, "average_pos/validation": pos_on_ave, "average_step/validation": steps_on_ave, "epoch": 0}
        wandb.log(results_dict)

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Process Settings')
    parser.add_argument('--dataset', type=str, default = '400',
                        help='select a symcat disease subset (200, 300, 400 or common)')
    parser.add_argument('--seed', type=int, default = 42,
                        help='set a random seed')
    parser.add_argument('--threshold', type=float, default = 1,
                        help='set a initial threshold')
    parser.add_argument('--threshold_random_initial', action="store_true",
                        help='randomly initialize threshold')
    parser.add_argument('--batch_size', type=int, default = 200,
                        help='games for each time onpolicy sample collection')
    parser.add_argument('--eval_on_train_epoch_end', action="store_true",
                        help='evaluate at the end of each epoch')
    parser.add_argument('--EPOCHS', type=int, default = 100,
                        help='training epochs')
    parser.add_argument('--MAXSTEP', type=int, default = 15,
                        help='max inquiring turns of each MAD round')
    parser.add_argument('--nu', type=float, default = 2.5,
                        help='nu')
    parser.add_argument('--mu', type=float, default = 1,
                        help='mu')
    parser.add_argument('--lr', type=float, default = 1e-4,
                        help='learning rate')
    parser.add_argument('--gamma', type=float, default = 0.99,
                        help='reward discount rate')
    parser.add_argument('--train', action="store_true",
                        help='whether test on the exsit result model or train a new model')
    parser.add_argument('--trail', type=int, default = 1)
    parser.add_argument('--lamb', type=float, default = 0.99,
                        help='polyak factor for threshold adjusting')
    parser.add_argument('--save_dir', type=str, default='.', help='directory to save the results')
    parser.add_argument('--checkpoint_dir', type=str, help='directory containing the checkpoints to restore')
    parser.add_argument('--train_data_path', type=str, required=True, help='path to the training data file')
    parser.add_argument('--val_data_path', type=str, required=True, help='path to the validation data file')
    parser.add_argument('--evi_meta_path', type=str, required=True, help='path to the evidences (symptoms) meta data')
    parser.add_argument('--patho_meta_path', type=str, required=True, help='path to the pathologies (diseases) meta data')
    args = parser.parse_args()
    
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # setup wandb
    time_stamp = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    wandb.init(name=time_stamp, group="aarlc", project="medical_evidence_collection")
    wandb.config.update(args)

    if args.train:
        # add save_dir
        args.save_dir = os.path.join(args.save_dir, time_stamp)
        os.makedirs(args.save_dir)
        main()
    else:
        test()
