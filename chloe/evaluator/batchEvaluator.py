import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from chloe.evaluator.metrics import compute_metrics, write_json
from chloe.utils.dev_utils import initialize_seed

def get_litteral_diff(pred, name_map, name_flag, tres):
    pred = np.array(pred)
    pred_diff_mask = (pred > tres)
    num = pred_diff_mask.sum()
    if num == 0:
        return []
    pred = pred * pred_diff_mask
    top_ind = np.argsort(pred, axis=-1)[::-1]
    klist = top_ind[:num]
    if name_flag:
        result = [name_map[i] for i in klist]
    else:
        result = klist.tolist()
    return result

@torch.no_grad()
def evaluate(
        env,
        agent,
        total_length,
        num_eval_trajs=0,
        eval_patient_ids=None,
        seed=None,
        compute_metrics_flag=True,
        batch_size=1,
        deterministic = False,
        output_fp=None,
        action_fp=None,
        diff_fp=None,
):
        device = agent.device
        assert total_length > 0        

        if seed:
            initialize_seed(seed)

        overall_recall = 0
        overall_il = 0

        if (num_eval_trajs == 0) and (eval_patient_ids is None):
            num_eval_trajs = env.sample_size
            total_patients = env.sample_size
            eval_patient_ids = list(range(total_patients))
        elif eval_patient_ids is None:
            total_patients = env.sample_size
            num_eval_trajs = min(num_eval_trajs, total_patients)
            eval_patient_ids = random.sample(list(range(total_patients)), num_eval_trajs)
        elif num_eval_trajs == 0:
            assert all([i < env.sample_size for i in eval_patient_ids])
            num_eval_trajs = len(eval_patient_ids)

        all_metrics = {}
        num_batches = len(eval_patient_ids) // batch_size
        if len(eval_patient_ids) % batch_size > 0:
            num_batches += 1
        print(f"Number patients: {len(eval_patient_ids)} - Number Batches: {num_batches} - Batch Size: {batch_size} - NoRandom: {env.use_initial_symptom_flag}")
        ds_action_list = []
        ds_diff_list = []
        invalid_action_idx = -1
        for batch_idx in tqdm(range(num_batches), total=num_batches, desc="Evaluation: "):
                    
            patientIndices = eval_patient_ids[batch_idx * batch_size: min(len(eval_patient_ids), (batch_idx + 1) * batch_size)]

            obs, true_disease, _, _, _, first_actions = env.initialize_state(indices=patientIndices)

            done = np.zeros((obs.shape[0],)).astype(bool)

            agent.reset()
            agent.model.reset()

            all_diags = []
            curr_turn = 0
            valid_timesteps = []
            final_diagnosis = np.zeros((obs.shape[0], env.diag_size))
            batch_action_list = np.ones((obs.shape[0], total_length + 1)) * invalid_action_idx
            batch_action_list[:obs.shape[0], 0] = first_actions
            batch_diff_list = np.ones((obs.shape[0], env.diag_size, total_length + 1)) * invalid_action_idx
            while curr_turn < total_length:
                valid_timesteps.append(np.zeros((obs.shape[0],)).astype(bool))
                inputs = torch.from_numpy(obs).float().to(device)

                # import pdb;pdb.set_trace()
                out = agent.model.predict(inputs)
                
                action = (
                    out["action"] if not hasattr(agent, "distribution") or deterministic
                    else agent.distribution.sample(out.get("p", out["q"])).detach().view(obs.shape[0],).cpu().numpy()
                )
                q = out["q"]
                tmpqStr = "\n".join([f"{i} - {q[:, i]}" for i in range(q.size(1)) ])
                # print(f"Q (turn {curr_turn}): \n {tmpqStr}")
                pi = out.get("pi")
                if pi is None:
                    pi = q[:, env.symptom_size:]

                patho_pred = (
                    None
                    if pi is None
                    else F.softmax(pi, dim=-1).detach().view(obs.shape[0], -1).cpu().numpy()
                )
                all_diags.append(patho_pred if patho_pred is not None else (np.ones((obs.shape[0], env.diag_size))/env.diag_size))                
                
                # print(f"turn {curr_turn} - Action = {action} - diff = {all_diags[-1].tolist()}")
                
                should_stop = action >= env.symptom_size
                newly_done = np.logical_and(~done, should_stop)
                done[should_stop] = True
                final_diagnosis[newly_done] = all_diags[-1][newly_done]                
                batch_diff_list[:obs.shape[0], :, curr_turn] = (all_diags[-1] * (1 - done).reshape(-1, 1)) + (invalid_action_idx * done.reshape(-1, 1))
                batch_diff_list[newly_done, :, curr_turn] = all_diags[-1][newly_done]
                if all(done):
                    if len(valid_timesteps) > 1:
                        valid_timesteps.pop()
                    break

                batch_action_list[:obs.shape[0], curr_turn + 1] = (action * (1 - done)) + (invalid_action_idx * done)
                obs, _, _ = env.step(obs, action, done)
                valid_timesteps[-1][~done] = True 
                curr_turn += 1

            if any(~done):
                inputs = torch.from_numpy(obs).float().to(device)
                out = agent.model.predict(inputs)
                pi = out.get("pi")
                if pi is None:
                    pi = out["q"][:, env.symptom_size:]

                # import pdb;pdb.set_trace()
                patho_pred = (
                    None
                    if pi is None
                    else F.softmax(pi, dim=-1).detach().view(obs.shape[0], -1).cpu().numpy()
                )
                all_diags.append(patho_pred if patho_pred is not None else (np.ones((obs.shape[0], env.diag_size))/env.diag_size))
                final_diagnosis[~done] = all_diags[-1][~done]
                batch_diff_list[:obs.shape[0], :, total_length] = (all_diags[-1] * (1 - done).reshape(-1, 1)) + (invalid_action_idx * done.reshape(-1, 1))
                # print(f"turn {curr_turn} - Action = {action} - diff = {all_diags[-1].tolist()}")
                
            ds_action_list.append(batch_action_list)
            ds_diff_list.append(batch_diff_list)
            valid_timesteps = np.array(valid_timesteps).swapaxes(0, 1)
            all_diags = np.array(all_diags).swapaxes(0, 1)
            
            ave_step = np.sum(valid_timesteps, axis=-1) + 1
            ave_pos = np.sum(env.inquired_symptoms * env.all_state, axis=-1) / np.maximum(1, ave_step)
            
            overall_il += np.sum(ave_step)
            overall_recall += np.sum(ave_pos)
                
            if compute_metrics_flag:
                batch_metrics = compute_metrics(
                    env.target_differential, true_disease,
                    final_diagnosis, all_diags, valid_timesteps,
                    env.all_state, env.inquired_symptoms,
                    env.symptom_mask, env.atcd_mask, env.severity_mask, tres=0.01
                )
                tmp_n = batch_idx * batch_size
                tmp_m = obs.shape[0]
                tmp_t = tmp_n + obs.shape[0]
                all_metrics = {a: (tmp_n / (tmp_t)) * all_metrics.get(a, 0) + (tmp_m / (tmp_t)) * batch_metrics[a] for a in batch_metrics.keys()}

        ds_action_list = np.concatenate(ds_action_list, axis=0).astype('int')
        df = pd.DataFrame(ds_action_list, columns = [str(i) for i in range(ds_action_list.shape[1])])
        df = df.applymap(lambda x: 'None' if x == invalid_action_idx else env.symptom_data[env.symptom_index_2_key[x]]['name'])
        name_map = {i: env.pathology_data[env.pathology_index_2_key[i]]['condition_name'] for i in range(env.diag_size)}
        ds_diff_list = np.concatenate(ds_diff_list, axis=0).transpose(0,2,1).tolist()
        df2 = pd.DataFrame(ds_diff_list, columns = [str(i) for i in range(ds_action_list.shape[1])])
        df2 = df2.applymap(lambda x: get_litteral_diff(x, name_map, True, tres=0.01))
        results_dict = {
            "average_evidence_recall": overall_recall / max(1, len(eval_patient_ids)),
            "average_step": overall_il / max(1, len(eval_patient_ids)),
        }
        if compute_metrics_flag:
            all_metrics = {a: all_metrics[a].tolist() if hasattr(all_metrics[a], "tolist") else all_metrics[a] for a in all_metrics.keys()}
            results_dict.update(all_metrics)

        if output_fp is not None:
            write_json(results_dict, output_fp)

        if action_fp is not None:
            df.to_csv(action_fp,  sep=',', index=False)

        if diff_fp is not None:
            df2.to_csv(diff_fp,  sep=',', index=False)

        return results_dict
