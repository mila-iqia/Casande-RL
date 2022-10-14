import os, pickle, argparse, json
from trajectory_metric_torch import evaluate_trajectory
from aarlc_metrics import compute_severity_stats
import torch
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate trajectory quality metrics given trajectories")

    # data io args
    parser.add_argument("--disease_logits_path", required=True, type=str, help="path to the saved disease logits file")
    parser.add_argument("--vocab_path", required=True, type=str, help="path to the vocab file indexing diseases")
    parser.add_argument("--goal_set_path", type=str, help="DEPRECIATED, use differentials_path instead. path to the goal set file that contains differentials")
    parser.add_argument("--differentials_path", type=str, help="path to the file that contains differentials. can be goal_set or test data")
    parser.add_argument("--patho_meta_path", type=str, help="path to the pathology meta data file. required for severe pathology metrics")
    parser.add_argument("--test_idxs_path", type=str, help="path to the saved test indices. if not given, will assume all samples are used")
    parser.add_argument("--save_dir", type=str, help="directory to save results. if not given, results will be saved to disease logits file's folder")

    # other args
    parser.add_argument("--metrics", choices=["kl_auc", "pareto_kl_auc", "severity"], action="append", help="which metrics to evaluate")
    parser.add_argument("--use_16_bit_precision", action="store_true", help="use 16 bit precision when possible to reduce memory footprint")
    parser.add_argument("--severity_threshold", type=float, default=3, help="threshold for considering a disease severe (non-inclusive)")
    parser.add_argument("--differential_threshold", type=float, default=0.01, help="threshold for including a disease in the differential")

    return parser.parse_args()

def _get_severity_mask(patho_meta_path, severity_threshold, disvocab):
    """Load patho meta data and generate severity mask

    Parameter
    ----------
    patho_meta_path: str
        path to the pathology meta data
    severity_threshold: float
        threshold for considering a disease severe
    disvocab: dict
        a dictionary mapping pathologies to their indices

    Return
    ----------
    severity_mask: numpy.array
        a binary numpy array indicating whether each disease is severe or not. Shape: (# diseases)

    """
    with open(patho_meta_path, "r") as file:
        patho_meta_data = json.load(file)
    
    severity_mask = np.zeros(len(disvocab), dtype=np.int8)
    for disease, idx in disvocab.items():
        if patho_meta_data[disease]["severity"] < severity_threshold:
            severity_mask[idx] = 1

    return severity_mask

if __name__ == "__main__":
    args = parse_args()
    args.metrics = set(args.metrics)
    print(args)

    if "severity" in args.metrics and args.patho_meta_path is None:
        raise Exception("pathology meta file is required for severity metrics")

    assert not (args.differentials_path is None and args.goal_set_path is None), "differentials_path is not provided"
    if args.differentials_path is None and args.goal_set_path is not None:
        print("goal_set_path is depreciated, use differentials_path instead")
        args.differentials_path = args.goal_set_path

    # get dtype
    dtype = torch.half if args.use_16_bit_precision else torch.float

    # load disease logits
    print("Loading disease logits")
    with open(args.disease_logits_path, "rb") as file:
        disease_logits = pickle.load(file)

    # load disease indices
    with open(args.vocab_path, "r", encoding="utf-8") as file:
        tokens = file.read().split('\n\n')
        dislist = tokens[1].splitlines()
    disvocab = {}
    for index, token in enumerate(dislist):
        disvocab[token] = index

    # load test idxs
    if args.test_idxs_path:
        with open(args.test_idxs_path, "rb") as file:
            test_idxs = pickle.load(file)
    else:
        test_idxs = list(range(len(disease_logits)))

    # load differentials
    print("Loading differentials")
    differentials, differentials_probs = [], []
    with open(args.differentials_path, "rb") as file:
        differentials_data = pickle.load(file)
    differentials_data = differentials_data["test"]
    if not args.test_idxs_path:
        assert len(disease_logits) == len(differentials_data), \
            f"disease digits must have the same number of samples as goal set when test indices are not given, but instead got {len(disease_logits)} and {len(differentials_data)} instead"
    for idx, item in tqdm(enumerate(differentials_data), total=len(differentials_data)):
        if idx not in test_idxs:
            continue
        current_differentials, current_differentials_probs = [], []
        for patho, proba in item["differentials"]:
            current_differentials.append(disvocab[patho])
            current_differentials_probs.append(proba)
        differentials.append(torch.tensor(current_differentials, dtype=torch.long))
        differentials_probs.append(torch.tensor(current_differentials_probs, dtype=dtype))

    # load severe pathos
    if "severity" in args.metrics:
        severity_mask = _get_severity_mask(args.patho_meta_path, args.severity_threshold, disvocab)

    # prepare data
    print("Preparing data")
    when_ends = torch.tensor([len(trajectory) - 1 for trajectory in disease_logits], dtype=torch.long)
    disease_logits = [torch.vstack(trajectory).cpu() for trajectory in disease_logits]
    disease_logits = torch.nn.utils.rnn.pad_sequence(disease_logits, batch_first=True).type(dtype)
    differentials = torch.nn.utils.rnn.pad_sequence(differentials, batch_first=True, padding_value=-1).type(torch.long)
    differentials_probs = torch.nn.utils.rnn.pad_sequence(differentials_probs, batch_first=True, padding_value=-1).type(dtype)

    # results saving prefix
    prefix = os.path.splitext(os.path.basename(args.disease_logits_path))[0]

    # evaluate
    if "kl_auc" in args.metrics:
        print("\nEvaluating KL AUC")
        results_dict = evaluate_trajectory(disease_logits, when_ends=when_ends, differentials=differentials, differentials_probs=differentials_probs, which_metrics={"kl_auc"}, summarize="mean")

        # print summary
        print(f"Mean KL AUC: {results_dict['kl_auc'].type(torch.float).mean().item()}") # cast to float before calculating mean to avoid overflow

        # save
        print("Saving results")
        save_dir = args.save_dir if args.save_dir else os.path.dirname(args.disease_logits_path)
        with open(os.path.join(save_dir, f"{prefix}_kl_auc_results_dict.pkl"), "wb") as file:
            pickle.dump(results_dict, file)
        print(f"Results saved to {save_dir}/{prefix}_kl_auc_results_dict.pkl")
        del results_dict
    if "pareto_kl_auc" in args.metrics:
        print("\nEvaluating Pareto KL AUC")
        results_dict = evaluate_trajectory(disease_logits, when_ends=when_ends, differentials=differentials, differentials_probs=differentials_probs, which_metrics={"pareto_kl_auc"})

        # print summary
        print(f"Mean Pareto KL AUC: {results_dict['pareto_kl_auc'].astype(np.float32).mean().item()}") # cast to float before calculating mean to avoid overflow

        # save
        print("Saving results")
        save_dir = args.save_dir if args.save_dir else os.path.dirname(args.disease_logits_path)
        with open(os.path.join(save_dir, f"{prefix}_pareto_kl_auc_results_dict.pkl"), "wb") as file:
            pickle.dump(results_dict, file)
        print(f"Results saved to {save_dir}/{prefix}_pareto_kl_auc_results_dict.pkl")
        del results_dict
    if "severity" in args.metrics:
        print("\nEvaluating severity metrics")
        differentials_full_matrix = torch.zeros(disease_logits.shape[0], disease_logits.shape[-1] + 1, dtype=dtype)
        differentials_full_matrix[torch.arange(differentials_full_matrix.shape[0]).unsqueeze(-1), differentials] = differentials_probs
        differentials_full_matrix = differentials_full_matrix[:, :-1]
        differentials_full_matrix = differentials_full_matrix.unsqueeze(dim=1)

        pred_no_gt, gt_no_pred, gt_pred_f1 = compute_severity_stats(torch.softmax(disease_logits.type(torch.float), dim=-1).type(dtype).numpy(), differentials_full_matrix.numpy(), severity_mask, args.differential_threshold)
        mean_final_f1 = np.mean(gt_pred_f1[np.arange(gt_pred_f1.shape[0]), when_ends.numpy()])
        print(f"Mean final severity F1: {mean_final_f1}")

        print("Saving results")
        save_dir = args.save_dir if args.save_dir else os.path.dirname(args.disease_logits_path)
        with open(os.path.join(save_dir, f"{prefix}_pred_no_gt.pkl"), "wb") as file:
            pickle.dump(pred_no_gt, file)
        del pred_no_gt
        with open(os.path.join(save_dir, f"{prefix}_gt_no_pred.pkl"), "wb") as file:
            pickle.dump(gt_no_pred, file)
        del gt_no_pred
        with open(os.path.join(save_dir, f"{prefix}_gt_pred_f1.pkl"), "wb") as file:
            pickle.dump(gt_pred_f1, file)
        del gt_pred_f1
        with open(os.path.join(save_dir, f"{prefix}_when_ends.pkl"), "wb") as file:
            pickle.dump(when_ends, file)
        print(f"Results saved to {save_dir}/{prefix}_pred_no_gt.pkl,{prefix}_gt_no_pred.pkl,{prefix}_gt_pred_f1.pkl,{prefix}_when_ends.pkl")