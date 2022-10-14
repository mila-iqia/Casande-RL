# -*- coding: utf-8 -*
from dataclasses import replace
from numpy.random import rand
from torch._C import DeviceObjType
from torch.utils.data.dataset import random_split
# import sys
# sys.path.append("./") 
from pytorch_transformers import BertConfig,DiaModel,AdamW,WarmupLinearSchedule
import torch
import os, time
# import json
import pickle
import json
import random
import numpy as np
import argparse
from datetime import datetime
from torch.nn import DataParallel
import logging
from os.path import join, exists
from dataset import diaDataset
from tokenizer import diaTokenizer
# from dataload import collate_fn_eval, collate_fn_train
from loss import lm_loss_func,mc_loss_func,lm_test_func
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from utils import preprocess_raw_data

# wandb stuff
import wandb

# evaluation metrics
from eval_utils import MetricFactory
from torchmetrics.functional import precision, recall, f1_score
from trajectory_metric_torch import evaluate_trajectory

def setup_train_args():
    """
    Set training parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_cuda', action='store_true', help='do not use the GPU')
    parser.add_argument('--model_config', default='config/diaformer_config.json', type=str, required=False,
                        help='the config of model')
    parser.add_argument('--max_turn', default=20, type=int, required=False,
                        help='the maximum turn of inquiring implicit symptom.')
    parser.add_argument('--min_probability', default= 0.01, type=float, required=False,
                        help='the minimum probability of inquiring implicit symptom.')
    parser.add_argument('--end_probability', default= 0.9, type=float, required=False,
                        help='the minimum probability of end symbol ([SEP]) to stop inquiring implicit symptom.')
    parser.add_argument('--dataset_path', default='data/synthetic_dataset', type=str, required=False, help='the path of dataset document')
    parser.add_argument('--vocab_path', default = None, type=str, required=False, help='the path of vocab')
    parser.add_argument('--goal_set_path', default = None, type=str, required=False, help='the path of goal_set.p')
    parser.add_argument('--test_set_path', default = None, type=str, help='the path of test_data.pkl')
    parser.add_argument('--train_tokenized_path', default='data/train_tokenized.txt', type=str,
                        required=False,
                        help='Where to store the tokenized train data')
    parser.add_argument('--valid_tokenized_path', default='data/validate_tokenized.txt', type=str,
                        required=False,
                        help='Where to store the tokenized dev data')
    parser.add_argument('--log_path', default='data/training.log', type=str, required=False, help='Where the training logs are stored')
    parser.add_argument('--no_preprocess_data', action='store_true', help='Whether not to tokenize the dataset')
    parser.add_argument('--epochs', default=150, type=int, required=False, help='training epochs')
    parser.add_argument('--batch_size', default=16, type=int, required=False, help='the batch size of training and evaluation')
    parser.add_argument('--lr', default=5e-5, type=float, required=False, help='learning rate')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up steps')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='how much steps to report a loss')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='the accumulation of gradients')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--pretrained_model', type=str, required=False, help='the path of pretrained model')
    parser.add_argument('--seed', type=int, default=8, help='random seed')
    parser.add_argument('--num_workers', type=int, default=1, help="the number of workers used to load data")
    parser.add_argument('--save_dir', type=str, help="root directory to save model and result")

    parser.add_argument('--no_synchronous_learning', action='store_true', help='without synchronous learning')
    parser.add_argument('--no_repeated_sequence', action='store_true', help='without repeated sequence')
    parser.add_argument('--no_sequence_shuffle', action='store_true', help='without sequence shufﬂe')

    parser.add_argument('--start_test', type=int, default=5, help='which epoch start generative test')
    parser.add_argument('--test_every', type=int, default=5, help='frequency of generative test')
    parser.add_argument('--test_size', type=float, default=1, help='size of the test set, either an acutal number or a ratio of the whole test set')
    parser.add_argument("--evi_meta_path", type=str, required=True, help="path to the evidences' meta data file")
    parser.add_argument("--mode", type=str, choices=["train", "generate", "generate_on_test_set"], default="train", help="the mode to run")
    parser.add_argument("--use_differentials_loss", action="store_true", help="calculate loss w.r.t. differentials instead of ground truth pathology")
    parser.add_argument("--compute_differential_metrics", action="store_true", help="calculate differentials' metrics, including KL AUC")
    return parser.parse_args()


def set_random_seed(args):
    """
    Set up random seeds for training
    """
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_logger(args):
    """
    Output logs to log files and consoles
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # log files
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # consoles
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def create_model(args, tokenizer:diaTokenizer):
    """
    create the diaformer
    """
    if args.pretrained_model: 
        # initialize the model using pretrained model
        logger.info('initialize the model using pretrained model')
        model = DiaModel.from_pretrained(args.pretrained_model)
    else:  
        # initialize the model using the cinfig
        logger.info('initialize the model using the cinfig')
        model_config = BertConfig.from_json_file(args.model_config)
        model_config.vocab_size = len(tokenizer.id_to_symptomid)
        model = DiaModel(config=model_config, symlen= len(tokenizer.vocab),dislen= len(tokenizer.disvocab),totalvocalsize=len(tokenizer.id_to_symptomid), num_option_indices=tokenizer.num_option_indices)
    logger.info('model config:\n{}'.format(model.config.to_json_string()))

    return model, model.config.to_dict().get("n_ctx")

def _create_option_tensors(ans_type_list, encoded_option_types, device=None):
    """Represent options and option types with tensors

    Parameters
    ----------
    ans_type_list: list
        a list of all ans_type_idx in the batch
    encoded_option_types: list
        a list of corresponding option_type in the batch
    device: any
        the device to put tensors on

    Return
    ----------
    option_idxs: torch.tensor
        padded option indices in tensor
    option_mask: torch.tensor
        mask for handling padding and numbers
    
    """
    max_num_options = max([len(option) if isinstance(option, list) else 1 for options in ans_type_list for option in options])
    option_idxs = torch.zeros(len(ans_type_list), len(ans_type_list[0]), max_num_options, dtype=torch.long, device=device, requires_grad=False)
    option_mask = torch.zeros(len(ans_type_list), len(ans_type_list[0]), max_num_options, dtype=torch.float, device=device, requires_grad=False)

    for sample_idx, options in enumerate(ans_type_list):
        for pos_idx, option in enumerate(options):
            if encoded_option_types[sample_idx][pos_idx] == "binary":
                option_mask[sample_idx, pos_idx, 0] = 1
                option_idxs[sample_idx, pos_idx, 0] = tokenizer.num_option_indices if option == 1 else tokenizer.num_option_indices + 1
            elif encoded_option_types[sample_idx][pos_idx] == "multi-class":
                option_mask[sample_idx, pos_idx, 0] = 1
                option_idxs[sample_idx, pos_idx, 0] = option
            elif encoded_option_types[sample_idx][pos_idx] == "multi-label":
                option_mask[sample_idx, pos_idx, :len(option)] = 1
                option_idxs[sample_idx, pos_idx, :len(option)] = torch.tensor(option, dtype=torch.long)
            else:
                option_mask[sample_idx, pos_idx, 0] = option
                option_idxs[sample_idx, pos_idx, 0] = tokenizer.num_option_indices

    return option_idxs, option_mask

def _compute_differentials_metrics(differentials, preds, k=None, metrics={}, threshold=0.01):
    """Calculate precision, recall or F1 for differentials. Optionally top-k metrics.

    Parameters
    ----------
    differentials: torch.tensor
        indices of differentials in the batch. Shape: (# patients, max # differentials)
    preds: torch.tensor
        predicted logits of pathologies. Shape: (# patients, # pathos)
    k: int
        number of top predictions to consider
    metrics: iterable
        the metrics to compute
    threshold: float
        threshold for selecting differentials

    Return
    ----------
    results_dict: dict
        a dictionary containing the results

    """
    metric_funcs_dict = {"precision": precision, "recall": recall, "f1": f1_score}

    assert all([metric in {"precision", "recall", "f1"} for metric in metrics])
    
    preds = torch.softmax(preds.detach().cpu(), dim=-1)
    differentials = differentials.detach().cpu()
    differentials_binary = torch.zeros(preds.shape[0], preds.shape[1] + 1, dtype=torch.int)
    differentials_binary[torch.arange(differentials_binary.shape[0]).unsqueeze(-1), differentials] = 1
    differentials_binary = differentials_binary[:, :-1]

    results_dict = {}
    for metric in metrics:
        key = metric if not k else f"{metric}@{k}"
        results_dict[key] = metric_funcs_dict[metric](preds, differentials_binary, average="macro", threshold=threshold, num_classes=differentials_binary.shape[-1], top_k=k).item()

    return results_dict

def collate_fn_train(batch):
    """
    Training data preprocessing.
    Integrate three training mechanisms
    """
    global tokenizer
    global args
    
    tokenids_list = []
    symlabels = []
    dislabels = []
    symlabels_list = []

    encoder_labels = []
    encoder_pos = []
    decoder_pos = []
    decoder_weight = []
    deweight = []
    symlen = []
    repeat_num_list = []
    symlabels = []
    sym_mask = []
    btc_size = len(batch)

    sym_type_list = []
    ans_type_list = []

    # The longest input in the batch, used for the data alignment of the batch
    max_input_len = 0  
    label_maxlen = 0
    
    encoded_option_types = []

    differentials, differentials_probs = [], []

    for btc_idx, sample in enumerate(batch):
        symlen.append(sample["symlen"])
        sym_mask.extend(sample["sym_mask"]["part_1"])
        symlabels.extend(sample["symlabels"])
        deweight.extend(sample["deweight"])
        sym_mask.append(sample["sym_mask"]["part_2"]) 
        encoder_labels.append(sample["encoder_labels"])
        encoder_pos.append(sample["encoder_pos"])
        repeat_num_list.append(sample["repeat_num_list"])
        sym_mask.extend(sample["sym_mask"]["part_3"])
        tokenids_list.append(sample["tokenids_list"])
        sym_type_list.append(sample["sym_type_list"])
        ans_type_list.append(sample["ans_type_list"])
        dislabels.append(sample["dislabels"])
        decoder_pos.append(sample["decoder_pos"])
        encoded_option_types.append(sample["encoded_option_types"])
        differentials.append(torch.tensor(sample["differentials"], dtype=torch.long))
        differentials_probs.append(torch.tensor(sample["differentials_probs"], dtype=torch.float))
        
        if label_maxlen < len(sample["decoder_pos"]):
            label_maxlen = len(sample["decoder_pos"])

        if max_input_len < len(sample["tokenids_list"]):
            max_input_len = len(sample["tokenids_list"])
    
    # attention mask input, 0 means can't see the token of the corresponding position
    attn_mask = torch.zeros(btc_size,max_input_len,max_input_len, dtype=torch.long)

    # padding and complete teh attention mask matrix
    for btc_idx in range(btc_size):
        sym_infer_num,lrlen = symlen[btc_idx]

        attn_mask[btc_idx,0,:lrlen+1] = 1
        # end symbol
        attn_mask[btc_idx,lrlen+sym_infer_num+1,1:lrlen+1] = 1
        attn_mask[btc_idx,lrlen+sym_infer_num+1,lrlen+1+sym_infer_num] = 1

        # explicit symptoms masked
        attn_mask[btc_idx,:lrlen+sym_infer_num+2,1:(lrlen-sym_infer_num+1)] = 1
        
        # the masks of inplicit symptoms and [S] sequence
        startpos = lrlen-sym_infer_num+1
        for i in range(lrlen-sym_infer_num+1,lrlen+1):
            attn_mask[btc_idx,i,startpos:i+1] = 1
            attn_mask[btc_idx,i+sym_infer_num,startpos:i] = 1
            attn_mask[btc_idx,i+sym_infer_num,i+sym_infer_num] = 1
            
        # encoder mask
        attn_mask[btc_idx,lrlen+sym_infer_num+2,:lrlen+1] = 1
        attn_mask[btc_idx,lrlen+sym_infer_num+2,lrlen+sym_infer_num+2] = 1

        # repeated sequence mask
        startpos = lrlen+sym_infer_num+3
        repeat_infer_num = sym_infer_num - 1
        for j in range(repeat_num_list[btc_idx]):
            # for explicit symptoms
            attn_mask[btc_idx,startpos:startpos+(repeat_infer_num+repeat_infer_num),1:(lrlen-sym_infer_num+1)] = 1
            for i in range(startpos,startpos+repeat_infer_num):
                attn_mask[btc_idx,i,startpos:i+1] = 1
                attn_mask[btc_idx,i+repeat_infer_num,startpos:i+1] = 1
                attn_mask[btc_idx,i+repeat_infer_num,i+repeat_infer_num] = 1
            startpos += 2*(repeat_infer_num)

        # Padding
        tokenids_list[btc_idx].extend([tokenizer.pad_token_id] * (max_input_len - len(tokenids_list[btc_idx])))
        decoder_pos[btc_idx].extend([0] * (label_maxlen - len(decoder_pos[btc_idx])))

        sym_type_list[btc_idx].extend([0] * (max_input_len - len(sym_type_list[btc_idx])))
        ans_type_list[btc_idx].extend([0] * (max_input_len - len(ans_type_list[btc_idx])))
        encoded_option_types[btc_idx].extend(["binary"] * (max_input_len - len(encoded_option_types[btc_idx])))

    # create index tensors for option types
    option_idxs, option_mask = _create_option_tensors(ans_type_list, encoded_option_types)

    # differentials padding
    differentials = torch.nn.utils.rnn.pad_sequence(differentials, batch_first=True, padding_value=-1).type(torch.long)
    differentials_probs = torch.nn.utils.rnn.pad_sequence(differentials_probs, batch_first=True, padding_value=-1).type(torch.float)

    return torch.tensor(tokenids_list, dtype=torch.long) ,torch.tensor(symlabels,dtype=torch.long),  torch.tensor(dislabels,dtype=torch.long) ,attn_mask, torch.tensor(encoder_labels,dtype=torch.long), \
        torch.tensor(encoder_pos,dtype=torch.long), torch.tensor(decoder_pos,dtype=torch.long), torch.tensor(deweight,dtype=torch.float), torch.tensor(sym_mask,dtype=torch.float), torch.tensor(sym_type_list,dtype=torch.long), \
        option_idxs, option_mask, differentials, differentials_probs

def collate_fn_eval(batch):
    """
    Evaluating data preprocessing.
    """
    global tokenizer
    tokenids_list = []
    symlabels = []
    dislabels = []
    symlabels_list = []

    encoder_labels = []
    encoder_pos = []
    decoder_pos = []
    decoder_weight = []
    symlen = []
    btc_size = len(batch)
    max_input_len = 0  
    label_maxlen = 0
    pred_num = 0
    sep_pos = []
    sym_type_list = []
    ans_type_list = []
    encoded_option_types = []
    differentials, differentials_probs = [], []

    for btc_idx, sample in enumerate(batch):
        symlen.append(sample["symlen"])
        sep_pos.append(sample["sep_pos"])
        encoder_labels.append(sample["encoder_labels"])
        encoder_pos.append(sample["encoder_pos"])
        tokenids_list.append(sample["tokenids_list"])
        dislabels.append(sample["dislabels"])
        symlabels_list.append(sample["symlabels"])
        decoder_weight.append(sample["deweight"])
        decoder_pos.append(sample["decoder_pos"])
        sym_type_list.append(sample["sym_type_list"])
        ans_type_list.append(sample["ans_type_list"]) 
        encoded_option_types.append(sample["encoded_option_types"])
        differentials.append(torch.tensor(sample["differentials"], dtype=torch.long))
        differentials_probs.append(torch.tensor(sample["differentials_probs"], dtype=torch.float))
        
        if label_maxlen < len(sample["symlabels"]):
            label_maxlen = len(sample["symlabels"])

        if max_input_len < len(sample["tokenids_list"]):
            max_input_len = len(sample["tokenids_list"])
        
        pred_num += sample["pred_num"]
        
    # attention mask 
    attn_mask = torch.zeros(btc_size,max_input_len,max_input_len, dtype=torch.long)

    # padding and complete teh attention mask matrix
    for btc_idx in range(btc_size):
        sym_infer_num,lrlen = symlen[btc_idx]
        attn_mask[btc_idx,0,:lrlen+1] = 1        

        # end symbol
        attn_mask[btc_idx,lrlen+sym_infer_num+1,1:lrlen+1] = 1
        attn_mask[btc_idx,lrlen+sym_infer_num+1,lrlen+1+sym_infer_num] = 1
        # explicit symptoms
        attn_mask[btc_idx,:lrlen+sym_infer_num+2,1:(lrlen-sym_infer_num+1)] = 1
        
        # implicit symptoms and [S] sequences
        startpos = lrlen-sym_infer_num+1
        for i in range(lrlen-sym_infer_num+1,lrlen+1):
            attn_mask[btc_idx,i,startpos:i+1] = 1
            attn_mask[btc_idx,i+sym_infer_num,startpos:i] = 1
            attn_mask[btc_idx,i+sym_infer_num,i+sym_infer_num] = 1
            
        # encoder mask
        attn_mask[btc_idx,lrlen+sym_infer_num+2,:lrlen+1] = 1
        attn_mask[btc_idx,lrlen+sym_infer_num+2,lrlen+sym_infer_num+2] = 1

        # padding
        tokenids_list[btc_idx].extend([tokenizer.pad_token_id] * (max_input_len - len(tokenids_list[btc_idx])))
        symlabels_list[btc_idx].extend([-1] * (label_maxlen - len(symlabels_list[btc_idx])))
        decoder_weight[btc_idx].extend([0] * (label_maxlen - len(decoder_weight[btc_idx])))
        decoder_pos[btc_idx].extend([0] * (label_maxlen - len(decoder_pos[btc_idx])))

        sym_type_list[btc_idx].extend([0] * (max_input_len - len(sym_type_list[btc_idx])))
        ans_type_list[btc_idx].extend([0] * (max_input_len - len(ans_type_list[btc_idx])))
        encoded_option_types[btc_idx].extend(["binary"] * (max_input_len - len(encoded_option_types[btc_idx])))


    # create index tensors for option types
    option_idxs, option_mask = _create_option_tensors(ans_type_list, encoded_option_types)

    # differentials padding
    differentials = torch.nn.utils.rnn.pad_sequence(differentials, batch_first=True, padding_value=-1).type(torch.long)
    differentials_probs = torch.nn.utils.rnn.pad_sequence(differentials_probs, batch_first=True, padding_value=-1).type(torch.float)

    return torch.tensor(tokenids_list, dtype=torch.long) ,torch.tensor(symlabels_list,dtype=torch.long),  torch.tensor(dislabels,dtype=torch.long) ,attn_mask, torch.tensor(encoder_labels,dtype=torch.long), \
        torch.tensor(encoder_pos,dtype=torch.long), torch.tensor(decoder_pos,dtype=torch.long), torch.tensor(decoder_weight,dtype=torch.float),pred_num, torch.tensor(sep_pos,dtype=torch.long), \
        torch.tensor(sym_type_list,dtype=torch.long), option_idxs, option_mask, differentials, differentials_probs


def train(model, device, train_list ,valid_list , tokenizer, args):
    logger.info('train num:{}, dev num:{}'.format(len(train_list),len(valid_list)))

    valid_dataset = diaDataset(valid_list, args.no_sequence_shuffle, args.no_synchronous_learning, args.no_repeated_sequence, tokenizer, max_len)
    train_dataset = diaDataset(train_list, args.no_sequence_shuffle, args.no_synchronous_learning, args.no_repeated_sequence, tokenizer, max_len)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers= args.num_workers,
                                  collate_fn=collate_fn_train, drop_last = True)
    model.train()
    # The total steps of parameter optimization for all epochs were calculated
    total_steps = int(train_dataset.__len__() * args.epochs / args.batch_size / args.gradient_accumulation)
    logger.info('total training steps = {}'.format(total_steps))

    # Set up the optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=total_steps)

    # rng for sampling test set
    rng = np.random.default_rng(args.seed)

    logger.info('starting training')
    # count the loss of each gradient accumulation
    running_loss = 0
    # count how many steps have been trained
    overall_step = 0
    oom_time = 0
    # training time
    traintime = datetime.now()
    traintime = traintime - traintime

    # evaluation metrics
    clf_metrics_func = MetricFactory().evaluate

    # start training
    for epoch in range(args.epochs):
        starttime = datetime.now()
        batch_idx = 0
        losses = 0
        mc_acc = 0.0
        encoder_acc = 0.0
        sym_acc = 0.0
        max_sym_acc = 0.0
        # encoder_acc_num = 0 
        for input_ids, symlabels, dislabels, attn_mask, encoder_labels, encoder_pos, decoder_pos, decoder_weight, sym_mask,sym_type_list, ans_type_list, option_mask, differentials, differentials_probs in tqdm(train_dataloader):
            input_ids = input_ids.to(device)
            symlabels = symlabels.to(device)
            dislabels = dislabels.to(device)
            attn_mask = attn_mask.to(device)
            encoder_labels = encoder_labels.to(device)
            encoder_pos = encoder_pos.to(device)
            decoder_pos = decoder_pos.to(device)
            decoder_weight = decoder_weight.to(device)
            sym_mask = sym_mask.to(device)
            sym_type_list = sym_type_list.to(device)
            # ans_type_list = ans_type_list.to(device)
            ans_type_list = ans_type_list.to(device)
            option_mask = option_mask.to(device)
            differentials = differentials.to(device) if args.use_differential else None
            differentials_probs = differentials_probs.to(device) if args.use_differential else None
            batch_idx += 1

            # Solve the problem of CUDA out of memory caused by insufficient video memory during operation
            try:
                outputs = model.forward(input_ids=input_ids,issym = False, isdis = True, attention_mask= attn_mask, encoderpos = encoder_pos, sym_type_ids = sym_type_list, ans_type_ids = ans_type_list, option_mask=option_mask)
                # symptom loss
                sym_loss,sym_accuracy = lm_loss_func(outputs[0].to(device)[...,:len(tokenizer.vocab)], symlabels, decoder_pos, decoder_weight, sym_mask, class_weight=tokenizer.class_weight.to(device))
                sym_acc += sym_accuracy
                
                # disease loss
                mc_loss, mc_accuracy = mc_loss_func(outputs[2].to(device), mc_labels=dislabels, \
                    differentials=differentials if args.use_differentials_loss else None, differentials_probs=differentials_probs if args.use_differentials_loss else None)
                mc_acc += mc_accuracy

                # disease classification metrics
                clf_results_dict = {f"{key}/train": clf_metrics_func(
                    key=key, y_true=dislabels.cpu(), zero_division=0,
                    y_pred=torch.argmax(outputs[2].detach().cpu(), dim=1).numpy()
                    ) for key in {"f1", "precision", "recall"}}
                clf_results_dict["epoch"] = epoch
                wandb.log(clf_results_dict)

                # differentials metrics
                if args.compute_differential_metrics:
                    diffs_results_dict = _compute_differentials_metrics(differentials=differentials, preds=outputs[2], metrics={"precision", "recall", "f1"})
                    diffs_results_dict = {f"differential_{key}/train": value for key, value in diffs_results_dict.items()}
                    top_5_diffs_results_dict = _compute_differentials_metrics(differentials=differentials, preds=outputs[2], metrics={"precision", "recall", "f1"}, k=5)
                    top_5_diffs_results_dict = {f"differential_{key}/train": value for key, value in top_5_diffs_results_dict.items()}
                    diffs_results_dict.update(top_5_diffs_results_dict)
                    diffs_results_dict["epoch"] = epoch
                    wandb.log(diffs_results_dict)

                # autoencoding loss
                # encoder_loss,encoder_accuracy = mc_loss_func(outputs[3].to(device),mc_labels=encoder_labels)
                # encoder_acc += encoder_accuracy

                # loss
                loss = mc_loss + sym_loss

                if args.multi_gpu:
                    loss = loss.mean()
                    # accuracy = accuracy.mean()
                if args.gradient_accumulation > 1:
                    loss = loss / args.gradient_accumulation
                    # accuracy = accuracy / args.gradient_accumulation
                loss.backward()
                # wandb logging
                wandb.log({"batch loss/train": loss.item(), "epoch": epoch})
                # Gradient cropping solves the problem of gradient disappearance or explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # gradient accumulate
                if batch_idx % args.gradient_accumulation == 0:
                    running_loss += loss.item()
                    # update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    # warm up
                    scheduler.step()
                    overall_step += 1
                    if (overall_step + 1) % args.log_step == 0:
                        losses += loss
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    oom_time += 1
                    logger.info("WARNING: ran out of memory,times: {}".format(oom_time))
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    logger.info(str(exception))
                    raise exception
        traintime += (datetime.now() - starttime)
        logger.info('epoch {} finished, total training time: {}'.format(epoch + 1,traintime))
        losses /= batch_idx
        # due to orderless training mechanism, the symptom accuracy here is not true！
        logger.info("Total training loss: {}, sym_acc:{},  dis_acc: {}".format(losses, sym_acc/batch_idx, mc_acc / batch_idx))

        # wandb logging
        wandb.log({
            "loss/train": losses, "symptom_accuracy/train": sym_acc / batch_idx, "disease_accuracy/train": mc_acc / batch_idx, "epoch": epoch,
        })
        
        evaluate(model, device, valid_dataset, args, epoch)
    
        # start test
        if epoch >= args.start_test - 1 and (epoch + 1) % args.test_every == 0:
            logger.info ("Start testing epoch{}".format(epoch + 1))
            # evaluate the metrics of automatic diagnosis on test set
            generate(model, device, tokenizer ,args, epoch, rng)
        
    logger.info('training finished')
    

# evaluating function
def evaluate(model, device, valid_dataset, args, epoch):
    model.eval()
    logger.info('starting evaluating')
    test_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                 collate_fn=collate_fn_eval, drop_last = True)
    
    # evaluation metrics
    clf_metrics_func = MetricFactory().evaluate
    
    total_num = 0
    all_preds_num = 0
    mc_acc = 0
    with torch.no_grad():
        mc_acc = 0.0
        encoder_acc = 0.0
        sep_acc = 0.0
        sym_acc = 0.0
        for input_ids, symlabels, dislabels, attn_mask, encoder_labels, encoder_pos, decoder_pos, decoder_weight,pred_num,sep_pos,sym_type_list,ans_type_list,option_mask, differentials, differentials_probs in tqdm(test_dataloader):
            input_ids = input_ids.to(device)
            symlabels = symlabels.to(device)
            dislabels = dislabels.to(device)
            attn_mask = attn_mask.to(device)
            encoder_labels = encoder_labels.to(device)
            encoder_pos = encoder_pos.to(device)
            decoder_pos = decoder_pos.to(device)
            decoder_weight = decoder_weight.to(device)
            sep_pos = sep_pos.to(device)
            sym_type_list = sym_type_list.to(device)
            ans_type_list = ans_type_list.to(device)
            option_mask = option_mask.to(device)
            differentials = differentials if args.use_differential else None
            differentials_probs = differentials_probs if args.use_differential else None
            total_num += 1
            all_preds_num += pred_num
            
            outputs = model.forward(input_ids=input_ids,issym = False, isdis = True, attention_mask= attn_mask, encoderpos = encoder_pos, sym_type_ids = sym_type_list, ans_type_ids = ans_type_list, option_mask=option_mask)
            # symptom inquiry
            sym_accuracy,sep_accuracy = lm_test_func(outputs[0].to(device), symlabels, decoder_pos, decoder_weight,pred_num,sep_pos, tokenizer.sep_token_id)
            sym_acc += sym_accuracy*pred_num
            # the accuracy of end inquiry prediction
            sep_acc += sep_accuracy
            
            # disease prediction
            _, mc_accuracy = mc_loss_func(outputs[2].to(device), mc_labels=dislabels)
            mc_acc += mc_accuracy

            # disease classification metrics
            clf_results_dict = {f"{key}/evaluate": clf_metrics_func(
                key=key, y_true=dislabels.cpu(), zero_division=0,
                y_pred=torch.argmax(outputs[2].detach().cpu(), dim=1).numpy()
                ) for key in {"f1", "precision", "recall"}}
            clf_results_dict["epoch"] = epoch
            wandb.log(clf_results_dict)

            # differentials metrics
            if args.compute_differential_metrics:
                diffs_results_dict = _compute_differentials_metrics(differentials=differentials, preds=outputs[2], metrics={"precision", "recall", "f1"})
                diffs_results_dict = {f"differential_{key}/evaluate": value for key, value in diffs_results_dict.items()}
                top_5_diffs_results_dict = _compute_differentials_metrics(differentials=differentials, preds=outputs[2], metrics={"precision", "recall", "f1"}, k=5)
                top_5_diffs_results_dict = {f"differential_{key}/evaluate": value for key, value in top_5_diffs_results_dict.items()}
                diffs_results_dict.update(top_5_diffs_results_dict)
                diffs_results_dict["epoch"] = epoch
                wandb.log(diffs_results_dict)

            # auto encodering
            _,encoder_accuracy = mc_loss_func(outputs[3].to(device),mc_labels=encoder_labels)
            encoder_acc += encoder_accuracy
        
        total_num = 1 if total_num==0 else total_num
        # evaluate the disease accuracy in the test set (due to orderless training mechanism, the symptom accuracy here is not true and is ignored)
        logger.info("evaluate overall: dis_accuracy {}".format(mc_acc / total_num))
        logger.info("finishing evaluating")

        # wandb logging
        wandb.log({"disease_accuracy/valid": mc_acc / total_num, "epoch": epoch})

        return mc_acc / total_num


maxscore = 0
testdata = None
max_len = 200
max_score = 0.0
# Use generation to simulate the diagnostic process
@torch.no_grad()
def generate(model, device, tokenizer: diaTokenizer, args, epoch, rng=None):
    global testdata
    global max_score
    if testdata is None:
        if args.mode != "generate_on_test_set":
            with open(args.goal_set_path,'rb') as f:
                testdata = pickle.load(f)
            testdata = testdata['test']
        else:
            with open(args.test_set_path, "rb") as file:
                testdata = pickle.load(file)
            testdata = testdata['test']

    # the result list
    reslist = []

    # record of symptom inquiry
    mc_acc = 0
    imp_acc = 0
    imp_all = 0
    imp_recall = 0

    # sample test set
    test_size = int(args.test_size) if args.test_size > 1 else int(len(testdata) * args.test_size)
    assert test_size <= len(testdata)
    if rng is not None:
        test_idxs = sorted(list(rng.choice(len(testdata), size=test_size, replace=False)))
    else:
        test_idxs = list(range(len(testdata)))[:test_size]

    len_test_data = len(test_idxs)

    # evaluation metrics
    clf_metrics_func = MetricFactory().evaluate

    def _predict_disease():
        curr_input_tensor = torch.tensor([[tokenizer.cls_token_id] + input_ids], dtype=torch.long, device=device)
        attn_mask = torch.zeros(1,len(input_ids)+1,len(input_ids)+1, device=device)
        explen = 1
        explen += 1
        attn_mask[0,:,1:explen] = 1
        for i in range(explen,len(input_ids)+1):
            attn_mask[0,i,explen:i+1] = 1
        attn_mask[0,0,:] = 1
        explen -= 1
        sym_type_list = torch.tensor([[0]+[1]*(explen)+[2]*(len(input_ids)-explen)], dtype=torch.long, device=device)
        # ans_type_list = torch.tensor([[0]+[1 if x < len(tokenizer.vocab) else 2 for x in input_ids]], dtype=torch.long, device=device)[0]
        ans_type_list = [0] + options_list
        encoded_option_types = ["binary"] + option_types_list
        option_idxs, option_mask = _create_option_tensors([ans_type_list], [encoded_option_types], device=device)
        outputs = model(input_ids=curr_input_tensor, attention_mask = attn_mask, issym = False, isdis = True,sym_type_ids = sym_type_list, ans_type_ids = option_idxs, option_mask=option_mask)
        mc_logits = outputs[2][0]
        # mc_logits = F.softmax(mc_logits, dim=-1)
        return mc_logits

    # save disease predictions
    all_disease_logits = []

    differentials, differentials_probs = [], []

    # start simulation for each testing data
    for idx in tqdm(test_idxs):
        item = testdata[idx]
        if args.use_differential:
            differentials.append(torch.tensor([tokenizer.disvocab[patho] for patho, _ in item["differentials"]], dtype=torch.long))
            differentials_probs.append(torch.tensor([proba for _, proba in item["differentials"]], dtype=torch.float))
        # input_ids = [] 
        explicit_ids = []
        disease_logits = []
        # Expset records explicit symptoms
        expset = set()
        for exp,label in item['goal']['explicit_inform_slots'].items():
            if label == 'UNK':
                continue
            symid = tokenizer.convert_token_to_id(exp)
            expset.add(symid)
            # if label:
            #     input_ids.append(symid)
            # else:
            #     input_ids.append(tokenizer.symptom_to_false[symid])
            explicit_ids.append({
                "token": symid, "token_type": "explicit_symptom", "option": label["option"], "option_type": label["option_type"],
            })
        input_ids = [explicit_id["token"] for explicit_id in explicit_ids]

        # reserve the implicit symptoms
        impslots = {}
        for exp,label in item['goal']['implicit_inform_slots'].items():
            if label == 'UNK':
                continue
            # if len(input_ids) == 0:
            if len(explicit_ids) == 0:
                # to avoid none explicit symptom in extreme cases
                symid = tokenizer.convert_token_to_id(exp)
                expset.add(symid)
                explicit_ids.append({
                    "token": symid, "token_type": "explicit_symptom", "option": label["option"], "option_type": label["option_type"],
                })
            else:
                impslots[tokenizer.convert_token_to_id(exp)] = label
        
        explen = len(expset)
        imp_all += len(impslots)

        # save all the requiry symptom
        generated = []
        options_list = [explicit_id["option"] for explicit_id in explicit_ids]
        option_types_list = [explicit_id["option_type"] for explicit_id in explicit_ids]

        for len_idx in range(max_len):
            # input tokens
            curr_input_tensor = torch.tensor([input_ids+[tokenizer.sep_token_id]], dtype=torch.long, device=device)
            # attention masks
            attn_mask = torch.zeros(1,len(input_ids)+1,len(input_ids)+1, device=device)
            attn_mask[0,:,0:explen] = 1
            for i in range(explen,len(input_ids)):
                attn_mask[0,i,explen:i+1] = 1
            attn_mask[0,len(input_ids),:] = 1
            attn_mask = attn_mask

            sym_type_list = torch.tensor([[2]*explen+[1]*(len(input_ids)-explen)+[0]], dtype=torch.long, device=device)
            # ans_type_list = torch.tensor([[1 if x < len(tokenizer.vocab) else 2 for x in input_ids]+[0]], dtype=torch.long, device=device)
            ans_type_list = options_list + [0]
            encoded_option_types = option_types_list + ["binary"]
            option_idxs, option_mask = _create_option_tensors([ans_type_list], [encoded_option_types], device=device)
            outputs = model(input_ids=curr_input_tensor, attention_mask = attn_mask,issym = False, isdis = False,sym_type_ids = sym_type_list, ans_type_ids = option_idxs, option_mask=option_mask)
            next_token_logits = outputs[0][0][len(input_ids)]

            # obtain the probability of inquiry symptoms
            next_token_logits = F.softmax(next_token_logits, dim=-1)
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            # whether stop inquring symptoms
            isDiease = False
            # find the next maximum probability of inquiry symptom
            for index,token_id in enumerate(sorted_indices):
                token_id = tokenizer.id_to_symptomid[token_id.item()]
                if len(generated) >= args.max_turn:
                    isDiease = True
                    break
                elif token_id == tokenizer.sep_token_id and sorted_logits[index] > args.end_probability:
                    isDiease = True
                    break
                elif token_id in expset:
                # check if the symptom inquired is a explicit symptoms
                    continue
                elif token_id in generated:
                # check if the symptom has been inquired 
                    continue
                elif token_id in tokenizer.special_tokens_id or token_id in tokenizer.tokenid_to_diseaseid:
                    continue
                elif sorted_logits[index] < args.min_probability:
                    isDiease = True
                    break
                else:
                    # inquire symptom
                    if token_id in impslots:
                        # in implicit symptom set
                        imp_acc += 1
                        generated.append(token_id)
                        addid = token_id if impslots[token_id] else tokenizer.symptom_to_false[token_id]
                        input_ids.append(addid)
                        options_list.append(impslots[token_id]["option"])
                        option_types_list.append(impslots[token_id]["option_type"])
                        break
                    else:
                        # not in implicit symptom set
                        generated.append(token_id)
                        mc_logits = _predict_disease()
                        disease_logits.append(mc_logits.detach().cpu())
            
            if isDiease:
                mc_logits = _predict_disease()
                disease_logits.append(mc_logits.detach().cpu())
                _, pre_disease = mc_logits.max(dim=-1)
                generated.append(pre_disease.item())
                break
        
        if item['disease_tag'] == tokenizer.convert_label_to_disease(generated[-1]):
            mc_acc += 1
        # res = {'symptom': [tokenizer.convert_id_to_token(x) for x in generated[:-1]] , 'disease': tokenizer.convert_label_to_disease(generated[-1])}
        res = {'explicit_symptoms':item['goal']['explicit_inform_slots'],'implicit_symptoms':item['goal']['implicit_inform_slots'],'target_disease':item['disease_tag'],'inquiry_symptom': [tokenizer.convert_id_to_token(x) for x in generated[:-1]] , 'pred_disease': tokenizer.convert_label_to_disease(generated[-1])}
        reslist.append(res)
        imp_recall += (len(generated)-1)

        all_disease_logits.append(disease_logits)

    # disease classification metrics
    clf_results_dict = {f"{key}/generate": clf_metrics_func(
        key=key, zero_division=0, y_true=np.array([res["target_disease"] for res in reslist]),
        y_pred=np.array([res["pred_disease"] for res in reslist])
        ) for key in {"f1", "precision", "recall"}}
    clf_results_dict["epoch"] = epoch
    wandb.log(clf_results_dict)
        
    # total metric
    tscore =  0.8*mc_acc/len_test_data+0.4*imp_acc/(imp_all+imp_recall)
    
    if tscore > max_score: 
        max_score = tscore
        if args.model_output_path is not None:
            logger.info('model saved')
            max_score = tscore
            if not os.path.exists(args.model_output_path):
                os.makedirs(args.model_output_path)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(args.model_output_path)
        if args.result_output_path is not None:
            logger.info('results saved')
            with open(args.result_output_path,'w') as f:
                json.dump(reslist,f,ensure_ascii=False,indent=4)
        if args.test_idxs_output_path is not None:
            logger.info("test idxs saved")
            with open(args.test_idxs_output_path, "wb") as file:
                pickle.dump(test_idxs, file)
        if args.disease_logits_output_path is not None:
            logger.info("disease logits saved")
            with open(args.disease_logits_output_path, "wb") as file:
                pickle.dump(all_disease_logits, file)
    logger.info('generative results\n sym_recall:{}, disease:{}, avg_turn:{}'.format(imp_acc/imp_all,mc_acc/len_test_data,imp_recall/len_test_data))
    # wandb logging
    wandb.log({
        "symptom_recall/generate": imp_acc / imp_all,
        "disease_accuracy/generate": mc_acc / len_test_data,
        "average_turn/generate": imp_recall / len_test_data,
        "epoch": epoch,
        })

    # differentials metrics
    if args.compute_differential_metrics:
        differentials = torch.nn.utils.rnn.pad_sequence(differentials, batch_first=True, padding_value=-1).type(torch.long)
        differentials_probs = torch.nn.utils.rnn.pad_sequence(differentials_probs, batch_first=True, padding_value=-1).type(torch.float)
        preds = torch.vstack([trajectory[-1] for trajectory in all_disease_logits])
        diffs_results_dict = _compute_differentials_metrics(differentials=differentials, preds=preds, metrics={"precision", "recall", "f1"})
        diffs_results_dict = {f"differential_{key}/generate": value for key, value in diffs_results_dict.items()}
        top_5_diffs_results_dict = _compute_differentials_metrics(differentials=differentials, preds=preds, metrics={"precision", "recall", "f1"}, k=5)
        top_5_diffs_results_dict = {f"differential_{key}/generate": value for key, value in top_5_diffs_results_dict.items()}
        diffs_results_dict.update(top_5_diffs_results_dict)
        diffs_results_dict["epoch"] = epoch
        wandb.log(diffs_results_dict)

        # trajectory metrics
        when_ends = torch.tensor([len(trajectory) - 1 for trajectory in all_disease_logits])
        traj_disease_logits = [torch.vstack(trajectory).cpu() for trajectory in all_disease_logits]
        traj_disease_logits = torch.nn.utils.rnn.pad_sequence(traj_disease_logits, batch_first=True)
        kl_auc_results_dict = evaluate_trajectory(traj_disease_logits, when_ends=when_ends, differentials=differentials, differentials_probs=differentials_probs, which_metrics={"kl_auc"}, summarize="mean")
        kl_auc_results_dict = {f"{key}/generate": torch.mean(value).item() for key, value in kl_auc_results_dict.items() if "mean" in key}
        kl_auc_results_dict["epoch"] = epoch
        wandb.log(kl_auc_results_dict)

# tokenizer = None
def main():
    global args
    args = setup_train_args()
    global logger
    logger = create_logger(args)

    if args.mode != "generate_on_test_set" and args.test_set_path is not None:
        logger.warning("test data path is provided but mode is not generate_on_test_set")

    # add flag for loading differentials
    args.use_differential = args.use_differentials_loss or args.compute_differential_metrics

    # handle SLURM_TMPDIR for tokenized data paths
    args.train_tokenized_path = args.train_tokenized_path.replace("$SLURM_TMPDIR", os.environ["SLURM_TMPDIR"])
    args.valid_tokenized_path = args.valid_tokenized_path.replace("$SLURM_TMPDIR", os.environ["SLURM_TMPDIR"])

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda' if args.cuda else 'cpu'
    device = torch.device(device)
    logger.info('using device:{}'.format(device))

    if args.seed:
        set_random_seed(args)

    if args.vocab_path is None:
        args.vocab_path = os.path.realpath(join(args.dataset_path,'vocab.txt'))
    if args.goal_set_path is None:
        args.goal_set_path = os.path.realpath(join(args.dataset_path,'goal_set.p'))
    if args.mode == "generate_on_test_set" and args.test_set_path is None:
        args.test_set_path = os.path.realpath(join(args.dataset_path, "test_data.pkl"))

    # Initializes tokenizer
    global tokenizer
    tokenizer = diaTokenizer(vocab_file=args.vocab_path, evi_meta_path=args.evi_meta_path)

    # Load the model
    model, n_ctx = create_model(args, tokenizer)
    model.to(device)
    
    if not args.no_preprocess_data and args.mode == "train":
        preprocess_raw_data(args, logger, tokenizer, n_ctx)

    args.multi_gpu = False
    # if you need multi-GPU to process the mass data, please enable the DataParallel.
    # if args.cuda and torch.cuda.device_count() > 1:
    #     logger.info("Let's use GPUs to train")
    #     model = DataParallel(model, device_ids=[int(i) for i in args.device.split(',')])
    #     multi_gpu = True

    # Record the number of model parameters
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    logger.info('number of model parameters: {}'.format(num_parameters))

    if args.mode == "train":
        logger.info("loading train data")
        with open(os.path.realpath(args.train_tokenized_path), "r") as f:
            train_list = json.load(f)
        # train_list = train_data.split("\n")

        logger.info("loading valid data")
        with open(os.path.realpath(args.valid_tokenized_path), "r") as f:
            valid_list = json.load(f)
        # valid_list = valid_data.split("\n")

    # time stamp and output dir
    time_stamp = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    output_dir = os.path.realpath(os.path.join(args.save_dir, time_stamp))
    args.model_output_path = output_dir
    args.result_output_path = os.path.join(output_dir, "results.json")
    args.disease_logits_output_path = os.path.join(output_dir, "val_disease_logits.pkl" if args.mode != "generate_on_test_set" else "test_disease_logits.pkl")
    args.test_idxs_output_path = os.path.join(output_dir, "test_idxs.pkl" if args.mode != "generate_on_test_set" else "test_idxs_for_test_set.pkl")

    # setup wandb
    wandb.init(name=time_stamp, group="supervised_baseline_diaformer", project="medical_evidence_collection")
    wandb.config.update(args)
    
    # training and testing
    if args.mode == "train":
        train(model, device, train_list ,valid_list, tokenizer, args)
    else:
        # only testing
        generate(model, device, tokenizer, args, 0)


if __name__ == '__main__':
    main()