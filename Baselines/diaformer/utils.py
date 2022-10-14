import json
import pickle
import torch
from tqdm import tqdm
from tokenizer import diaTokenizer

def preprocess_raw_data(args, logger, tokenizer : diaTokenizer, n_ctx):
    logger.info("dataset path:{}".format(args.goal_set_path))
    logger.info("tokenizing train raw data, token output path:{}".format(args.train_tokenized_path))

    with open(args.goal_set_path,'rb') as f:
        data = pickle.load(f)
    testdata = data['test']

    # train dataset
    exp_sym_num = [0]*len(tokenizer.vocab)
    imp_sym_num = [0]*len(tokenizer.vocab)

    data = data['train']
    logger.info("there are {} dialogue in train dataset".format(len(data)))

    datalist = []
    for item_index, item in enumerate(tqdm(data)):
        explicit_ids = []
        for exp,label in item['goal']['explicit_inform_slots'].items():
            # label will have these fields: option, option_type
            symid = tokenizer.convert_token_to_id(exp)
            exp_sym_num[symid] += 1
            if label == 'UNK':
                continue
            # if label:
            #     explicit_ids.append(symid)
            # else:
            #     explicit_ids.append(tokenizer.symptom_to_false[symid])
            explicit_ids.append({
                "token": symid, "token_type": "explicit_symptom", "option": label["option"], "option_type": label["option_type"],
            })

        implicit_ids = []
        for exp,label in item['goal']['implicit_inform_slots'].items():
            if label == 'UNK':
                continue
            symid = tokenizer.convert_token_to_id(exp)
            imp_sym_num[symid] += 1
            # if label:
            #     implicit_ids.append(symid)
            # else:
            #     implicit_ids.append(tokenizer.symptom_to_false[symid])
            implicit_ids.append({
                "token": symid, "token_type": "implicit_symptom", "option": label["option"], "option_type": label["option_type"],
            })

        differentials = []
        if args.use_differential:
            for patho, proba in item["differentials"]:
                differentials.append({
                    "token": tokenizer.disvocab[patho], "token_type": "differential", "option": proba, "option_type": "number",
                })
        
        # drop the data with none symptom
        if len(explicit_ids) != 0 or len(implicit_ids) != 0:
            # datalist.append(' '.join([str(token_id) for token_id in explicit_ids])+'\t'+' '.join([str(token_id) for token_id in implicit_ids])+'\t'+str(tokenizer.disvocab[item['disease_tag']]))
            datalist.append(explicit_ids + implicit_ids + [{"token": tokenizer.disvocab[item['disease_tag']], "token_type": "disease", "option": 1, "option_type": "binary",}] + differentials)
    with open(args.train_tokenized_path, "w") as f:
        json.dump(datalist, f)
    logger.info('data sample:{} '.format(datalist[-1]))

    max_num =  max(imp_sym_num)
    class_weight = torch.tensor([max_num/sym_num if sym_num > 0 else 0 for sym_num in imp_sym_num])
    tokenizer.class_weight = class_weight.sqrt()
    assert len(tokenizer.class_weight) == len(tokenizer.vocab)

    # valid dataset
    data = testdata
    logger.info("there are {} dialogue in valid dataset".format(len(data)))

    # [cls] explicit_inform [pad] implicit_inform [sep] [sep] means the model could infer the disease
    datalist = []
    for item_index, item in enumerate(tqdm(data)):
        explicit_ids = []
        for exp,label in item['goal']['explicit_inform_slots'].items():
            # label will have these fields: option, option_type
            if label == 'UNK':
                continue
            symid = tokenizer.convert_token_to_id(exp)
            # if label:
            #     explicit_ids.append(symid)
            # else:
            #     explicit_ids.append(tokenizer.symptom_to_false[symid])
            explicit_ids.append({
                "token": symid, "token_type": "explicit_symptom", "option": label["option"], "option_type": label["option_type"],
            })

        implicit_ids = []
        for exp,label in item['goal']['implicit_inform_slots'].items():
            if label == 'UNK':
                continue
            symid = tokenizer.convert_token_to_id(exp)
            # if label:
            #     implicit_ids.append(symid)
            # else:
            #     implicit_ids.append(tokenizer.symptom_to_false[symid])
            implicit_ids.append({
                "token": symid, "token_type": "implicit_symptom", "option": label["option"], "option_type": label["option_type"],
            })

        differentials = []
        if args.use_differential:
            for patho, proba in item["differentials"]:
                differentials.append({
                    "token": tokenizer.disvocab[patho], "token_type": "differential", "option": proba, "option_type": "number",
                })

        # drop the data with none symptom
        if len(explicit_ids) != 0 or len(implicit_ids) != 0:
            # datalist.append(' '.join([str(token_id) for token_id in explicit_ids])+'\t'+' '.join([str(token_id) for token_id in implicit_ids])+'\t'+str(tokenizer.disvocab[item['disease_tag']]))
            datalist.append(explicit_ids + implicit_ids + [{"token": tokenizer.disvocab[item['disease_tag']], "token_type": "disease", "option": 1, "option_type": "binary",}] + differentials)
    with open(args.valid_tokenized_path, "w") as f:
        json.dump(datalist, f)
    logger.info('data sample:{} '.format(datalist[-1]))

    logger.info("finish preprocessing raw data,the result is stored in {} and {}".format(args.train_tokenized_path,args.valid_tokenized_path))

def _depreciated_preprocess_raw_data(args, logger, tokenizer : diaTokenizer, n_ctx):
    """
    Processing the original data and tokenize each symptoms of data to tokens
    convert to explicit symptom tokens \t implicit symptom tokens \t disease
    """
    logger.info("dataset path:{}".format(args.goal_set_path))
    logger.info("tokenizing train raw data, token output path:{}".format(args.train_tokenized_path))

    with open(args.goal_set_path,'rb') as f:
        data = pickle.load(f)
    testdata = data['test']
    
    # train dataset
    exp_sym_num = [0]*len(tokenizer.vocab)
    imp_sym_num = [0]*len(tokenizer.vocab)

    data = data['train']
    logger.info("there are {} dialogue in train dataset".format(len(data)))
    with open(args.train_tokenized_path, "w", encoding="utf-8") as f:
        datalist = []
        for item_index, item in enumerate(tqdm(data)):
            explicit_ids = []
            for exp,label in item['goal']['explicit_inform_slots'].items():
                symid = tokenizer.convert_token_to_id(exp)
                exp_sym_num[symid] += 1
                if label == 'UNK':
                    continue
                if label:
                    explicit_ids.append(symid)
                else:
                    explicit_ids.append(tokenizer.symptom_to_false[symid])

            implicit_ids = []
            for exp,label in item['goal']['implicit_inform_slots'].items():
                if label == 'UNK':
                    continue
                symid = tokenizer.convert_token_to_id(exp)
                imp_sym_num[symid] += 1
                if label:
                    implicit_ids.append(symid)
                else:
                    implicit_ids.append(tokenizer.symptom_to_false[symid])
            
            # drop the data with none symptom
            if len(explicit_ids) != 0 or len(implicit_ids) != 0:
                datalist.append(' '.join([str(token_id) for token_id in explicit_ids])+'\t'+' '.join([str(token_id) for token_id in implicit_ids])+'\t'+str(tokenizer.disvocab[item['disease_tag']]))

        for item_index,dataitem in enumerate(datalist):
            if item_index < len(datalist) - 1:
                f.write(dataitem+"\n")
            else:
                f.write(dataitem)
                logger.info('data sample:{} '.format(dataitem))
        
        # logger.info("explict & implict symptoms distribution:")
        # logger.info([(x,y) for x,y in zip(exp_sym_num,imp_sym_num)])
        max_num =  max(imp_sym_num)
        class_weight = torch.tensor([max_num/sym_num if sym_num > 0 else 0 for sym_num in imp_sym_num])
        tokenizer.class_weight = class_weight.sqrt()
        assert len(tokenizer.class_weight) == len(tokenizer.vocab)

    # valid dataset
    data = testdata
    logger.info("there are {} dialogue in valid dataset".format(len(data)))
    with open(args.valid_tokenized_path, "w", encoding="utf-8") as f:
        # [cls] explicit_inform [pad] implicit_inform [sep] [sep] means the model could infer the disease
        datalist = []
        for item_index, item in enumerate(tqdm(data)):
            explicit_ids = []
            for exp,label in item['goal']['explicit_inform_slots'].items():
                if label == 'UNK':
                    continue
                symid = tokenizer.convert_token_to_id(exp)
                if label:
                    explicit_ids.append(symid)
                else:
                    explicit_ids.append(tokenizer.symptom_to_false[symid])

            implicit_ids = []
            for exp,label in item['goal']['implicit_inform_slots'].items():
                if label == 'UNK':
                    continue
                symid = tokenizer.convert_token_to_id(exp)
                if label:
                    implicit_ids.append(symid)
                else:
                    implicit_ids.append(tokenizer.symptom_to_false[symid])

            # drop the data with none symptom
            if len(explicit_ids) != 0 or len(implicit_ids) != 0:
                datalist.append(' '.join([str(token_id) for token_id in explicit_ids])+'\t'+' '.join([str(token_id) for token_id in implicit_ids])+'\t'+str(tokenizer.disvocab[item['disease_tag']]))

        for item_index,dataitem in enumerate(datalist):
            if item_index < len(datalist) - 1:
                f.write(dataitem+"\n")
            else:
                f.write(dataitem)
                logger.info('data sample:{} '.format(dataitem))
    logger.info("finish preprocessing raw data,the result is stored in {} and {}".format(args.train_tokenized_path,args.valid_tokenized_path))
