from torch.utils.data import Dataset
import random
from sklearn.utils import shuffle

# Dataset of Diaformer
class diaDataset(Dataset):

    def __init__(self, data_list, no_sequence_shuffle, no_synchronous_learning, no_repeated_sequence, tokenizer, max_len):
        self.data_list = data_list
        self.no_sequence_shuffle = no_sequence_shuffle
        self.no_synchronous_learning = no_synchronous_learning
        self.no_repeated_sequence = no_repeated_sequence
        self.tokenizer = tokenizer
        self.max_len = max_len

    def _parse_data(self, data):
        """Parse a patient's data that is a dictionary containing rich information into the original formats of Diaformer, and 
        outputs options (ans_type).

        The input, data, should be of following format:
        [
            {
                token: evidence/disease index
                token_type: specify the type of token, i.e. explicit symptom, implicit symptom, or disease
                option: options' value, could be 1, index, a list of indices, or a float
                option_type: options' type
            },
        ]

        Parameters
        ----------
        data: dict
            a dictionary containing rich patient's data

        Return
        ----------
        input_ids: tuple
            encoded disease and (explicit and implicit) evidences, as in original Diaformer format
        options: tuple
            options of evidences in corresponding positions
        options_type: tuple
            strings specifying the type of option of each evidence in corresponding positions
        differentials: list
            a list of pathologies in the differential set
        differentials_probs: list
            differentials' probabilities

        """
        explicit_symptoms, implicit_symptoms, explicit_options, implicit_options, explicit_option_types, implicit_option_types, differentials, differentials_probs = [], [], [], [], [], [], [], []
        
        for token_data in data:
            if token_data["token_type"] == "disease":
                disease = token_data["token"]
            elif token_data["token_type"] == "differential":
                differentials.append(token_data["token"])
                differentials_probs.append(token_data["option"])
            elif token_data["token_type"] == "explicit_symptom":
                explicit_symptoms.append(token_data["token"])
                explicit_option_types.append(token_data["option_type"])
                if token_data["option_type"] in {"binary", "multi-class"}:
                    explicit_options.append(int(token_data["option"]))
                elif token_data["option_type"] == "multi-label":
                    explicit_options.append([int(option) for option in token_data["option"]])
                else:
                    explicit_options.append(float(token_data["option"]))
            else:
                implicit_symptoms.append(token_data["token"])
                implicit_option_types.append(token_data["option_type"])
                if token_data["option_type"] in {"binary", "multi-class"}:
                    implicit_options.append(int(token_data["option"]))
                elif token_data["option_type"] == "multi-label":
                    implicit_options.append([int(option) for option in token_data["option"]])
                else:
                    implicit_options.append(float(token_data["option"]))
        
        input_ids = tuple([explicit_symptoms, implicit_symptoms, disease])
        options = tuple([explicit_options, implicit_options, 1])
        option_types = tuple([explicit_option_types, implicit_option_types, "binary"])

        return input_ids, options, option_types, differentials, differentials_probs

    def __getitem__(self, index):
        # input_ids = self.data_list[index].strip().split('\t')
        # input_ids = self.data_list[index].split('\t')
        # input_ids = tuple([
        #     [int(token_id) for token_id in input_ids[0].split()],
        #     [int(token_id) for token_id in input_ids[1].split()],
        #     int(input_ids[2])
        #     ])
        input_ids, options, option_types, differentials, differentials_probs = self._parse_data(self.data_list[index])

        pred_num = len(input_ids[1])

        sym_mask = {}
        symlabels = []
        deweight = []

        imp_sym_list = input_ids[1]
        imp_options_list = options[1]
        imp_option_types_list = option_types[1]
        # sequence shuffle
        if not self.no_sequence_shuffle:
            # random.shuffle(imp_sym_list)
            imp_sym_list, imp_options_list, imp_option_types_list = shuffle(imp_sym_list, imp_options_list, imp_option_types_list)
        lr = input_ids[0]  + imp_sym_list
        lr_options_list = options[0] + imp_options_list
        lr_option_types_list = option_types[0] + imp_option_types_list
        sym_infer_num = len(input_ids[1])
        if len(input_ids[0]) == 0:
            sym_infer_num -= 1
            imp_sym_list = imp_sym_list[1:]
            imp_options_list = imp_options_list[1:]
            imp_option_types_list = imp_option_types_list[1:]

        tokenids = [self.tokenizer.cls_token_id] + lr 
        ans_type_idx = [1] + lr_options_list
        encoded_option_types = ["binary"] + lr_option_types_list
        symlen = (sym_infer_num,len(lr))
        startpos = len(tokenids)
        tokenids += [self.tokenizer.sep_token_id] * (sym_infer_num+1)
        ans_type_idx += [0] * (sym_infer_num + 1)
        encoded_option_types += ["binary"] * (sym_infer_num + 1)

        sym_tag_offset = len(deweight)
        depos = []
        # synchronous prediction
        sym_mask["part_1"] = []
        for i in range(sym_infer_num):
            if self.no_synchronous_learning:
                x = [lr[-sym_infer_num+i]]
            else:
                x = lr[-sym_infer_num+i:]
            depos.extend([startpos+i]*len(x))
            sym_tag_list = [self.tokenizer.id_to_symptomid[y] for y in x]
            mask_list = [0]*len(self.tokenizer.vocab)
            for sym_type in sym_tag_list:
                mask_list[sym_type] = 1
            for sym_type in sym_tag_list:
                ml = mask_list.copy()
                ml[sym_type] = 0
                sym_mask["part_1"].append(ml)
            symlabels.extend(sym_tag_list)
            deweight.extend([1/len(x)]*len(x))

        sep_pos = startpos+sym_infer_num
        
        # predict the end of symptom inquiry
        symlabels.append(self.tokenizer.sep_token_id)
        depos.append(startpos+sym_infer_num)
        sym_mask["part_2"] = [0]*len(self.tokenizer.vocab)
        deweight.append(1)

        # autoencoding for symptom attention framework, which is not used in Diaformer 
        # random.shuffle(lr)
        lr, lr_options_list, lr_option_types_list = shuffle(lr, lr_options_list, lr_option_types_list)
        # autoencoding:  80% mask as BERT, 10% random token, 10% original token
        rd =  random.random()
        encoder_token = self.tokenizer.dis_pad_token_id
        encoder_token_option = 0
        encoder_token_option_type = "binary"
        if rd < 0.1:
            encoder_token = random.randint(8,len(self.tokenizer.id_to_symptomid)-1)
            encoder_token_option = self.tokenizer.id_to_default_options[encoder_token]
            encoder_token_option_type = self.tokenizer.id_to_option_type[encoder_token]
            if encoder_token_option_type == "multi-label":
                # in meta data default values are not lists for multi-label
                encoder_token_option = [encoder_token_option]
        elif rd < 0.2:
            encoder_token = lr[-1]
            encoder_token_option = lr_options_list[-1]
            encoder_token_option_type = lr_option_types_list[-1]
        tokenids += [encoder_token]
        ans_type_idx += [encoder_token_option]
        encoded_option_types += [encoder_token_option_type]
        
        encoder_labels = lr[-1]
        encoder_pos = len(tokenids)-1

        # repeated sequence
        if self.no_repeated_sequence:
            repeat_num = 0
        else:
            repeat_num = 4

        while repeat_num*(len(lr)+sym_infer_num-1) + len(tokenids) + 3 > self.max_len:
            repeat_num -= 1
        repeat_num_list = repeat_num
        # +2 of prediction of end and prediction of  encoder
        startpos_repeat = startpos + sym_infer_num + 2
        sym_mask["part_3"] = []
        for i in range(repeat_num):
            if sym_infer_num < 2:
                continue
            if not self.no_sequence_shuffle:
                # random.shuffle(imp_sym_list)
                imp_sym_list, imp_options_list, imp_option_types_list = shuffle(imp_sym_list, imp_options_list, imp_option_types_list)
            # imp_sym_list = imp_sym_list[1:]+[imp_sym_list[0]]
            tokenids.extend(imp_sym_list[:-1])
            ans_type_idx.extend(imp_options_list[:-1])
            encoded_option_types.extend(imp_option_types_list[:-1])
            tokenids += [self.tokenizer.sep_token_id] * (sym_infer_num-1)
            ans_type_idx += [0] * (sym_infer_num - 1)
            encoded_option_types += ["binary"] * (sym_infer_num - 1)
            # increase the position for the previous new AR token
            startpos_repeat += (len(imp_sym_list)-1)
            # synchronous prediction of repeated sequences
            for j in range(sym_infer_num-1):
                if self.no_synchronous_learning:
                    x = [imp_sym_list[-sym_infer_num+j+1]]
                else:
                    x = imp_sym_list[-sym_infer_num+j+1:]
                depos.extend([startpos_repeat+j]*len(x))

                sym_tag_list = [self.tokenizer.id_to_symptomid[y] for y in x]
                mask_list = [0]*len(self.tokenizer.vocab)
                for sym_type in sym_tag_list:
                    mask_list[sym_type] = 1
                for sym_type in sym_tag_list:
                    ml = mask_list.copy()
                    ml[sym_type] = 0
                    sym_mask["part_3"].append(ml)
                symlabels.extend(sym_tag_list)

                deweight.extend([1/(len(x)*(repeat_num+1))]*len(x))

            startpos_repeat += (sym_infer_num-1)
        
        if repeat_num > 0:
            if self.no_synchronous_learning:
                for i in range(sym_tag_offset+1,sym_infer_num+sym_tag_offset):
                    if deweight[i] != 0:
                        deweight[i]/=(repeat_num+1)
            else:
                for i in range(sym_infer_num+sym_tag_offset,sym_tag_offset+int((1+sym_infer_num)*sym_infer_num/2+1)):
                    deweight[i]/=(repeat_num+1)

        tokenids_list = tokenids

        # 0 none 1 imp 2 exp
        sym_type_idx = [1]*len(tokenids)
        sym_type_idx[1:len(input_ids[0])+1] = [2]*len(input_ids[0])
        # 0 none 1 true 2 false
        # ans_type_idx = [1]*len(tokenids)
        
        sym_type_idx[0] = 0
        # ans_type_idx[0] = 0
        for index,idx in enumerate(tokenids):
            if idx == self.tokenizer.sep_token_id:
                sym_type_idx[index] = 0
                # ans_type_idx[index] = 0

            # if idx >= len(self.tokenizer.vocab):
                # ans_type_idx[index] = 2
        sym_type_list = sym_type_idx
        ans_type_list = ans_type_idx
       
        dislabels = input_ids[2]
        
        decoder_pos = depos

        return {
            "symlen": symlen, 
            "pred_num": pred_num,
            "sym_mask": sym_mask, 
            "sep_pos": sep_pos,
            "symlabels": symlabels, 
            "deweight": deweight, 
            "encoder_labels": encoder_labels, 
            "encoder_pos": encoder_pos, 
            "repeat_num_list": repeat_num_list, 
            "tokenids_list": tokenids_list, 
            "sym_type_list": sym_type_list, 
            "ans_type_list": ans_type_list, 
            "dislabels": dislabels, 
            "decoder_pos": decoder_pos,
            "encoded_option_types": encoded_option_types,
            "differentials": differentials,
            "differentials_probs": differentials_probs,
            }

    def __len__(self):
        return len(self.data_list)