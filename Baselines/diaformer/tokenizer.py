import json

"""
Encoding that considers evidences' types, as CASANDE does. 
"""

def convert_evidence_data_type(meta_data):
    """Convert the data type in json files to the one used in supervised baselines.

    The main difference is that a new data type, "number", is introduced. It corresponds to the integer situation under
    "C" data type in json files.

    Parameter:
    ----------
    meta_data: dict
        the meta data of an evidence, as extracted from the json file
    
    Return:
    data_type: str
        the converted data type
    """
    if meta_data["data_type"] == "B":
        return "binary"
    if meta_data["data_type"] == "M":
        return "multi-label"
    if meta_data["data_type"] == "C":
        if isinstance(meta_data["possible-values"][0], str):
            return "multi-class"
        else:
            return "number"

class diaTokenizer():
    vocab = {}
    ids_to_tokens = {}
    special_tokens_id = range(0,8)
    disease_tokens_id = None
    # symptom_tokens_id = None
    symptom_to_false = {}
    id_to_symptomid = {}
    tokenid_to_diseaseid = {}
    disvocab = {}
    labels_to_diseases = {}
    class_weight = None

    def __init__(self, vocab_file, unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]",
                 mask_token="[MASK]",true_token="[true]",false_token="[false]",dis_pad_token='[PAD2]', evi_meta_path=None):
        with open(vocab_file, "r", encoding="utf-8") as reader:
            tokens = reader.read().split('\n\n')
            symlist = tokens[0].splitlines()
            dislist = tokens[1].splitlines()
        for index, token in enumerate(symlist):
            self.vocab[token] = index
            self.ids_to_tokens[index] = token
            index += 1
        for index, token in enumerate(dislist):
            self.disvocab[token] = index
            self.labels_to_diseases[index] = token
            index += 1
        self.unk_token_id = self.vocab[unk_token]
        self.sep_token_id = self.vocab[sep_token]
        self.pad_token_id = self.vocab[pad_token]
        self.cls_token_id = self.vocab[cls_token]
        self.mask_token_id = self.vocab[mask_token]
        self.true_token_id = self.vocab[true_token]
        self.false_token_id = self.vocab[false_token]
        self.dis_pad_token_id = self.vocab[dis_pad_token]
        self.disease_tokens_id =  range(len(self.vocab)-12,len(self.vocab))
        vocablen = len(self.vocab)
        for index in range(8,len(self.vocab)):
            false_id = index + vocablen - 8 # this index is larger than or equal to len(self.vocab), which is used as the indicator of false symptom
            self.symptom_to_false[index] = false_id
            # self.id_to_symptomid[false_id] = index
            self.id_to_symptomid[index] = index
        for index in range(8):
            self.id_to_symptomid[index] = index

        # read evidence meta data
        special_evidences = {unk_token, sep_token, pad_token, cls_token, mask_token, true_token, false_token, dis_pad_token}
        with open(evi_meta_path, "r") as file:
            evi_meta_data = json.load(file)
        self.id_to_default_options, self.id_to_option_type = {}, {}
        for evidence, index in self.vocab.items():
            if evidence in special_evidences:
                continue
            if evi_meta_data[evidence]["default_value"] in evi_meta_data[evidence]["possible-values"]:
                self.id_to_default_options[index] = evi_meta_data[evidence]["possible-values"].index(evi_meta_data[evidence]["default_value"])
            else:
                self.id_to_default_options[index] = 0
            self.id_to_option_type[index] = convert_evidence_data_type(evi_meta_data[evidence])
        self.num_option_indices = max([len(meta_data["possible-values"]) for meta_data in evi_meta_data.values()])

    def convert_token_to_id(self, token):
        if token not in self.vocab:
            return self.unk_token_id
        return self.vocab[token]
    
    def __len__(self):
        return len(self.vocab)
    

    def convert_id_to_token(self,id):
        if id in self.ids_to_tokens:
            return self.ids_to_tokens[id]
        return self.unk_token_id

    def convert_disease_to_label(self,disease):
        return self.disvocab[disease]
    
    def convert_label_to_disease(self,label):
        return self.labels_to_diseases[label]