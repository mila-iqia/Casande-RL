# copied from dxy dataset's preprocess.py
import pickle
from tqdm import tqdm
data_set = pickle.load(open('symcat_200/train_valid_data.pkl', 'rb'))
# consturct goal_set.p
pickle.dump(data_set,open('{path to goal set}', 'wb'))  # anonymized
data_set.keys()
print(f'{data_set.keys()}:{[len(x) for x in data_set.values()]}')
sym_set = set()
dis_set = set()
# label_set = set()
for items in tqdm((data_set['train'],data_set['test'])):
    for item in items:
        dis_set.add(item['disease_tag'])
        for key,val in item['goal']['explicit_inform_slots'].items():
            sym_set.add(key)
            # label_set.add(val)
        for key,val in item['goal']['implicit_inform_slots'].items():
            sym_set.add(key)
            # label_set.add(val)

# build a vocabulary for the dataset
art_symbols = ['[PAD]','[PAD2]','[UNK]','[SEP]','[CLS]','[MASK]','[true]','[false]']
print(list(sym_set))
print(list(dis_set))
with open('{path to vocab}','w') as f:   # anonymized
    f.write('\n'.join(art_symbols) +'\n')
    f.write('\n'.join(list(sym_set)) +'\n')
    f.write('\n')
    f.write('\n'.join(list(dis_set)))