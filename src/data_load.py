import os
import pandas as pd
from tqdm import tqdm
import lingpy
from sklearn.model_selection import train_test_split
import json

delim = '|'
valid_lim = 10

def get_dict(row, is_test= False):
    ret_D = {}
    mult = []
    mult_langs = []
    for key, val in row.items():
        if key == 'COGID' or val in ['-']:
            continue
        if val in ['?']:
            if '_sol' in key:
                ret_D[key.replace('_sol','')] = val
            else:
                ret_D[key] = val
        else:
            if '_sol' in key:
                ret_D[key.replace('_sol','')] = val
            else:
                ret_D[key] = val

    for key, val in ret_D.items():
        if val not in ['?']:
            mult.append(val.split())
            mult_langs.append(key)
    mult = lingpy.Multiple(mult)
    mult.prog_align()
    mult = mult.alm_matrix
    labels = False
    if len(mult) == 1: ## labels
        labels = True
    ## Trim Alignments
    
    seq_len = len(mult[0])
    mult_t = {lng: [] for lng in mult_langs}
    buff = {lng: '' for lng in mult_langs}
    for i in range(seq_len):
        num_fills = 0
        lngs_trim = []
        inds_trim = []
        for j, algn_lng in enumerate(zip(mult, mult_langs)):
            algn, lng = algn_lng
            if algn[i] != '-':
                num_fills += 1
                lngs_trim.append(lng)
                inds_trim.append(j)
        if num_fills == 1 and not labels and not is_test: ## trim
            lng_trim = lngs_trim[0]
            indx_trim = inds_trim[0]
            if buff[lng_trim] != '':
                buff[lng_trim] += ' ' + mult[indx_trim][i]
            else:
                buff[lng_trim] = mult[indx_trim][i]
        else:
            for lng, algn in zip(mult_langs,mult):
                if buff[lng] == '':
                    mult_t[lng] += [algn[i]]
                else:
                    if algn[i] != '-':
                        mult_t[lng] += [buff[lng] + ' ' + algn[i]]
                    else:
                        mult_t[lng] += [buff[lng]]
                buff[lng] = ''
    for lng in buff:
        if buff[lng] != '':
            if len(mult_t[lng]) == 0:
                mult_t[lng].append(buff[lng])
            elif mult_t[lng][-1] == '-':
                mult_t[lng][-1] = buff[lng]
            else:
                mult_t[lng][-1] += ' ' + buff[lng]
            buff[lng] = ''
    mult = [mult_t[lng] for lng in mult_langs]       
    seq_len = len(mult[0])

    for lng, algn in zip(mult_langs,mult):
        ret_D[lng] = algn
    
    aligned = {lng: [f"[{lng}]"] for lng in ret_D}
    for key, val in ret_D.items():
        if val == '?':
            val = ['?']*seq_len
        aligned[key] += val
        aligned[key] = delim.join(aligned[key])
    return aligned

def load_reflex_data():
    data_paths = ["data/reflex-prediction/data-surprise/", "data/reflex-prediction/data/"]
    files = {'train-0.10':[], 'train-0.30': [], 'train-0.50': [], 
            'test-0.10':[], 'test-0.30':[], 'test-0.50': []}

    for data_path in data_paths:
        dirs = os.listdir(data_path)
        for fd in dirs:
            if '.' in fd:
                continue
            sub_path = os.path.join(data_path,fd)
            subdirs = os.listdir(sub_path)
            for f in subdirs:
                if 'training' in f and 'surprise' in data_path:
                    prop = f.replace('training-','').replace('.tsv','')
                    files[f"train-{prop}"].append(os.path.join(sub_path,f))
                if 'cognates' in f and 'surprise' not in data_path:
                    for prop in ['0.10','0.30','0.50']:
                        files[f"train-{prop}"].append(os.path.join(sub_path,f))
                if 'test' in f and 'surprise' in data_path:
                    prop = f.replace('test-','').replace('.tsv','')
                    sol_f = f"solutions-{prop}.tsv"
                    files[f"test-{prop}"].append((os.path.join(sub_path,f), os.path.join(sub_path,sol_f)))

    data =  {'train-0.10': {'data':[]},
            'train-0.30': {'data':[]},
            'train-0.50': {'data':[]},
            'test-0.10': {'data':[], 'solns':[]}, 
            'test-0.30': {'data':[], 'solns':[]},
            'test-0.50': {'data':[], 'solns':[]}}

    for key, f_list in tqdm(files.items()):
        if 'train' in key:
            for f in f_list:
                df = pd.read_csv(f, sep='\t')
                df.fillna('-', inplace=True)
                data[key]['data'] += df.apply(lambda x: get_dict(x), axis=1).tolist()
            
        else: ## test
            for f in f_list:
                df_test = pd.read_csv(f[0], sep='\t')
                df_test.fillna('-', inplace= True)
                df_sol = pd.read_csv(f[1], sep='\t')
                df_sol.fillna('-', inplace= True)
                df_sol.rename(columns={col: col+'_sol' for col in list(df_sol.columns) if col != 'COGID'}, inplace=True)
                df_test = df_test.merge(df_sol, on= 'COGID')
                df_sol = df_test.drop(columns=[col for col in list(df_test.columns) if '_sol' not in col and col !='COGID'])
                df_test = df_test.drop(columns=[col for col in list(df_test.columns) if '_sol' in col and col !='COGID'])
                data[key]['data'] += df_test.apply(lambda x: get_dict(x, is_test= True), axis=1).tolist()
                data[key]['solns'] += df_sol.apply(lambda x: get_dict(x, is_test= True), axis=1).tolist()
    
    return data



def get_vocab(row, vocab, max_length_per_msa, max_seq_length):
    
    max_seq_length = max(len(row), max_seq_length)
    for lng, algn in row.items():
        max_length_per_msa = max(len(algn) + 2, max_length_per_msa)
        if delim == '':    
            for char in row[lng]:
                if char not in vocab:
                    vocab.append(char)
        else:
            for char in row[lng].split(delim):
                if char not in vocab:
                    vocab.append(char)
    return vocab, max_length_per_msa, max_seq_length

def get_vocab_nmt(row, vocab, lang_vocab, max_length_per_msa, max_seq_length):
    
    max_seq_length = max(len(row), max_seq_length)
    for lng, algn in row.items():
        max_length_per_msa = max(len(algn) + 2, max_length_per_msa)
        if lng not in lang_vocab:
            lang_vocab.append(lng)
        if delim == '':    
            for char in row[lng]:
                if char not in vocab:
                    vocab.append(char)
        else:
            for char in row[lng].split(delim):
                if char not in vocab:
                    vocab.append(char)
    return vocab, lang_vocab, max_length_per_msa, max_seq_length

def get_dict_proto(row, form= None, sol_col= None, is_test= False):
    ret_D = {}
    mult = []
    mult_langs = []
    for key, val in row.items():
        if key == 'COGID' or val in ['-']:
            continue
        ret_D[key] = val
    
    for key, val in ret_D.items():
        if not (is_test and key == sol_col):
            if form is str:
                mult.append(val)
            else:
                mult.append(val.split())
            mult_langs.append(key)
         
    mult = lingpy.Multiple(mult)
    mult.prog_align()
    mult = mult.alm_matrix
    labels = False
    if len(mult) == 1: ## labels
        labels = True
    ## Trim Alignments
    
    seq_len = len(mult[0])
    mult_t = {lng: [] for lng in mult_langs}
    buff = {lng: '' for lng in mult_langs}
    for i in range(seq_len):
        num_fills = 0
        lngs_trim = []
        inds_trim = []
        for j, algn_lng in enumerate(zip(mult, mult_langs)):
            algn, lng = algn_lng
            if algn[i] != '-':
                num_fills += 1
                lngs_trim.append(lng)
                inds_trim.append(j)
        if num_fills == 1 and not labels and not is_test: ## trim
            lng_trim = lngs_trim[0]
            indx_trim = inds_trim[0]
            if buff[lng_trim] != '':
                buff[lng_trim] += ' ' + mult[indx_trim][i]
            else:
                buff[lng_trim] = mult[indx_trim][i]
        else:
            for lng, algn in zip(mult_langs,mult):
                if buff[lng] == '':
                    mult_t[lng] += [algn[i]]
                else:
                    if algn[i] != '-':
                        mult_t[lng] += [buff[lng] + ' ' + algn[i]]
                    else:
                        mult_t[lng] += [buff[lng]]
                buff[lng] = ''
    for lng in buff:
        if buff[lng] != '':
            if len(mult_t[lng]) == 0:
                mult_t[lng].append(buff[lng])
            elif mult_t[lng][-1] == '-':
                mult_t[lng][-1] = buff[lng]
            else:
                mult_t[lng][-1] += ' ' + buff[lng]
            buff[lng] = ''
    mult = [mult_t[lng] for lng in mult_langs]       
    seq_len = len(mult[0])

    for lng, algn in zip(mult_langs,mult):
        ret_D[lng] = algn
    
    aligned = {lng: [f"[{lng}]"] for lng in ret_D}
    soln = {}
    for key, val in ret_D.items():
        if key == sol_col:
            if is_test:
                mult_sol = lingpy.Multiple([val])
                mult_sol.prog_align()
                mult_sol = mult_sol.alm_matrix
                soln[key] = [f"[{key}]"]+mult_sol[0]
            else:
                soln[key] = [f"[{key}]"]+val

            soln[key] = delim.join(soln[key])
            val = ['?']*seq_len

                                   
        aligned[key] += val
        aligned[key] = delim.join(aligned[key])
    
    if sol_col is None:
        return aligned
    else:
        return aligned, soln

def Merge(dict_lst):
        out = {}
        for D in dict_lst:
            out.update(D)
        return out

def load_proto_data():
    data_paths = ["data/reflex-prediction/data-surprise/", "data/reflex-prediction/data/"]
    files = []
    print("Loading proto reconstruction data...")
    for data_path in data_paths:
        dirs = os.listdir(data_path)
        for fd in dirs:
            if '.' in fd:
                continue
            sub_path = os.path.join(data_path,fd)
            subdirs = os.listdir(sub_path)
            for f in subdirs:
                if 'cognates' in f:
                    files.append(os.path.join(sub_path,f))


    data =  {'pretrain': {'data':[]}}
    for prop in ['0.1', '0.5', '0.8']:
        for valid in range(valid_lim):
            data[f"finetune_train_{prop}_{valid + 1}"] = {'data':[], 'solns':[]}
            data[f"finetune_test_{prop}_{valid + 1}"] = {'data':[], 'solns':[]}
            data[f"finetune_dev_{prop}_{valid + 1}"] = {'data':[], 'solns':[]}


    proto_map = {'Burmish':'ProtoBurmish', 'Purus':'ProtoPurus',
                'Lalo':'ProtoLalo', 'Bai':'ProtoBai', 'Karen':'ProtoKaren',
                'Romance':'Latin'}
    for f in tqdm(files):
        df = pd.read_csv(f, sep='\t')
        df.fillna('-', inplace=True)
        for lng, plng in proto_map.items():
            if plng in df.columns:
                df.drop(columns=[plng], inplace=True)
        data['pretrain']['data'] += df.apply(lambda x: get_dict_proto(x), axis=1).tolist()
    

    for prop in ['0.1','0.5', '0.8']:
    
        train_path = f"data/proto-reconstruction/data-{prop}/testlists/"
        test_path = f"data/proto-reconstruction/data-{prop}/testitems/"

        
        for file_num in range(1,valid_lim+1):
            dirs = os.listdir(train_path)
            for fd in tqdm(dirs):
                if '.' in fd:
                    continue
                df_ = {}
                f_str = "test-{0}".format(file_num)
    
                sub_path = os.path.join(test_path,fd)    
                f_pth = os.path.join(sub_path, f_str+'.json')
                f_json = json.load(open(f_pth,'r'))
                rows = []
                for line in f_json:
                    row = {proto_map[fd]: ' '.join(line[1])}
                    for word, lng in zip(line[2],line[3]):
                        row[lng] = ' '.join(word)
                    rows.append(row)
                df_['test'] = pd.DataFrame(rows).fillna('-')
    
                sub_path = os.path.join(train_path,fd)  
  
                f_pth = os.path.join(sub_path, f_str+'.tsv')
                df = pd.read_csv(f_pth, sep= '\t')
                df.drop(columns=['FORM', 'ID', 'ALIGNMENT', 'CONCEPT'], inplace=True)
                df['dict'] = df.apply(lambda x: {x['DOCULECT']: x['TOKENS']}, axis=1)
                df.drop(columns=['DOCULECT', 'TOKENS'], inplace=True)
                df = df.groupby('COGID').agg(Merge).reset_index()
                rows = df['dict'].tolist()
                df = pd.DataFrame(rows).fillna('-')
                df_['train'], df_['dev'] = train_test_split(df, test_size= 0.08)
    
                for div in ['train', 'dev', 'test']:
                    if div in ['dev', 'test']:
                        is_test = True
                    else:
                        is_test = False

                    dat = df_[div].apply(lambda x: get_dict_proto(x, sol_col= proto_map[fd], is_test= is_test), axis=1).tolist()
                    sol = [row[1] for row in dat]
                    dat = [row[0] for row in dat]
                    data[f"finetune_{div}_{prop}_{file_num}"]['data'] += dat
                    data[f"finetune_{div}_{prop}_{file_num}"]['solns'] += sol
                df = pd.concat([df, df_['test']])
                df.drop(columns=[proto_map[fd]], inplace=True)
                if prop == '0.1' and file_num == 1:
                    data['pretrain']['data'] += df.apply(lambda x: get_dict_proto(x), axis=1).tolist()

    return data

def get_dict_nmt(row, form= None, sol_col= None, is_test= False):
    ret_D = {}
    mult = []
    mult_langs = []
    for key, val in row.items():
        if key == 'COGID' or val in ['-']:
            continue
        ret_D[key] = val
    
    for key, val in ret_D.items():
        if not (is_test and key == sol_col):
            if form is str:
                mult.append(val)
            else:
                mult.append(val.split())
            mult_langs.append(key)
 
    for lng, algn in zip(mult_langs,mult):
        ret_D[lng] = algn
        
    aligned = {lng: [] for lng in ret_D}
    soln = {}
    for key, val in ret_D.items():
        if key == sol_col:
            if is_test:
            #    mult_sol = lingpy.Multiple([val])
            #    mult_sol.prog_align()
            #    mult_sol = mult_sol.alm_matrix
                soln[key] = []+ val.split()#mult_sol[0]
            else:
                soln[key] = []+val

            soln[key] = delim.join(soln[key])
            val = ['?'] #*seq_len

                                   
        aligned[key] += val
        aligned[key] = delim.join(aligned[key])
    
    if sol_col is None:
        return aligned
    else:
        return aligned, soln


def load_nmt_data():
    data =  {}

    for prop in ['0.1','0.5', '0.8']:
        for valid in range(1,1+valid_lim):
            data[f"finetune_train_{prop}_{valid}"] = {'data':[], 'solns':[]}
            data[f"finetune_test_{prop}_{valid}"] = {'data':[], 'solns':[]}
            data[f"finetune_dev_{prop}_{valid}"] = {'data':[], 'solns':[]}
    proto_map = {'Burmish':'ProtoBurmish', 'Purus':'ProtoPurus',
             'Lalo':'ProtoLalo', 'Bai':'ProtoBai', 'Karen':'ProtoKaren',
             'Romance':'Latin'}
    
    for prop in ['0.1','0.5', '0.8']:
    
        train_path = f"data/proto-reconstruction/data-{prop}/testlists/"
        test_path = f"data/proto-reconstruction/data-{prop}/testitems/"

        
        for file_num in range(1,1+valid_lim):
            dirs = os.listdir(train_path)
            for fd in tqdm(dirs):
                if '.' in fd:
                    continue
                df_ = {}
                f_str = "test-{0}".format(file_num)
    
                sub_path = os.path.join(test_path,fd)    
                f_pth = os.path.join(sub_path, f_str+'.json')
                f_json = json.load(open(f_pth,'r'))
                rows = []
                for line in f_json:
                    row = {proto_map[fd]: ' '.join(line[1])}
                    for word, lng in zip(line[2],line[3]):
                        row[lng] = ' '.join(word)
                    rows.append(row)
                df_['test'] = pd.DataFrame(rows).fillna('-')
    
                sub_path = os.path.join(train_path,fd)  
  
                f_pth = os.path.join(sub_path, f_str+'.tsv')
                df = pd.read_csv(f_pth, sep= '\t')
                df.drop(columns=['FORM', 'ID', 'ALIGNMENT', 'CONCEPT'], inplace=True)
                df['dict'] = df.apply(lambda x: {x['DOCULECT']: x['TOKENS']}, axis=1)
                df.drop(columns=['DOCULECT', 'TOKENS'], inplace=True)
                df = df.groupby('COGID').agg(Merge).reset_index()
                rows = df['dict'].tolist()
                df = pd.DataFrame(rows).fillna('-')
                df_['train'], df_['dev'] = train_test_split(df, test_size= 0.08)
    
                for div in ['train', 'dev', 'test']:
                    if div in ['dev', 'test']:
                        is_test = True
                    else:
                        is_test = False

                    dat = df_[div].apply(lambda x: get_dict_nmt(x, sol_col= proto_map[fd], is_test= is_test), axis=1).tolist()
                    sol = [row[1] for row in dat]
                    dat = [row[0] for row in dat]
                    data[f"finetune_{div}_{prop}_{file_num}"]['data'] += dat
                    data[f"finetune_{div}_{prop}_{file_num}"]['solns'] += sol
                df = pd.concat([df, df_['test']])
                df.drop(columns=[proto_map[fd]], inplace=True)
    return data