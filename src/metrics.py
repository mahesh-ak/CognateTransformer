import numpy as np
from lingrex.reconstruct import eval_by_bcubes, eval_by_dist

delim = '|'

def clean_decodes(x):
    x = x.split(delim)
    x_new = []
    for tok in x:
        if tok == '[SEP]':
            break
        if tok in ['[CLS]', '-'] or '[' in tok:
            continue

        x_new += tok.split()
    if len(x_new) == 0:
        x_new = ['_']
    x_new = ''.join(x_new)
    x_new = ' '.join(list(x_new))
    return x_new

def compute_metrics(eval_preds,tokenizer):
    preds, labels = eval_preds
    
    preds = np.argmax(preds, axis=-1)
    decoded_preds = np.array(tokenizer.batch_decode(preds, skip_special_tokens=False))
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = np.array(tokenizer.batch_decode(labels, skip_special_tokens=False))
    
    clean_decodes_v = np.vectorize(clean_decodes)
    decoded_preds = clean_decodes_v(decoded_preds)
    decoded_labels = clean_decodes_v(decoded_labels)


    
    result = {'Avg ED': eval_by_dist([[x.split(), y.split()] for x,y in zip(p,l)]), 
              'Avg NED':  eval_by_dist([[x.split(), y.split()] for x,y in zip(p,l)], normalized= True),
              'B^3 F1' : eval_by_bcubes([[x.split(), y.split()] for x,y in zip(decoded_preds, decoded_labels)])}
    
    result = {key: round(value,4) for key, value in result.items()}
    
    return result

def clean_decodes_sig(x):
    x = x.split(delim)
    x_new = []
    lng = ''
    for tok in x:
        if tok == '[SEP]':
            break
        if tok in ['[CLS]', '-']:
            continue
        if '[' in tok:
            lng = tok
            continue
        x_new += tok.split()
    if len(x_new) == 0:
        x_new = ['_']
    x_new = ' '.join(x_new)
    return x_new, lng


def compute_metrics_SIGTYP(eval_preds, tokenizer):
    preds, labels = eval_preds
    
    preds = np.argmax(preds, axis=-1)
    decoded_preds = np.array(tokenizer.batch_decode(preds, skip_special_tokens=False))
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = np.array(tokenizer.batch_decode(labels, skip_special_tokens=False))
    
    decoded_preds = [clean_decodes_sig(x) for x in decoded_preds]
    decoded_labels = [clean_decodes_sig(x) for x in decoded_labels]
    
    languages = {}
    
    for x,y in zip(decoded_preds, decoded_labels):
        lng = x[1]
        if lng not in languages:
            languages[lng] = {'preds':[], 'labels':[]}
        languages[x[1]]['preds'].append(x[0])
        languages[x[1]]['labels'].append(y[0])

    
    result = np.array([0.0, 0.0, 0.0])
    for lng, D in languages.items():
        p = np.array(D['preds'])
        l = np.array(D['labels'])
       
        languages[lng]['res'] = np.array([ eval_by_dist([[x.split(), y.split()] for x,y in zip(p,l)]), \
                               eval_by_dist([[x.split(), y.split()] for x,y in zip(p,l)], normalized= True), \
                               eval_by_bcubes([[x.split(), y.split()] for x,y in zip(p,l)])])
        result += languages[lng]['res']

    result /= len(languages)
    result = {'Avg ED': result[0],
              'Avg NED': result[1],
              'B^3 F1' : result[2]}
    
    result = {key: round(value,4) for key, value in result.items()}
    result.update({lng: '\t'.join([str(x) for x in languages[lng]['res'].round(4)]) for lng in languages})

    return result

def compute_metrics_proto(eval_preds, tokenizer):
    preds, labels = eval_preds
    
    preds = np.argmax(preds, axis=-1)
    decoded_preds = np.array(tokenizer.batch_decode(preds, skip_special_tokens=False))
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = np.array(tokenizer.batch_decode(labels, skip_special_tokens=False))
    
    decoded_preds = [clean_decodes_sig(x) for x in decoded_preds]
    decoded_labels = [clean_decodes_sig(x) for x in decoded_labels]
    proto_map = {'Burmish':'ProtoBurmish', 'Purus':'ProtoPurus',
            'Lalo':'ProtoLalo', 'Bai':'ProtoBai', 'Karen':'ProtoKaren',
            'Romance':'Latin'}  
    languages = {}
    lngs = [val for key,val in proto_map.items()]
    count = 0
    for x,y in zip(decoded_preds, decoded_labels):
        lng = y[1]
        lng = lng.replace("[","").replace("]","")
        if lng not in lngs:
            count += 1
            continue
        if lng not in languages:
            languages[lng] = {'preds':[], 'labels':[], 'res': np.array([0,0,0])}
        languages[lng]['preds'].append(x[0])
        languages[lng]['labels'].append(y[0])

    result = np.array([0.0, 0.0, 0.0])

    
    for lng, D in languages.items():
        #if lng != '[LATIN-CORRECT]':
        #    continue
        p = np.array(D['preds'])
        l = np.array(D['labels'])
        

       
        languages[lng]['res'] = np.array([ eval_by_dist([[x.split(), y.split()] for x,y in zip(p,l)]), \
                               eval_by_dist([[x.split(), y.split()] for x,y in zip(p,l)], normalized= True), \
                               eval_by_bcubes([[x.split(), y.split()] for x,y in zip(p,l)])])
        result += languages[lng]['res']

    result /= len(languages)
    result = {'Avg ED': result[0],
              'Avg NED': result[1],
              'B^3 F1' : result[2]}
    
    result = {key: round(value,4) for key, value in result.items()}
    result.update({lng: '\t'.join([str(x) for x in languages[lng]['res'].round(4)]) for lng in languages})

    return result

def compute_metrics_nmt(eval_preds, raw_data, tokenizer):
    preds, labels = eval_preds
    
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = tokenizer.batch_decode(preds, skip_special_tokens=False)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=False)
    
    decoded_preds, decoded_labels = [], []
    for x,y in zip(preds, labels):
        decoded_preds.append(clean_decodes(x))
        decoded_labels.append(clean_decodes(y))
    
    lngs = []
    for row in raw_data['data']:
        lng = ''
        for key, val in row.items():
            if val == '?':
                lng = key
        lngs.append(lng)
    
    languages = {}
    valid = ['ProtoBurmish', 'ProtoPurus', 'ProtoLalo',\
             'ProtoBai', 'ProtoKaren', 'Latin']

    for x,y,lng in zip(decoded_preds, decoded_labels, lngs):
        if lng not in valid:
            continue
        if lng not in languages:
            languages[lng] = {'preds':[], 'labels':[], 'res': np.array([0,0,0])}
        languages[lng]['preds'].append(x)
        languages[lng]['labels'].append(y)

    result = np.array([0.0, 0.0, 0.0])
    
    for lng, D in languages.items():
        #if lng != '[LATIN-CORRECT]':
        #    continue
        p = np.array(D['preds'])
        l = np.array(D['labels'])
       
        corr = lambda x: ' '.join(x)
        languages[lng]['res'] = np.array([eval_by_dist([[x.split(), y.split()] for x,y in zip(p,l)]), \
                               eval_by_dist([[x.split(), y.split()] for x,y in zip(p,l)], normalized= True), \
                               eval_by_bcubes([[x.split(), y.split()] for x,y in zip(p,l)])])
        result += languages[lng]['res']

    result /= len(languages)
        
    result = {'Avg ED': result[0],
              'Avg NED': result[1],
              'B^3 F1' : result[2]}
    
    result = {key: round(value,4) for key, value in result.items()}
    result.update({lng: '\t'.join([str(x) for x in languages[lng]['res'].round(4)]) for lng in languages})

    return result