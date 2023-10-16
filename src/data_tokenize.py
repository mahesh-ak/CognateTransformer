import torch

delim = '|'

def tokenize(row, tokenizer, return_tensors=None):

    result = {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}
    
    for lng, algn in row['data'].items():
        if algn is None or len(algn.split(delim)) == 1:
            continue
        if '?' in algn:
            algn = algn.replace('?', tokenizer.mask_token)
        tokenize_lng = tokenizer(algn, return_tensors=return_tensors)
        for key in result:
            result[key].append(tokenize_lng[key])
        
    if return_tensors == 'pt':
        for key in result:
            result[key] = torch.stack(result[key])

    if 'solns' in row:
        for lng, algn in row['solns'].items():
            if algn is not None:
                result['labels'] = tokenizer(algn, return_tensors=return_tensors)['input_ids']

    
    return result


def tokenize_nmt(row, tokenizer, lang_tokenizer, return_tensors=None):
    
    row_txt = []
    row_lang = []
    for lng, algn in row['data'].items():
        if algn is None or len(algn.split(delim)) == 0 or algn == '?':
            continue
        row_txt.append(algn)
        row_lang.append(delim.join([lng]*len(algn.split(delim))))
    row_txt = f"{delim}{tokenizer.sep_token}{delim}".join(row_txt)
    row_lang = f"{delim}{lang_tokenizer.sep_token}{delim}".join(row_lang)
    result = tokenizer(row_txt, return_tensors=return_tensors)
    result['langs'] = lang_tokenizer(row_lang, return_tensors=return_tensors)['input_ids']
    if 'solns' in row:
        for lng, algn in row['solns'].items():
            if algn is not None:
                result['labels'] = tokenizer(algn, return_tensors=return_tensors)['input_ids'][1:]

    
    return result

