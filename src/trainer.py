from .data_load import *
from .data_tokenize import *
from .data_collate import *
from .modelling_cogtran import MSATConfig, MSATForLM
from .modelling_nmt import NMTConfig, NMT
from .charactertokenizer.charactertokenizer import CharacterTokenizer
from .metrics import *
from transformers import TrainingArguments, Trainer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset, DatasetDict
import os
import numpy as np

def train_evaluate_reflex():
    ######## batch -> 64, epochs -> 24
    batch_size = 64
    embed_dim = 256
    hidden_dim = 512
    num_epochs = 32
    num_layers = 4
    num_attention_heads = 4
    lr_rate = 1e-3

    ## Load data
    ## Get vocab

    print('Loading cognate reflex data ...')
    data = load_reflex_data()
    ### LIMITING ####
    #limit = 3
    #for key, val in data.items():
    #    data[key] = {k: v[:limit] for k,v in val.items()}
    ################
    for i in range(2):
        embed_dim //= (i+1)
        hidden_dim //= (i+1)
        num_layers //= (i+1)
        num_attention_heads //= (i+1)
        lines = [f"test_prop\tED\tNED\tB3_F1"]
        lines1 = ['test_prop\tvalid\tlng\tED\tNED\tB3_F1']
        for prop in  ['0.10', '0.30', '0.50']:
            train = f"train-{prop}"
            test = f"test-{prop}"
            vocab = []
            max_length_per_msa = 0
            max_seq_length = 0
            dataset = {train: {'data':[]}, 'dev': {'data':[]}, test: {'data':[], 'labels': []}}
            dataset['dev'] = {'data': []}

    
            dataset[train]['data'], dataset['dev']['data'] = train_test_split(data[train]['data'], test_size= 0.08)
            dataset[test] = data[test]

            for row in dataset[train]['data']: 
                vocab, max_length_per_msa, max_seq_length =  get_vocab(row, vocab, max_length_per_msa, max_seq_length)

            dataset = DatasetDict({key: Dataset.from_dict(val) for key, val in dataset.items() if key in [train, test, 'dev']})

            ## Tokenize
            tokenizer = CharacterTokenizer(vocab, 64, delim= delim)
            tokenized_datasets = dataset.map(lambda x: tokenize(x, tokenizer=tokenizer), remove_columns=['data'])

            for key in tokenized_datasets:
                if 'test' in key:
                    tokenized_datasets[key] = tokenized_datasets[key].remove_columns(['solns'])

            config = MSATConfig(
                        vocab_size=tokenizer.vocab_size,
                        mask_token_id=tokenizer.mask_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        cls_token_id=tokenizer.cls_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        hidden_size=embed_dim,
                        num_hidden_layers=num_layers,
                        num_attention_heads=num_attention_heads,
                        intermediate_size=hidden_dim,
                        hidden_dropout_prob=0.1,
                        attention_probs_dropout_prob=0.1,
                        max_position_embeddings=32,
                        max_position_embeddings_per_msa=64,
                        layer_norm_eps=1e-12,
                    )

            model = MSATForLM(config=config)
            data_collator = DataCollatorForMSALM(tokenizer=tokenizer, padding= True)
    
            training_args = TrainingArguments(
                output_dir=f"models/msatransformermlm/",
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=lr_rate,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size*4,
                weight_decay=0,
                save_total_limit=1,
                num_train_epochs=num_epochs,
            #   fp16=True,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets[train],
                eval_dataset=tokenized_datasets['dev'],
                data_collator=data_collator,
                compute_metrics= lambda x: compute_metrics_SIGTYP(x, tokenizer= tokenizer),
            )

            evaler = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets[train],
                eval_dataset=tokenized_datasets[test],
                data_collator=data_collator,
                compute_metrics= lambda x: compute_metrics_SIGTYP(x, tokenizer= tokenizer),
            )

            trainer.train()
            eval_dict = evaler.evaluate()
            for key in eval_dict:
                if key in ['eval_Avg ED', 'eval_Avg NED', 'eval_B^3 F1']:
                    continue
                lines1.append(f"{prop}\t{0}\t{key.replace('eval_','')}\t{eval_dict[key]}")
            lines.append(f"{prop}\t{eval_dict['eval_Avg ED']}\t{eval_dict['eval_Avg NED']}\t{eval_dict['eval_B^3 F1']}")
        txt = '\n'.join(lines)
        txt1 = '\n'.join(lines1)
        name = {0: 'small', 1: 'tiny'}
        with open(f'results/reflex_{name[i]}.tsv','w') as fp:
            fp.write(txt)
        with open(f'results/reflex_{name[i]}_detailed.tsv','w') as fp:
            fp.write(txt1)
        print(txt)

def train_eval_proto_cog():
    ## batch_size -> 64, num_epochs -> 24
    batch_size = 64
    embed_dim = 256
    hidden_dim = 512
    num_epochs = 48
    num_layers = 4
    num_attention_heads = 4
    lr_rate = 1e-3

    ## Pretrain
    vocab = []
    max_length_per_msa = 0
    max_seq_length = 0
    dataset = {'pretrain_train': {'data':[]}, 'pretrain_dev': {'data':[]}}

    data = load_proto_data()
    ### LIMITING ####
    #limit = 10
    #for key, val in data.items():
    #    data[key] = {k: v[:limit] for k,v in val.items()}
    ################
    for row in data['pretrain']['data']: 
        vocab, max_length_per_msa, max_seq_length =  get_vocab(row, vocab, max_length_per_msa, max_seq_length)
    for row in data['finetune_train_0.1_1']['solns']:
        vocab, max_length_per_msa, max_seq_length =  get_vocab(row, vocab, max_length_per_msa, max_seq_length)

    for key, val in data.items():
        data[key] = pd.DataFrame(val)

    dataset['pretrain_train'], dataset['pretrain_dev'] = train_test_split(data['pretrain'], test_size= 0.10)


    dataset = DatasetDict({key: Dataset.from_pandas(val, preserve_index= False) for key, val in dataset.items()})

    tokenizer = CharacterTokenizer(vocab, 64, delim= delim)

    tokenized_datasets = dataset.map(lambda x: tokenize(x, tokenizer=tokenizer), remove_columns=['data'])

    config = MSATConfig(
                vocab_size=tokenizer.vocab_size,
                mask_token_id=tokenizer.mask_token_id,
                pad_token_id=tokenizer.pad_token_id,
                cls_token_id=tokenizer.cls_token_id,
                eos_token_id=tokenizer.eos_token_id,
                hidden_size=embed_dim,
                num_hidden_layers=num_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=hidden_dim,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=32,
                max_position_embeddings_per_msa=64,
                layer_norm_eps=1e-12,
            )

    model = MSATForLM(config=config)
    data_collator = DataCollatorForMSALM(tokenizer=tokenizer, padding= True)
    training_args = TrainingArguments(
        output_dir="models/msatransformerpretrain/",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size*4,
        weight_decay=0,
        save_total_limit=1,
        num_train_epochs=num_epochs,
    #   fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['pretrain_train'],
        eval_dataset=tokenized_datasets['pretrain_dev'],
        data_collator=data_collator,
        compute_metrics= lambda x: compute_metrics_proto(x, tokenizer=tokenizer),
    )

    trainer.train()
    tokenizer.save_pretrained('models/msatransformerpretrain/')
    lines = ['test_prop\tED\tNED\tB3_F1']
    lines1 = ['test_prop\tvalid\tlng\tED\tNED\tB3_F1']
    for prop in ['0.1','0.5','0.8']:
        res = np.array([0.0,0.0,0.0])
        for valid in range(1,1+valid_lim):
            ## batch_size -> 48
            ## num_epochs -> 9
            batch_size = 48
            num_epochs = 9
            lr_rate = 1e-3

            checkpoint = 'models/msatransformerpretrain/'
            dir_lst = os.listdir(checkpoint)
            for dir in dir_lst:
                if 'checkpoint' in dir:
                    checkpoint += dir
                    break
            model_fineT = MSATForLM.from_pretrained(checkpoint)
            tokenizer_fineT = CharacterTokenizer.from_pretrained('models/msatransformerpretrain/')
            data_collator_fineT = DataCollatorForMSALM(tokenizer= tokenizer_fineT, padding= True)

            dataset = {}
            for div in ['train', 'test', 'dev']:
                dataset[f"finetune_{div}_{prop}_{valid}"] = data[f"finetune_{div}_{prop}_{valid}"]
            
            dataset = DatasetDict({key: Dataset.from_pandas(val, preserve_index= False) for key, val in dataset.items()})
            tokenized_datasets = dataset.map(lambda x: tokenize(x, tokenizer= tokenizer_fineT), remove_columns=['data'])

            for key in tokenized_datasets:
                if 'finetune' in key:
                    tokenized_datasets[key] = tokenized_datasets[key].remove_columns(['solns'])

            training_args = TrainingArguments(
                output_dir="models/msatransformertune/",
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=lr_rate,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size*4,
                weight_decay=0,
                save_total_limit=1,
                num_train_epochs=num_epochs,
            #   fp16=True,
            )

            trainer = Trainer(
                model=model_fineT,
                args=training_args,
                train_dataset=tokenized_datasets[f'finetune_train_{prop}_{valid}'],
                eval_dataset=tokenized_datasets[f'finetune_dev_{prop}_{valid}'],
                data_collator=data_collator_fineT,
                compute_metrics=lambda x: compute_metrics_proto(x, tokenizer=tokenizer_fineT),
            )
            evaler = Trainer(
                model=model_fineT,
                args=training_args,
                train_dataset=tokenized_datasets[f'finetune_train_{prop}_{valid}'],
                eval_dataset=tokenized_datasets[f'finetune_test_{prop}_{valid}'],
                data_collator=data_collator_fineT,
                compute_metrics=lambda x: compute_metrics_proto(x, tokenizer=tokenizer_fineT),
            )

            trainer.train()
            eval_dict = evaler.evaluate()
            for key in eval_dict:
                if key in ['eval_Avg ED', 'eval_Avg NED', 'eval_B^3 F1']:
                    continue
                lines1.append(f"{prop}\t{valid}\t{key.replace('eval_','')}\t{eval_dict[key]}")
            res += np.array([eval_dict['eval_Avg ED'], eval_dict['eval_Avg NED'], eval_dict['eval_B^3 F1']])
        res = res/valid_lim
        lines.append(f"{prop}\t{round(res[0],4)}\t{round(res[1],4)}\t{round(res[2],4)}")

    txt = '\n'.join(lines)
    txt1 = '\n'.join(lines1)
    with open("results/proto_pretrained_cog.tsv",'w') as fp:
        fp.write(txt)
    with open("results/proto_pretrained_cog_detailed.tsv",'w') as fp:
        fp.write(txt1)
    print(txt)

    lines = ['test_prop\tED\tNED\tB3_F1']
    lines1 = ['test_prop\tvalid\tlng\tED\tNED\tB3_F1']
    for prop in ['0.1','0.5','0.8']:
        res = np.array([0.0,0.0,0.0])
        for valid in range(1,1+valid_lim):
            ## batch_size -> 48
            ## num_epochs -> 24
            batch_size = 48
            embed_dim = 128
            hidden_dim = 256
            num_epochs = 24
            num_layers = 2
            num_attention_heads = 2
            lr_rate = 1e-3
            vocab = []
            max_length_per_msa = 0
            max_seq_length = 0

            for row in data[f'finetune_train_{prop}_{valid}']['data']:
                vocab, max_length_per_msa, max_seq_length =  get_vocab(row, vocab, max_length_per_msa, max_seq_length)
            for row in data[f'finetune_train_{prop}_{valid}']['solns']:
                vocab, max_length_per_msa, max_seq_length =  get_vocab(row, vocab, max_length_per_msa, max_seq_length)

            tokenizer_fineT = CharacterTokenizer(vocab, 64, delim= delim)
            data_collator_fineT = DataCollatorForMSALM(tokenizer= tokenizer_fineT, padding= True)

            dataset = {}
            for div in ['train', 'test', 'dev']:
                dataset[f"finetune_{div}_{prop}_{valid}"] = data[f"finetune_{div}_{prop}_{valid}"]
            
            dataset = DatasetDict({key: Dataset.from_pandas(val, preserve_index= False) for key, val in dataset.items()})
            tokenized_datasets = dataset.map(lambda x: tokenize(x, tokenizer= tokenizer_fineT), remove_columns=['data'])

            for key in tokenized_datasets:
                if 'finetune' in key:
                    tokenized_datasets[key] = tokenized_datasets[key].remove_columns(['solns'])
            
            config = MSATConfig(
                        vocab_size=tokenizer_fineT.vocab_size,
                        mask_token_id=tokenizer_fineT.mask_token_id,
                        pad_token_id=tokenizer_fineT.pad_token_id,
                        cls_token_id=tokenizer_fineT.cls_token_id,
                        eos_token_id=tokenizer_fineT.eos_token_id,
                        hidden_size=embed_dim,
                        num_hidden_layers=num_layers,
                        num_attention_heads=num_attention_heads,
                        intermediate_size=hidden_dim,
                        hidden_dropout_prob=0.1,
                        attention_probs_dropout_prob=0.1,
                        max_position_embeddings=32,
                        max_position_embeddings_per_msa=64,
                        layer_norm_eps=1e-12,
                    )

            model_fineT = MSATForLM(config=config)

            training_args = TrainingArguments(
                output_dir="models/msatransformertune/",
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=lr_rate,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size*4,
                weight_decay=0,
                save_total_limit=1,
                num_train_epochs=num_epochs,
            #   fp16=True,
            )

            trainer = Trainer(
                model=model_fineT,
                args=training_args,
                train_dataset=tokenized_datasets[f'finetune_train_{prop}_{valid}'],
                eval_dataset=tokenized_datasets[f'finetune_dev_{prop}_{valid}'],
                data_collator=data_collator_fineT,
                compute_metrics=lambda x: compute_metrics_proto(x, tokenizer=tokenizer_fineT),
            )
            evaler = Trainer(
                model=model_fineT,
                args=training_args,
                train_dataset=tokenized_datasets[f'finetune_train_{prop}_{valid}'],
                eval_dataset=tokenized_datasets[f'finetune_test_{prop}_{valid}'],
                data_collator=data_collator_fineT,
                compute_metrics=lambda x: compute_metrics_proto(x, tokenizer=tokenizer_fineT),
            )

            trainer.train()
            eval_dict = evaler.evaluate()
            for key in eval_dict:
                if key in ['eval_Avg ED', 'eval_Avg NED', 'eval_B^3 F1']:
                    continue
                lines1.append(f"{prop}\t{valid}\t{key.replace('eval_','')}\t{eval_dict[key]}")
            res += np.array([eval_dict['eval_Avg ED'], eval_dict['eval_Avg NED'], eval_dict['eval_B^3 F1']])
        res = res/valid_lim
        lines.append(f"{prop}\t{round(res[0],4)}\t{round(res[1],4)}\t{round(res[2],4)}")

    txt = '\n'.join(lines)
    with open("results/proto_plain_cog.tsv",'w') as fp:
        fp.write(txt)
 
    txt1 = '\n'.join(lines1)
    with open("results/proto_plain_cog_detailed.tsv",'w') as fp:
        fp.write(txt1)
    print(txt)


def train_eval_proto_nmt():
    ## batch_size -> 16
    ## num_epochs -> 32
    batch_size = 16
    input_dim = 96
    lang_dim = 32
    hidden_size = 128
    intermediate_size = 256
    num_epochs = 32
    num_layers = 1
    lr_rate = 1e-3
    delim = '|'

    ## Load data
    ## Get vocab

    print('Loading cognate reflex data ...')
    data = load_nmt_data()
    ### LIMITING ####
    #limit = 3
    #for key, val in data.items():
    #    data[key] = {k: v[:limit] for k,v in val.items()}
    ################

    lines = ['test_prop\tED\tNED\tB3_F1']
    lines1 = ['test_prop\tvalid\tlng\tED\tNED\tB3_F1']
    for prop in ['0.1','0.5','0.8']:
        res = np.array([0.0,0.0,0.0])
        for valid in range(1,1+valid_lim):
            vocab = []
            lang_vocab = []
            max_length_per_msa = 0
            max_seq_length = 0
            for row in data[f'finetune_train_{prop}_{valid}']['data']:
                vocab, lang_vocab, max_length_per_msa, max_seq_length =  get_vocab_nmt(row, vocab, lang_vocab, max_length_per_msa, max_seq_length)
            for row in data[f'finetune_train_{prop}_{valid}']['solns']:
                vocab, lang_vocab, max_length_per_msa, max_seq_length =  get_vocab_nmt(row, vocab, lang_vocab, max_length_per_msa, max_seq_length)
            

            for key, val in data.items():
                data[key] = pd.DataFrame(val)
            dataset = {}
            for div in ['train', 'test', 'dev']:
                dataset[f"finetune_{div}_{prop}_{valid}"] = data[f"finetune_{div}_{prop}_{valid}"]
            dataset = DatasetDict({key: Dataset.from_pandas(val, preserve_index= False) for key, val in dataset.items()})

            tokenizer = CharacterTokenizer(vocab, 384, delim= delim)
            lang_tokenizer = CharacterTokenizer(lang_vocab, 96, delim= delim)

            tokenized_datasets = dataset.map(lambda x: tokenize_nmt(x, tokenizer=tokenizer, lang_tokenizer= lang_tokenizer), remove_columns=['data'])

            for key in tokenized_datasets:
                if 'finetune' in key:
                    tokenized_datasets[key] = tokenized_datasets[key].remove_columns(['solns'])

            config = NMTConfig(
                        vocab_size=tokenizer.vocab_size,
                        lang_vocab_size=lang_tokenizer.vocab_size,
                        mask_token_id=tokenizer.mask_token_id,
                        lang_pad_token_id=lang_tokenizer.pad_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        cls_token_id=tokenizer.cls_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        decoder_start_token_id=tokenizer.cls_token_id,
                        input_dim= input_dim,
                        lang_dim= lang_dim,
                        hidden_size=hidden_size,
                        num_hidden_layers=num_layers,
                        intermediate_size=intermediate_size,
                        hidden_dropout_prob=0.1,
                        attention_probs_dropout_prob=0.1,
                        max_position_embeddings=384,
                        max_position_embeddings_per_msa=96,
                        initializer_range=0.02,
                    )

            model_fineT = NMT(config=config)
            tokenizer_fineT = tokenizer
            data_collator_fineT = DataCollatorForSeq2SeqNMT(tokenizer=tokenizer_fineT, padding= True, model=model_fineT) 
            training_args = Seq2SeqTrainingArguments(
                output_dir="models/nmttorch/",
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=lr_rate,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size*4,
                weight_decay=0,
                save_total_limit=1,
                num_train_epochs=num_epochs,
                predict_with_generate= True,
            #   fp16=True,
            )

            trainer = Seq2SeqTrainer(
                model=model_fineT,
                args=training_args,
                train_dataset=tokenized_datasets[f'finetune_train_{prop}_{valid}'],
                eval_dataset=tokenized_datasets[f'finetune_dev_{prop}_{valid}'],
                data_collator=data_collator_fineT,
                compute_metrics=lambda x: compute_metrics_nmt(x, dataset[f'finetune_dev_{prop}_{valid}'], tokenizer_fineT),
            )
            
            evaler = Seq2SeqTrainer(
                model=model_fineT,
                args=training_args,
                train_dataset=tokenized_datasets[f'finetune_train_{prop}_{valid}'],
                eval_dataset=tokenized_datasets[f'finetune_test_{prop}_{valid}'],
                data_collator=data_collator_fineT,
                compute_metrics=lambda x: compute_metrics_nmt(x, dataset[f'finetune_test_{prop}_{valid}'], tokenizer_fineT),
            )

            trainer.train()
            eval_dict = evaler.evaluate()
            for key in eval_dict:
                if key in ['eval_Avg ED', 'eval_Avg NED', 'eval_B^3 F1']:
                    continue
                lines1.append(f"{prop}\t{valid}\t{key.replace('eval_','')}\t{eval_dict[key]}")
            res += np.array([eval_dict['eval_Avg ED'], eval_dict['eval_Avg NED'], eval_dict['eval_B^3 F1']])
        res = res/valid_lim
        lines.append(f"{prop}\t{round(res[0],4)}\t{round(res[1],4)}\t{round(res[2],4)}")

    txt = '\n'.join(lines)
    with open("results/proto_nmt.tsv",'w') as fp:
        fp.write(txt)
    txt1 = '\n'.join(lines1)
    with open("results/proto_nmt_detailed.tsv",'w') as fp:
        fp.write(txt1)
    print(txt)

