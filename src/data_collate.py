from transformers import DataCollatorForTokenClassification, PreTrainedTokenizerBase, DataCollatorForSeq2Seq
from typing import Union, Optional
import random
import torch
import numpy as np

class DataCollatorForMSALM(DataCollatorForTokenClassification):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    mlm_probability: float = 0.15
    return_tensors: str = "pt"


    def torch_call(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        no_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]
        
        batch_size = len(no_labels_features)
        max_alignments = max(len(msa["input_ids"]) for msa in no_labels_features)
        max_seqlen = max(len(msa["input_ids"][0]) for msa in no_labels_features)
        
        batch = {}

        batch["input_ids"] = torch.empty(
            (
                batch_size,
                max_alignments,
                max_seqlen,
            ),
            dtype=torch.int64,
        )
        batch["input_ids"].fill_(self.tokenizer.pad_token_id)
        
        if "attention_mask" in no_labels_features[0]:
            batch["attention_mask"] = torch.empty(
                (
                    batch_size,
                    max_alignments,
                    max_seqlen,
                ),
                dtype=torch.int64,
            )
            batch["attention_mask"].fill_(0)
            
        if "token_type_ids" in no_labels_features[0]:
            batch["token_type_ids"] = torch.empty(
                (
                    batch_size,
                    max_alignments,
                    max_seqlen,
                ),
                dtype=torch.int64,
            )
            batch["token_type_ids"].fill_(self.tokenizer.pad_token_type_id)
        
        test_bool = True
        if labels is None:
            test_bool = False
            labels = []
            label_name = 'labels'
        for i, msa in enumerate(no_labels_features):
            msa_seqlens = set(len(seq) for seq in msa["input_ids"])
            if not len(msa_seqlens) == 1:
                raise RuntimeError(
                    "Received unaligned sequences for input to MSA, all sequence "
                    "lengths must be equal."
                )
            if not test_bool:
                idx = random.choice(list(range(len(msa['input_ids'])))) ## idx to mask
                seq_len = len(msa["input_ids"][0])
                labels.append(msa['input_ids'][idx][:])
                for j in range(2,seq_len-1):
                    msa["input_ids"][idx][j] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
            for feat in batch:
                batch[feat][i, : len(msa[feat]), : len(msa[feat][0])] = torch.Tensor(msa[feat])


        if labels is None:
            return batch

        if not test_bool:
            sequence_length = batch["input_ids"].shape[2]
        else:
            sequence_length = max(len(label) for label in labels)
            sequence_length = max(sequence_length, batch["input_ids"].shape[2])
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        if padding_side == "right":
            batch[label_name] = [
                to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in labels
            ]

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)
        return batch


class DataCollatorForSeq2SeqNMT(DataCollatorForSeq2Seq):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        for feat in ["labels", "langs"]:
            labels = [feature[feat] for feature in features] if feat in features[0].keys() else None
            # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
            # same length to return tensors.
            if labels is not None:
                max_label_length = max(len(l) for l in labels)
                if self.pad_to_multiple_of is not None:
                    max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )

                padding_side = self.tokenizer.padding_side
                for feature in features:
                    if feat == "labels":
                        remainder = [self.label_pad_token_id] * (max_label_length - len(feature[feat]))
                    else: ## Pad with 0
                        remainder = [0] * (max_label_length - len(feature[feat]))
                    if isinstance(feature[feat], list):
                        feature[feat] = (
                            feature[feat] + remainder if padding_side == "right" else remainder + feature[feat]
                        )
                    elif padding_side == "right":
                        feature[feat] = np.concatenate([feature[feat], remainder]).astype(np.int64)
                    else:
                        feature[feat] = np.concatenate([remainder, feature[feat]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids
        
        del features["attention_mask"]
        return features