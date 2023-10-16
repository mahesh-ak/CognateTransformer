from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from typing import Optional, Union
import copy

class NMTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`NMTModel`]. It is used to instantiate a MSA model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the NMT.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*):
            Vocabulary size of the MSA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`MSAModel`].
        mask_token_id (`int`, *optional*):
            The index of the mask token in the vocabulary. This must be included in the config because of the
            "mask-dropout" scaling trick, which will scale the inputs depending on the number of masked tokens.
        pad_token_id (`int`, *optional*):
            The index of the padding token in the vocabulary. This must be included in the config because certain parts
            of the ESM code use this instead of the attention mask.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 1026):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        emb_layer_norm_before (`bool`, *optional*):
            Whether to apply layer normalization after embeddings but before the main stem of the network.
        token_dropout (`bool`, defaults to `False`):
            When this is enabled, masked tokens are treated as if they had been dropped out by input dropout.
    """
    model_type = "nmt"

    def __init__(
        self,
        vocab_size=None,
        lang_vocab_size=None,
        mask_token_id=None,
        pad_token_id=None,
        lang_pad_token_id=None,
        cls_token_id=None,
        eos_token_id=None,
        decoder_start_token_id=None,
        input_dim= 96,
        lang_dim= 32,
        hidden_size=128,
        num_hidden_layers=1,
        intermediate_size=256,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=384,
        max_position_embeddings_per_msa=64,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_cache=True,
        emb_layer_norm_before=None,
        is_encoder_decoder=True,
        token_dropout=False,
        vocab_list=None,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, mask_token_id=mask_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.lang_vocab_size = lang_vocab_size
        self.cls_token_id = cls_token_id
        self.eos_token_id = eos_token_id
        self.lang_pad_token_id = lang_pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.input_dim = input_dim
        self.lang_dim = lang_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.max_position_embeddings_per_msa = max_position_embeddings_per_msa
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.emb_layer_norm_before = emb_layer_norm_before
        self.is_encoder_decoder = is_encoder_decoder
        self.token_dropout = token_dropout

        
class NMTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = NMTConfig
    base_model_prefix = "nmt"
    _no_split_modules = []

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    # Copied from transformers.models.t5.modeling_t5.T5PreTrainedModel._shift_right
    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id."
                "See T5 docs for more information."
            )

        # shift inputs to the right

        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

class Encoder(NMTPreTrainedModel):

    def __init__(self, input_embedding, config):
        super().__init__(config)
        self.config= config
        self.hidden_size = config.hidden_size
        self.embedding = input_embedding
        self.lang_size = config.lang_vocab_size
        self.lang_dim = config.lang_dim
        self.lang_embedding = nn.Embedding(self.lang_size, self.lang_dim)
        self.lang_embedding.padding_idx = config.lang_pad_token_id
        self.n_layers = config.num_hidden_layers
        self.dropout = config.hidden_dropout_prob
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers,
                            dropout=self.dropout, bidirectional=True, batch_first=True)
        
        self.init_weights()
        
    def forward(
        self, 
        input_ids: Optional[torch.LongTensor] = None,
        langs: Optional[torch.LongTensor] = None,
        hidden: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    )-> Union[tuple, dict]:
 
        embedded = self.embedding(input_ids)
        lang_embedded = self.lang_embedding(langs)
        outputs = torch.cat((embedded, lang_embedded), dim=-1)
        outputs, hidden = self.gru(outputs, hidden)
        output = (outputs[..., :self.hidden_size] + 
                   outputs[..., self.hidden_size:])

        
        return output, hidden

        
class AttnDecoderRNN(NMTPreTrainedModel):
    def __init__(self, input_embedding, config):
        super().__init__(config)
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.embed_dim = config.input_dim
        
        self.gru = nn.GRU(self.embed_dim, self.hidden_size, batch_first= True)
        
        self.max_length = config.max_position_embeddings
        self.intermediate_size = config.intermediate_size

        self.embedding = input_embedding
        self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.linear = nn.Linear(self.hidden_size, self.intermediate_size)
        
        self.dropout_p = config.attention_probs_dropout_prob
        self.dropout = nn.Dropout(self.dropout_p)
        
        self.out = nn.Linear(self.intermediate_size, config.vocab_size)
        
        self.init_weights()

    def forward(
        self, 
        decoder_input_ids: Optional[torch.LongTensor] = None,
        hidden: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    )-> Union[tuple, dict]:
        
        embedded = self.embedding(decoder_input_ids)
        
        hidden = hidden.unsqueeze(0)
        output, decoder_hidden = self.gru(embedded, hidden)
        
    
        attn_weights = F.softmax(
            torch.bmm(output, self.attn(encoder_outputs).transpose(1,2)), dim=-1)
    
        attn_applied = torch.bmm(attn_weights, encoder_outputs)

        
        output = output + attn_applied
        output = self.linear(output)

        output = F.relu(output)
        output = self.dropout(output)

        output = self.out(output)
        
        return output, decoder_hidden, attn_weights


          
class NMT(NMTPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.input_dim)
        self.shared.padding_idx = config.pad_token_id
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False 
        self.encoder = Encoder(
                            input_embedding= self.shared,
                            config= encoder_config
                        )
        
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_hidden_layers
        self.decoder = AttnDecoderRNN(
                            input_embedding= self.shared,
                            config= decoder_config
                        )

        self.init_weights()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        langs: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
    ) -> Union[tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = self._shift_right(labels)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids= input_ids, langs= langs, hidden= None)
            

        encoder_output = encoder_outputs[0]
        encoder_hidden = encoder_outputs[1]
        encoder_hidden = encoder_hidden[0] + encoder_hidden[1]
        decoder_outputs, hidden_states, attentions = self.decoder(decoder_input_ids, encoder_hidden, encoder_output)
        
        lm_logits = decoder_outputs

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100) 
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))


        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=None,
            decoder_hidden_states=hidden_states,
            decoder_attentions=None,
            cross_attentions=attentions,
            encoder_last_hidden_state= encoder_hidden,
            encoder_hidden_states= None,
            encoder_attentions=None,
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        #if past_key_values is not None:
        #    input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "encoder_outputs": encoder_outputs,
            "use_cache": use_cache,
        }
    
    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder
    
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)