from torch import nn
import torch
from transformers.modeling_bert import BertLayerNorm, gelu
from transformers.modeling_utils import PreTrainedModel
from torch.nn import CrossEntropyLoss


class RobertaConditionalLMHead(nn.Module):
    """Roberta Head for masked language modeling with conditionality on other variable(s)"""

    def __init__(self, config):
        super().__init__()
        print("Conditional LM Head Instantiation.")
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size+1, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, features, cond_var=None, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # take the conditional variable into account
        # print(x.shape)
        cond_var = cond_var.repeat(x.shape[1]).reshape(cond_var.shape[0], -1, 1)
        x = torch.cat((x, cond_var), dim=2)
        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x


# The following class is a mix of the RobertaForMaskedLM and XLMRobertaForMaskedLM from transformers' library
# to adapt it to the conditional LM head!
# @add_start_docstrings(
#     """XLM-RoBERTa Model with a `language modeling` head on top. """, XLM_ROBERTA_START_DOCSTRING,
# )
class XLMRobertaForConditionalMaskedLM(PreTrainedModel):
    """
    This class overrides :class:`~transformers.RobertaForMaskedLM`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    # config_class = XLMRobertaConfig
    # pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP

    def __init__(self, backbone, config):
        super().__init__(config)

        print("XLMRoberta Conditional Masked LM Instantiation.")
        self.xlm_roberta = backbone #RobertaModel(config)
        self.lm_head = RobertaConditionalLMHead(config)

        # self.config = config
        print(type(self.lm_head))
        # self.init_weights()

    # def get_output_embeddings(self):
    #     return self.lm_head.decoder

    #     @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            conditional_var=None,
    ):
        outputs = self.xlm_roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output, conditional_var)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)
