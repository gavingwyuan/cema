from transformers.models.bert.modeling_bert import *

from torch import nn

@add_start_docstrings(
    """
    Bert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    BERT_START_DOCSTRING,
)

class BertForActionClassification(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if hasattr(config, "cls_model_name"):
            if config.cls_model_name == "linear":
                self.classifier = nn.Linear(config.hidden_size*2, config.num_labels)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError("Please set cls_model_name for action predictor")

        self.init_weights()

    def forward(
        self,
        tag_input,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        reduction="mean"
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        token_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        def _get_tag_output(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,):
            return self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        tag_outputs = _get_tag_output(**tag_input)

        outputs = merge_bert_outputs(token_outputs, tag_outputs)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction=reduction)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def save_classifier(self, output_folder):

        def _save_classifier(classifier, output_folder, cls_name = "action_classifier"):
            classifier_path = os.path.join(output_folder, cls_name)
            if not os.path.exists(classifier_path):
                os.makedirs(classifier_path)

            torch.save({"state_dict":classifier.state_dict()}, os.path.join(classifier_path, "{}.pth.tar".format(cls_name)))

        _save_classifier(self.classifier, output_folder, cls_name="action_classifier")

    def load_classifier(self, output_folder):
        def _load_classifier(classifier, output_folder, cls_name="action_classifier"):
            found = True
            classifier_path = os.path.join(output_folder, cls_name)
            if not os.path.exists(classifier_path):
                print("{} is not exist".format(classifier_path))
                found = False
            try:
                # load classifer layer
                classifer_checkpoint = torch.load(os.path.join(classifier_path, "{}.pth.tar".format(cls_name)))
                classifier.load_state_dict(classifer_checkpoint["state_dict"])
            except Exception as err:
                print(err.args)
                print(err)
            return classifier, found

        self.classifier, found1 = _load_classifier(self.classifier, output_folder, cls_name="action_classifier")

        return found1

@dataclass
class MergeBertOutput(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) further processed by a
            Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence
            prediction (classification) objective during pretraining.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` and ``config.add_cross_attention=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            Tuple of :obj:`tuple(torch.FloatTensor)` of length :obj:`config.n_layers`, with each tuple having 2 tensors
            of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            ``config.is_encoder_decoder=True`` 2 additional tensors of shape :obj:`(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            ``config.is_encoder_decoder=True`` in the cross-attention blocks) that can be used (see
            :obj:`past_key_values` input) to speed up sequential decoding.
    """

    last_hidden_state: torch.FloatTensor = None

    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

    token_last_hidden_state: torch.FloatTensor = None
    token_pooler_output: torch.FloatTensor = None
    token_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    token_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    token_attentions: Optional[Tuple[torch.FloatTensor]] = None
    token_cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

    tag_last_hidden_state: torch.FloatTensor = None
    tag_pooler_output: torch.FloatTensor = None
    tag_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    tag_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    tag_attentions: Optional[Tuple[torch.FloatTensor]] = None
    tag_cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

def merge_bert_outputs(token_output, tag_output):
    last_hidden_state = torch.cat([token_output.last_hidden_state, tag_output.last_hidden_state], dim=2)

    pooler_output, hidden_states, past_key_values, attentions, cross_attentions = None, None, None, None, None
    if token_output.pooler_output is not None and tag_output.pooler_output is not None:
        pooler_output = torch.cat([token_output.pooler_output, tag_output.pooler_output], dim=2)
    if token_output.hidden_states is not None and tag_output.hidden_states is not None:
        hidden_states = token_output.hidden_states + tag_output.hidden_states
    if token_output.past_key_values is not None and tag_output.past_key_values is not None:
        past_key_values = token_output.past_key_values + tag_output.past_key_values
    if token_output.attentions is not None and tag_output.attentions is not None:
        attentions = token_output.attentions + tag_output.attentions
    if token_output.cross_attentions is not None and tag_output.cross_attentions is not None:
        cross_attentions = token_output.cross_attentions + tag_output.cross_attentions

    token_last_hidden_state = token_output.last_hidden_state
    token_pooler_output = token_output.pooler_output
    token_hidden_states = token_output.hidden_states
    token_past_key_values = token_output.past_key_values
    token_attentions = token_output.attentions
    token_cross_attentions = token_output.cross_attentions

    tag_last_hidden_state = tag_output.last_hidden_state
    tag_pooler_output = tag_output.pooler_output
    tag_hidden_states = tag_output.hidden_states
    tag_past_key_values = tag_output.past_key_values
    tag_attentions = tag_output.attentions
    tag_cross_attentions = tag_output.cross_attentions

    return MergeBertOutput(
        last_hidden_state = last_hidden_state,
        pooler_output = pooler_output,
        hidden_states = hidden_states,
        past_key_values=past_key_values,
        attentions = attentions,
        cross_attentions = cross_attentions,

        token_last_hidden_state = token_last_hidden_state,
        token_pooler_output = token_pooler_output,
        token_hidden_states = token_hidden_states,
        token_past_key_values=token_past_key_values,
        token_attentions = token_attentions,
        token_cross_attentions = token_cross_attentions,

        tag_last_hidden_state = tag_last_hidden_state,
        tag_pooler_output = tag_pooler_output,
        tag_hidden_states = tag_hidden_states,
        tag_past_key_values=tag_past_key_values,
        tag_attentions = tag_attentions,
        tag_cross_attentions = tag_cross_attentions
    )


