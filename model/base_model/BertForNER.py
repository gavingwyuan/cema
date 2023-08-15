from transformers.models.bert.modeling_bert import *

@add_start_docstrings(
    """
    Bert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    The code is from https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/bert/modeling_bert.py with modified
    """,
    BERT_START_DOCSTRING,
)

class BertForNER(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        if hasattr(config, "required_insert_action_ner_label"):
            config.required_insert_action_ner_label = config.required_insert_action_ner_label

            if hasattr(config, "nerlabel2ebdidx") and hasattr(config, "actionlabel2ebdidx") :
                self.nerlabel2ebdidx = config.nerlabel2ebdidx
                self.actionlabel2ebdidx = config.actionlabel2ebdidx
            else:
                config.required_insert_action_ner_label = False
        else:
            self.required_insert_action_ner_label = False

        self.init_weights()

    def insert_action_ner_label(self, input_ids, attention_mask = None):
        n_instances = input_ids.size()[0]
        num_ner_label_embedding = len(self.nerlabel2ebdidx)
        num_action_label_embedding = len(self.actionlabel2ebdidx)

        action_label_idx = torch.tensor([[self.nerlabel2ebdidx[idx] for idx in self.nerlabel2ebdidx]] * n_instances).to(input_ids.device)
        ner_label_idx = torch.tensor([[self.actionlabel2ebdidx[idx] for idx in self.actionlabel2ebdidx]] * n_instances).to(input_ids.device)

        prefix_idx = torch.cat([action_label_idx, ner_label_idx], dim=1)
        new_input_ids = torch.cat([prefix_idx, input_ids], dim=1)

        if attention_mask is not None:
            prefix_attention_mask = torch.ones(n_instances, num_ner_label_embedding + num_action_label_embedding).to(self.bert.device)
            new_attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)
            return new_input_ids, new_attention_mask
        else:
            return new_input_ids

    def remove_action_ner_label(self, tensor1):
        num_ner_label_embedding = len(self.nerlabel2ebdidx)
        num_action_label_embedding = len(self.actionlabel2ebdidx)
        prefix_len = num_action_label_embedding + num_ner_label_embedding

        if len(tensor1.size()) == 3:
            new_tensor1 = tensor1[:,prefix_len:,:]
        elif len(tensor1.size()) == 3:
            new_tensor1 = tensor1[:, prefix_len:]
        else:
            raise NotImplementedError

        return new_tensor1

    def forward(
        self,
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

        if self.required_insert_action_ner_label:
            input_ids, attention_mask = self.insert_action_ner_label(input_ids, attention_mask=attention_mask)

        outputs = self.bert(
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

        sequence_output = outputs[0]

        if self.required_insert_action_ner_label:
            sequence_output = self.remove_action_ner_label(sequence_output)

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
        classifier_path = os.path.join(output_folder, "classifier")
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        if not os.path.exists(classifier_path):
            os.mkdir(classifier_path)
        torch.save({"state_dict":self.classifier.state_dict()}, os.path.join(classifier_path, "classifier.pth.tar"))

    def load_classifier(self, output_folder):
        classifier_path = os.path.join(output_folder, "classifier")
        if not os.path.exists(classifier_path):
            print("{} is not exist".format(classifier_path))
        try:
            # load classifer layer
            classifer_checkpoint = torch.load(os.path.join(classifier_path, "classifier.pth.tar"))
            self.classifier.load_state_dict(classifer_checkpoint["state_dict"])
        except Exception as err:
            print(err.args)
            print(err)