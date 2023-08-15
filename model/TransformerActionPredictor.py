from transformers import LongformerTokenizer, LongformerConfig

from transformers import BertTokenizer, BertConfig
from model.base_model.BertForActionClassification import BertForActionClassification

import numpy as np
from seqeval.metrics.sequence_labeling import get_entities

from transformers import AdamW
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
from copy import deepcopy
from model.CostEstimator import CostPerformaceEval, AnnotationAction
from torch.nn import CrossEntropyLoss
import pickle
import hiddenlayer as hl
from active_learning.acquisition import CEMAPartialStrategy



from model.CustomizedTokenizer import TokenizerWrapper
from model.Doc2Piece import Doc2Piece
# from utils.utils import compute_seq_entropy, init_logger

class TFAPDataset(Doc2Piece):
    NoneNer = "[nonener]"
    NoneAction = "[noneaction]"
    def __init__(self, sents, tokenizer = None, end_symbol=["EndSymbol"], max_length=510, split_way = "both_symbol_length"):
        super().__init__(sents, tokenizer = tokenizer, end_symbol=end_symbol, max_length=max_length, split_way = split_way)

    @staticmethod
    def merge_sents_actions(sents, actions):
        sents, actions = deepcopy(sents), deepcopy(actions)
        individual_list = TFAPDataset.unpackage(sents)
        new_sents = TFAPDataset.package(individual_list + [actions])
        return new_sents

    @staticmethod
    def get_tf_ap_input(sents, pred_labels, sim_threshold=0, sim_method="not_intersection"):
        # from
        ## sents = [instance,...,...], instance = [(token, none label or true label),...,...]
        ## pred_label = [instance,...,...], instance = [label,...,...]
        # to
        ## sents_with_pred_label = [instance,...,...], instance = [(token, label),...,...]
        ## action_labels = [instance,...,...], instance = [label,...,...]

        tokens, true_label_list = TFAPDataset.unpackage(sents)
        tokens = [list(row) for row in tokens]
        true_label_list = [list(row) for row in true_label_list]

        cpe = CostPerformaceEval(true_label_list=true_label_list, pred_labels_list=pred_labels,
                                 sim_threshold=sim_threshold, sim_method=sim_method)

        action_labels = [aa.action_labels for aa in cpe.annotaion_actions]

        sents4ap = TFAPDataset.package([tokens, pred_labels, action_labels])

        return sents4ap

    @staticmethod
    def get_ap_input(train_sents, pred_labels, labels_set, ner_end_symbol, sim_threshold=0,
                                sim_method="not_intersection"):

        labels_set = {label.split("-")[-1] for label in labels_set} - {"O"}
        train_true_labels = []
        for sent in train_sents:
            train_true_labels.append([label for token, label in sent])

        _, sents_partial, pred_labels_partial, action_labels_partial, _ = CEMAPartialStrategy.partial_with_ety(
            train_sents, pred_labels,
            None, ner_end_symbol,
            refer_pred_labels=train_true_labels)

        new_sents_partial = sents_partial
        new_seg_train_preds = [[x[1] for x in row] for row in sents_partial]

        sents_4ap = TFAPDataset.get_tf_ap_input(new_sents_partial, new_seg_train_preds,
                                                      sim_threshold=sim_threshold,
                                                      sim_method=sim_method)

        new_true_labels = TFAPDataset.unpackage(new_sents_partial)[1]

        return sents_4ap, new_true_labels, new_sents_partial, new_seg_train_preds

class TFAPTokenizer(TokenizerWrapper):
    def __init__(self, tokenizer, output_folder="", special_prefix = "[special]"):
        """
        add labels tokenization base on SimpleTokenizer
        :param tokenizer:
        :param output_folder:
        """
        super(TFAPTokenizer, self).__init__(tokenizer, output_folder=output_folder, special_prefix = special_prefix)
        self.idx_pad_action_label = CrossEntropyLoss().ignore_index

    def encode_action_labels(self, label_sents, label_mask, label2idx):
        # if label for [Shen, ##zhen] is [S-CONFIRMATION, S-CONFIRMATION] AND label_mask = [1, 0]
        # Then, idx_lbaels = [[label2idx[S-CONFIRMATION], idx_pad_ner_label]
        # when loss is calculating the loss for idx_pad_ner_label will be ignored
        # label mask of the labels assigned to padding, [CLS], and [SEP] are 0.
        max_length = max([len(x) for x in label_sents])
        max_length = min([max_length, self.max_len_single_sentence]) # max_len_single_sentence did not consider start and end symbol
        idx_labels = []
        for ith, sent in enumerate(label_sents):
            sent = list(sent)
            sent = sent[:max_length]
            sent = [label2idx[label] if label_mask[ith][jth] else self.idx_pad_action_label for jth, label in enumerate(sent) ]
            sent = [self.idx_pad_action_label] + sent + [self.idx_pad_action_label]+ [self.idx_pad_action_label] * (max_length - len(sent))
            idx_labels.append(sent)
        return idx_labels

    def encode_action_label_mask(self, label_mask):
        # if label for [Shen, ##zhen] is [S-LOC, S-LOC] AND label_mask = [1, 0]
        # Then, idx_lbaels = [[label2idx[S-LOC], idx_pad_ner_label]
        # when loss is calculating the loss for idx_pad_ner_label will be ignored
        # label mask of the labels assigned to padding, [CLS], and [SEP] are 0.
        max_length = max([len(x) for x in label_mask])
        max_length = min([max_length, self.max_len_single_sentence]) # max_len_single_sentence did not consider start and end symbol
        new_label_mask = []
        for sent in label_mask:
            sent = list(sent)
            sent = sent[:max_length]
            sent = [0] + sent + [0]+ [0] * (max_length - len(sent))
            new_label_mask.append(sent)

        return new_label_mask

    def decode_action_labels(self, seq_list, label_mask, idx2label = None):
        assert len(seq_list) == len(label_mask)
        if idx2label is None:
            label_list = [[idx for idx in seq[1:1 + len(label_mask[ith])]] for ith, seq in
                          enumerate(seq_list)]
        else:
            label_list = [[idx2label[idx] for idx in seq[1:1+len(label_mask[ith])]] for ith, seq in enumerate(seq_list)]
        return label_list

class TFAPDataProcessor(object):
    def __init__(self, sents, action2idx, tokenizer, end_symbol=["EndSymbol"], max_length=0):
        self.action2idx = action2idx
        self.idx2action = {self.action2idx[k]: k for k in self.action2idx}
        self.tokenizer = tokenizer
        self.dataset = TFAPDataset(sents, tokenizer = tokenizer, end_symbol=end_symbol, max_length=max_length)
        self.collate_fn = lambda batch: TFAPDataProcessor.get_encode_data(batch, tokenizer = self.tokenizer,
                                                                        action2idx = self.action2idx)

    @staticmethod
    def get_encoded_x(token_sents, tags_sents, tokenizer):
        # elements in sents is tuple [(token, action_label),,,], each tuple represent a sentence
        # actions is a string tuple, each element is a action label
        # the input should be tokenized.

        token_input = tokenizer(
            token_sents
        )

        tag_input = tokenizer(
            tags_sents,
            are_special_token = True
        )

        token_input_ids = token_input['input_ids']
        token_attention_mask = token_input['attention_mask']
        tag_input_ids = tag_input['input_ids']
        tag_attention_mask = tag_input['attention_mask']

        return torch.LongTensor(token_input_ids), torch.LongTensor(token_attention_mask), \
               torch.LongTensor(tag_input_ids), torch.LongTensor(tag_attention_mask)

    # @staticmethod
    # def get_encoded_y(action_sents, action2idx, tokenizer):
    #     # inputs = tokenizer("Hello, a bird flys in the sky.", return_tensors="pt", max_length=3)
    #     # tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    #     # ['<s>', 'Hello', '</s>']
    #     idx_action_list = tokenizer.encode_action_labels(action_sents, action2idx)
    #     return torch.LongTensor(idx_action_list)
    @staticmethod
    def get_encoded_y(label_sents, label_mask, label2idx, tokenizer):
        # inputs = tokenizer("Hello, a bird flys in the sky.", return_tensors="pt", max_length=3)
        # tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        # ['<s>', 'Hello', '</s>']
        idx_label_list = tokenizer.encode_action_labels(label_sents, label_mask, label2idx)
        return torch.LongTensor(idx_label_list)

    @staticmethod
    def get_encode_data(batch, tokenizer = None, action2idx = None):
        data = TFAPDataset.unpackage(batch)
        if len(data) < 1:
            raise NotImplementedError("error in get_encode_data")

        token_sents, tags_sents, action_labels = data[0], data[1], None

        token_input_ids, token_attention_mask, tag_input_ids, tag_attention_mask = \
            TFAPDataProcessor.get_encoded_x(token_sents, tags_sents, tokenizer)

        label_mask = data[-1]

        if len(data) >= 3: # token_seq, ner_label, action_label, label_mask
            action_labels = data[2] # action_label
            idx_labels = TFAPDataProcessor.get_encoded_y(action_labels, label_mask, action2idx, tokenizer)

            return {"token_input_ids":token_input_ids, "token_attention_mask":token_attention_mask,
                    "tag_input_ids": tag_input_ids, "tag_attention_mask": tag_attention_mask,
                    "idx_labels":idx_labels, "label_mask":label_mask}

        return {"token_input_ids":token_input_ids, "token_attention_mask":token_attention_mask,
                "tag_input_ids": tag_input_ids, "tag_attention_mask": tag_attention_mask,
                "label_mask":label_mask}

    def get_dataloader4training(self, batch_size, shuffle = True):
        trainData = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn) # 处理成多个batch的形式
        return trainData

    def get_dataloader4val(self, batch_size):
        valData = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn) # 处理成多个batch的形式
        return valData

    def get_decoded_y(self, idx_action_list, label_mask, need_idx2labels = True):
        if need_idx2labels:
            idx2action = self.idx2action
        else:
            idx2action = None
        label_list = self.tokenizer.decode_action_labels(idx_action_list, label_mask, idx2action)
        return label_list

    def recover_from_encode_y(self, idx_label_list, label_mask, need_idx2labels = True):
        # align the predicted labels to original input shape
        # label_mask = [instance 1, instance 2, ..] where the length of the instance i is same with the no. of token in sequence i
        y_segmented = self.get_decoded_y(idx_label_list, label_mask, need_idx2labels = need_idx2labels)
        y_tokenized = self.dataset.segmented2tokenized(y_segmented, self.dataset.piece_span, is_token=False)
        mask_tokenized = self.dataset.segmented2tokenized(label_mask, self.dataset.piece_span, is_token=False)
        y_original = self.dataset.tokenized2original(y_tokenized, mask_tokenized)
        return y_original

class TransformerActionPredictor(object):
    def __init__(self,
                 ner_labels,
                 learning_rate,
                 epochs,
                 batch_size,
                 cls_model_name = "linear",
                 lstm_out_features=None,
                 transformers_nhid=None,
                 fusion_way = "concatenate",
                 use_fea_encoder = False,
                 device='cuda',
                 output_folder='ActionPredictor',
                 base_model_name = "bert-base-uncased",
                 max_length = 510,
                 end_symbol = ["EndSymbol"],
                 gradient_checkpointing = False
                 ):

        self.gradient_checkpointing = gradient_checkpointing
        self.output_folder = output_folder  # 模型存储的文件夹

        self.lr = learning_rate  # 学习率
        self.epochs = epochs

        self.batch_size = batch_size  # 训练集的batch_size

        self.action2idx = AnnotationAction.action2idx
        self.idx2action = {self.action2idx[k]:k for k in self.action2idx}
        self.base_model_name = base_model_name
        self.ner_labels = ner_labels[:]
        # self.idx2nerlabel = {i:ll for i,ll in enumerate(self.ner_labels)}

        # this 4 parameter will be initialized in self.load_model
        self.config = None
        self.base_tokenizer = None
        self.model = None
        self.tokenizer = None
        self.cls_model_name = cls_model_name
        self.lstm_out_features = lstm_out_features
        self.transformers_nhid = transformers_nhid
        self.fusion_way = fusion_way
        self.use_fea_encoder = use_fea_encoder
        self.device = device # "cpu", "cuda", "cuda:0"
        self.found_classifier = self.load_model()

        self.max_length = max_length
        self.end_symbol = end_symbol
        self.predicted_batch_size = 15*self.batch_size
        self.batch_size_probability = 10*self.batch_size

    def flat_accuracy(self, label_ids, preds_ids, label_mask):
        """A function for calculating accuracy scores"""
        label_ids_flat = []
        for ith, seq in enumerate(label_ids):
            for jth, label in enumerate(seq):
                if label_mask[ith][jth]:
                    label_ids_flat.append(label)

        preds_ids_flat = []
        for ith, seq in enumerate(preds_ids):
            for jth, label in enumerate(seq):
                if label_mask[ith][jth]:
                    preds_ids_flat.append(label)

        assert len(label_ids_flat) == len(preds_ids_flat)
        return accuracy_score(label_ids_flat, preds_ids_flat)

    def get_batches(self, sents):
        end_symbol = self.end_symbol
        max_length = self.max_length

        apdp = TFAPDataProcessor(sents, self.action2idx, self.tokenizer, max_length = max_length, end_symbol = end_symbol)
        trainData = apdp.get_dataloader4training(self.batch_size)

        return trainData

    def train_on_batch(self, batch, optimizer = None, has_set_train_state = False):
        if not has_set_train_state:
            self.model = self.model.to(self.device)
            self.model.train()

        if optimizer is None:
            optimizer = AdamW(self.model.parameters(), lr=self.lr)

        idx_labels = batch["idx_labels"].to(self.device)
        label_mask = batch["label_mask"]

        token_inputs = {}
        token_inputs["input_ids"] = batch["token_input_ids"].to(self.device)
        token_inputs["attention_mask"] = batch["token_attention_mask"].to(self.device)
        token_inputs["labels"] = idx_labels
        tag_inputs = {}
        tag_inputs["input_ids"] = batch["tag_input_ids"].to(self.device)
        tag_inputs["attention_mask"] = batch["tag_attention_mask"].to(self.device)

        optimizer.zero_grad()
        outputs = self.get_output(tag_inputs, token_inputs)
        loss, logits = outputs.loss, outputs.logits
        step_loss = loss.item()
        logits = logits.detach().cpu().numpy()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()

        torch.cuda.empty_cache()

        return step_loss, logits

    def train(self, sents, log_folder="", record_training = True, logger = None, save_model=True):
        if logger is None:
            class logger():
                info = None
            logger.info = print

        end_symbol = self.end_symbol
        max_length = self.max_length

        # labels is int sequece
        epoch_history, step_history = None, None

        apdp = TFAPDataProcessor(sents, self.action2idx, self.tokenizer, max_length = max_length, end_symbol = end_symbol)
        trainData = apdp.get_dataloader4training(self.batch_size)

        total_epoch = self.epochs

        if log_folder != "":
            # 记录训练过程的指标
            epoch_history = hl.History()
            step_history = hl.History()

        self.model = self.model.to(self.device)
        optimizer = AdamW(self.model.parameters(), lr=self.lr)  # 后面看看需不需要改参数

        total_steps = len(trainData) * total_epoch
        # warm_up_ratio = 0.1
        # num_warmup_steps = int(warm_up_ratio * total_steps)
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)

        interval = int(total_steps/50)
        interval = max([interval, 1])
        interval = interval * 10

        cur_steps = 0 # 记录已经经过的step数
        total_loss = 0
        for epoch in range(self.epochs):
            epoch_loss = 0
            # logger.info('epoch: {}'.format(epoch))
            # training
            for step, batch in enumerate(trainData):
                self.model.train()

                idx_labels = batch["idx_labels"].to(self.device)
                label_mask = batch["label_mask"]

                token_inputs = {}
                token_inputs["input_ids"] = batch["token_input_ids"].to(self.device)
                token_inputs["attention_mask"] = batch["token_attention_mask"].to(self.device)
                token_inputs["labels"] = idx_labels
                tag_inputs = {}
                tag_inputs["input_ids"] = batch["tag_input_ids"].to(self.device)
                tag_inputs["attention_mask"] = batch["tag_attention_mask"].to(self.device)

                # print(idx_labels.shape)

                optimizer.zero_grad()
                # outputs = self.model(tag_inputs,
                #                      **token_inputs
                #                      )  # 输出loss 和 每个分类对应的输出，softmax后才是预测是对应分类的概率
                outputs = self.get_output(tag_inputs, token_inputs)
                loss, logits = outputs.loss, outputs.logits
                total_loss += loss.item()
                epoch_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                # scheduler.step()
                cur_steps += 1
                if record_training and cur_steps % interval == 0:  # 每10步输出一下训练的结果，flat_accuracy()会对logits进行softmax
                    self.model.eval()
                    logits = logits.detach().cpu().numpy()
                    preds_ids = logits.argmax(axis=2).tolist()
                    label_ids = idx_labels.cpu().tolist()
                    format_preds_ids = apdp.get_decoded_y(preds_ids, label_mask, need_idx2labels=False)
                    format_label_ids = apdp.get_decoded_y(label_ids, label_mask, need_idx2labels=False)
                    train_accuracy = self.flat_accuracy(format_label_ids, format_preds_ids, label_mask)
                    logger.info('action predictor - epoch {}/{}, current total step: {}/{}, step in an epoch: {}'.format(epoch+1, total_epoch, cur_steps, total_steps, step))
                    logger.info('action predictor - Train Accuracy: {}'.format(train_accuracy))
                    logger.info('action predictor - total_loss/cur_steps: {}'.format(total_loss/cur_steps))
                    logger.info("-------")

                    if epoch_history is not None:
                        step_history.log(cur_steps,
                                    step_train_loss=loss.item(),
                                    step_train_accuracy=train_accuracy,
                                    avg_train_loss=total_loss/cur_steps,
                                    cur_steps=cur_steps,
                                    i_training_epoch=epoch,
                                    j_step_in_i_epcch=step,
                                    )
                        step_history.save(log_folder+"tfner_training_step_history.log.pkl")

                torch.cuda.empty_cache()

            if record_training and epoch_history is not None:
                epoch_history.log(epoch,
                                  epoch_loss=epoch_loss,
                                  avg_train_loss=total_loss/cur_steps,
                                  train_f1_score=self.test(sents))
                epoch_history.save(log_folder + "tfner_training_epoch_history.log.pkl")

            if save_model and epoch % 3 == 0:
                self.save_model(output_folder = self.output_folder)

        if save_model:
            self.save_model(output_folder=self.output_folder)

    def get_output(self, tag_inputs, token_inputs):
        cur_model = self.model

        if isinstance(cur_model, torch.torch.nn.DataParallel):
            if len(token_inputs["input_ids"]) < torch.cuda.device_count():
                    cur_model = cur_model.module
            outputs = cur_model(tag_inputs,
                                **token_inputs,
                                reduction="sum")
            if hasattr(outputs.loss, "sum"):
                outputs.loss = outputs.loss.sum()
            if 'labels' in token_inputs:
                n_tokens = (token_inputs['labels'] != CrossEntropyLoss().ignore_index).sum()
                outputs.loss = outputs.loss/n_tokens
        else:
            outputs = cur_model(tag_inputs,
                                **token_inputs)

        return outputs

    def test(self, sents):
        action_labels = [[x[2] for x in instance] for instance in sents]
        preds = self.predict(sents)
        f1_score = self.cal_token_level_f1(action_labels, preds)
        print("**action predictor token level f1**:", f1_score)
        return f1_score

    def predict_core(self, sents):
        ner_labels = TFAPDataset.unpackage(sents)[1]
        ner_labels = [list(row) for row in ner_labels]
        probability = self.get_probability(sents)
        # action_idxs = [probs.argmax(-1) for probs in probability]
        action_idxs = TransformerActionPredictor.adjust_action_sequence(probability,
                               ner_labels,
                               NoneLabel="O",
                               action2idx=self.action2idx,
                               O_idx=self.action2idx["O"],
                               CONFIRMATION_idx=self.action2idx["B-CONFIRMATION"],
                               DELETE_idx=self.action2idx["B-DELETE"],
                               ADD_idx=self.action2idx["B-ADD"],
                               REVISE_idx=self.action2idx["B-REVISE"],
                               DELETEADD_idx=self.action2idx["B-DELETEADD"])

        pred_actions = [[self.idx2action[idx] for idx in row] for row in action_idxs]
        return pred_actions

    def predict(self, sents):
        batch_size = self.predicted_batch_size
        max_length = self.max_length
        step_size = int(20*batch_size*max_length/len(sents[-1]))
        step_size = max([1, step_size])
        pred_actions = []

        start = 0
        while start < len(sents):
            batch_sent = sents[start:(start + step_size)]
            batch_pred_actions = self.predict_core(batch_sent)
            pred_actions.extend(batch_pred_actions)
            start += len(batch_sent)

        return pred_actions

    def _get_probability(self, tfap_dp, device, batch_size):
        # to obtain the probability of assigning each label for each tokens in the sequence
        # recovered_probability is N*LENGTH*NUMBER_CLUSS
        valData = tfap_dp.get_dataloader4val(batch_size)
        model = self.model.to(device)
        model.eval()
        all_probability = []
        all_label_mask = []
        for i, batch in enumerate(valData):
            token_inputs = {}
            token_inputs["input_ids"] = batch["token_input_ids"].to(self.device)
            token_inputs["attention_mask"] = batch["token_attention_mask"].to(self.device)
            tag_inputs = {}
            tag_inputs["input_ids"] = batch["tag_input_ids"].to(self.device)
            tag_inputs["attention_mask"] = batch["tag_attention_mask"].to(self.device)

            label_mask = batch["label_mask"]
            all_label_mask.extend(label_mask)

            with torch.no_grad():
                outputs = self.get_output(tag_inputs, token_inputs)
                logits = outputs[0]
                probability = torch.nn.functional.softmax(logits, dim=2)
                all_probability += probability.tolist()

            torch.cuda.empty_cache()

        # recover pieces to original documents
        recovered_probability = tfap_dp.recover_from_encode_y(all_probability, all_label_mask, need_idx2labels=False)
        recovered_probability = [np.array(row) for row in recovered_probability]
        return recovered_probability

    def get_probability(self, sents):
        end_symbol = self.end_symbol
        batch_size = self.batch_size_probability
        max_length = self.max_length
        device = self.device

        apdp = TFAPDataProcessor(sents, self.action2idx, self.tokenizer, end_symbol=end_symbol, max_length=max_length)
        probability = self._get_probability(apdp, batch_size=batch_size, device=device)
        return probability

    @staticmethod
    def adjust_action_sequence(probabilities,
                               ner_labels,
                               NoneLabel="O",
                               action2idx=None,
                               O_idx=None,
                               CONFIRMATION_idx=None,
                               DELETE_idx=None,
                               ADD_idx=None,
                               REVISE_idx=None,
                               DELETEADD_idx=None):
        """
        probabilities = [np.array([[0, 2, 100, 1, 3, 3],
                                   [100, 2, 5, 1, 3, 10],
                                   [0, 2, 2, 1, 3, 30],
                                   [10, 2, 2, 1, 3, 0], [10, 2, 2, 1, 3, 0],
                                   [10, 2, 2, 1, 3, 3], [0, 2, 2, 100, 3, 3], [10, 2, 2, 1, 3, 3],
                                   [10, 2, 2, 100, 2, 3], [10, 2, 2, 2, 3, 3], [10, 2, 2, 1, 3, 3]], dtype=np.float)]
        ner_labels = [["O",
                       "S-A",
                       "O",
                       "B-B", "E-B",
                       "B-A", "I-A", "E-A",
                       "B-C", "I-C", "E-C"]]
        adjust_action_sequence(probabilities,
                                   ner_labels,
                                   NoneLabel = "O",
                                   action2idx = action2idx,
                                   O_idx = 0,
                                   CONFIRMATION_idx = 1,
                                   DELETE_idx = 2,
                                   ADD_idx = 3,
                                   REVISE_idx = 4,
                                   DELETEADD_idx = 5) == np.array([[3, 5, 3, 4, 0, 2, 3, 0, 5, 0, 0]])
        """

        # it is also work if logits is input instead of probabilities
        # actions = ["O", "B-CONFIRMATION", "B-DELETE", "B-ADD", "B-REVISE", "B-DELETEADD"]
        assert action2idx is not None or (O_idx is not None
                                          or CONFIRMATION_idx is not None
                                          or DELETE_idx is not None
                                          or ADD_idx is not None
                                          or REVISE_idx is not None
                                          or DELETEADD_idx is not None)
        if action2idx is not None:
            O_idx = action2idx["O"]
            CONFIRMATION_idx = action2idx["B-CONFIRMATION"]
            DELETE_idx = action2idx["B-DELETE"]
            ADD_idx = action2idx["B-ADD"]
            REVISE_idx = action2idx["B-REVISE"]
            DELETEADD_idx = action2idx["B-DELETEADD"]

        assert [len(row) for row in ner_labels] == [len(row) for row in probabilities]
        row_len = [len(row) for row in ner_labels]
        for ith in range(len(row_len)):
            for jth in range(row_len[ith]):
                if ner_labels[ith][jth] == NoneLabel:
                    probabilities[ith][jth, CONFIRMATION_idx] = -float("inf")
                    probabilities[ith][jth, DELETE_idx] = -float("inf")
                    probabilities[ith][jth, REVISE_idx] = -float("inf")
                    probabilities[ith][jth, DELETEADD_idx] = -float("inf")

        #############################################################################################
        # get continous labeled spans
        continous_labeled_spans = []
        for ith in range(len(row_len)):
            spans = TransformerActionPredictor.get_continue_labeled_spans(ner_labels[ith])
            continous_labeled_spans.append(spans)


        for ith in range(len(row_len)):
            for span in continous_labeled_spans[ith]:
                s = span[1]
                t = span[2]
                partial = probabilities[ith][s:t + 1]
                if TransformerActionPredictor.is_all_o(partial, O_idx):
                    probabilities[ith][s, O_idx] = -float("inf")

        ##################################################################################
        for ith in range(len(row_len)):
            for span in continous_labeled_spans[ith]:
                s = span[1]
                t = span[2]
                probabilities[ith][s, O_idx] = -float("inf")

        for ith in range(len(row_len)):
            for span in continous_labeled_spans[ith]:
                s = span[1]
                t = span[2]
                partial = probabilities[ith][s:t + 1]
                op_idx_list = partial.argmax(-1)
                if op_idx_list[0] == CONFIRMATION_idx or op_idx_list[0] == REVISE_idx:
                    for relative_loc in range(1, len(op_idx_list)):
                        probabilities[ith][s + relative_loc, O_idx] = float("inf")

        # If delete, subsequence token may be
        # deleteadd => add
        # delete => O
        # confirm => O
        # revise => O
        for ith in range(len(row_len)):
            for span in continous_labeled_spans[ith]:
                s = span[1]
                t = span[2]
                partial = probabilities[ith][s:t + 1]
                op_idx_list = partial.argmax(-1)
                if op_idx_list[0] == DELETE_idx:
                    for relative_loc in range(1, len(op_idx_list)):
                        if op_idx_list[relative_loc] == DELETEADD_idx:
                            probabilities[ith][s + relative_loc, ADD_idx] = float("inf")

                        if op_idx_list[relative_loc] == DELETE_idx:
                            probabilities[ith][s + relative_loc, O_idx] = float("inf")

                        if op_idx_list[relative_loc] == CONFIRMATION_idx:
                            probabilities[ith][s + relative_loc, O_idx] = float("inf")

                        if op_idx_list[relative_loc] == REVISE_idx:
                            probabilities[ith][s + relative_loc, O_idx] = float("inf")

        # If deleteadd, subsequence token may be
        # deleteadd => add
        # delete => O
        # confirm => O
        # revise => O
        for ith in range(len(row_len)):
            for span in continous_labeled_spans[ith]:
                s = span[1]
                t = span[2]
                partial = probabilities[ith][s:t + 1]
                op_idx_list = partial.argmax(-1)
                if op_idx_list[0] == DELETEADD_idx:
                    for relative_loc in range(1, len(op_idx_list)):
                        if op_idx_list[relative_loc] == DELETEADD_idx:
                            probabilities[ith][s + relative_loc, ADD_idx] = float("inf")

                        if op_idx_list[relative_loc] == DELETE_idx:
                            probabilities[ith][s + relative_loc, O_idx] = float("inf")

                        if op_idx_list[relative_loc] == CONFIRMATION_idx:
                            probabilities[ith][s + relative_loc, O_idx] = float("inf")

                        if op_idx_list[relative_loc] == REVISE_idx:
                            probabilities[ith][s + relative_loc, O_idx] = float("inf")

        # If there is an ADD, subsequence token may be
        # add => deleteadd
        # deleteadd => add
        # delete => O
        # confirm => O
        # revise => O
        for ith in range(len(row_len)):
            for span in continous_labeled_spans[ith]:
                s = span[1]
                t = span[2]
                partial = probabilities[ith][s:t + 1]
                op_idx_list = partial.argmax(-1)
                if op_idx_list[0] == ADD_idx:
                    probabilities[ith][s, DELETEADD_idx] = float("inf")

                    for relative_loc in range(1, len(op_idx_list)):
                        if op_idx_list[relative_loc] == DELETEADD_idx:
                            probabilities[ith][s + relative_loc, ADD_idx] = float("inf")

                        if op_idx_list[relative_loc] == DELETE_idx:
                            probabilities[ith][s + relative_loc, O_idx] = float("inf")

                        if op_idx_list[relative_loc] == CONFIRMATION_idx:
                            probabilities[ith][s + relative_loc, O_idx] = float("inf")

                        if op_idx_list[relative_loc] == REVISE_idx:
                            probabilities[ith][s + relative_loc, O_idx] = float("inf")

        ##################################################################################
        # If there is an ADD, then delete action is add to be DELETE-ADD
        for ith in range(len(row_len)):
            for span in continous_labeled_spans[ith]:
                s = span[1]
                t = span[2]
                partial = probabilities[ith][s:t + 1]
                if TransformerActionPredictor.is_only_add(partial, O_idx, ADD_idx):
                    loc = probabilities[ith][s:s + 1].argmax(-1)
                    if loc == ADD_idx:
                        probabilities[ith][s, DELETEADD_idx] = float("inf")
                    else:
                        probabilities[ith][s, DELETE_idx] = float("inf")

        action_idx = [probs.argmax(-1) for probs in probabilities]

        # return probabilities, action_idx
        return action_idx

    @staticmethod
    def is_all_o(probabilities, O_idx):
        """
        probabilities = np.array([[2, 1], [3, 4], [5, 4]])
        O_idx = 0
        is_all_o(probabilities[[0,2],:], O_idx) == True
        is_all_o(probabilities, O_idx) == False
        """
        labels_set = set(probabilities.argmax(-1)) - {O_idx}
        # print(probabilities, labels_set, O_idx)
        if len(labels_set) == 0:
            return True
        else:
            return False

    @staticmethod
    def is_only_add(probabilities, O_idx, ADD_idx):
        """
        at least one add
        probabilities = np.array([[2, 1], [3, 4], [5, 4]])
        O_idx = 0
        ADD_idx = 1
        is_only_add(probabilities[[0,2],:], O_idx, ADD_idx) == False
        is_only_add(probabilities, O_idx, ADD_idx) == True
        """
        labels = list(set(probabilities.argmax(-1)) - {O_idx})
        # print(probabilities, labels, O_idx, ADD_idx)
        if len(labels) == 1 and labels[0] == ADD_idx:
            return True
        else:
            return False

    @staticmethod
    def get_continue_labeled_spans(ner_labels):
        spans = get_entities(ner_labels)
        return spans


    def cal_token_level_f1(self, Y_true, y_pred):
        # token level f1 score
        pre = 0
        pre_tot = 0
        rec = 0
        rec_tot = 0
        corr = 0
        total = 0
        for i in range(len(Y_true)):
            for j in range(len(Y_true[i])):
                total += 1
                if y_pred[i][j] == Y_true[i][j]:
                    corr += 1
                if y_pred[i][j] != 'O':  # not 'O'
                    pre_tot += 1
                    if y_pred[i][j] == Y_true[i][j]:
                        pre += 1
                if Y_true[i][j] != 'O':
                    rec_tot += 1
                    if y_pred[i][j] == Y_true[i][j]:
                        rec += 1

        res = corr * 1. / total
        #print("Accuracy (token level)", res)
        if pre_tot == 0:
            pre = 0
        else:
            pre = 1. * pre / pre_tot
        rec = 1. * rec / rec_tot
       #print(pre, rec)

        beta = 1
        f1score = 0
        if pre != 0 or rec != 0:
            f1score = (beta * beta + 1) * pre * rec / \
                (beta * beta * pre + rec)
        #print("F1", f1score)
        return f1score

    def load_model(self, input_folder = None):
        num_labels = len(self.action2idx)
        if input_folder is None: input_folder = self.output_folder
        found_classifier = False
        if self.base_model_name == "bert-base-uncased":
            found_classifier = self.load_bert(input_folder, num_labels, self.fusion_way, self.use_fea_encoder, cls_model_name=self.cls_model_name, lstm_out_features=self.lstm_out_features,
                                                                         transformers_nhid=self.transformers_nhid)
        elif self.base_model_name == "dbmdz/bert-base-german-cased":
            found_classifier = self.load_german_bert(input_folder, num_labels, self.fusion_way, self.use_fea_encoder, cls_model_name=self.cls_model_name, lstm_out_features=self.lstm_out_features,
                           transformers_nhid=self.transformers_nhid)

        # max_position_embeddings containing start and end position, model_max_length exclude start and end position
        self.tokenizer.model_max_length = self.model.config.max_position_embeddings - 2
        self.tokenizer.add_tokens([Doc2Piece.NewLineLabel])
        self.tokenizer.add_tokens([Doc2Piece.StrangeSymbols])
        self.tokenizer.add_tokens([Doc2Piece.NoneToken])
        # ner labels 要设置成特殊的样子，区分开普通的token
        self.tokenizer.add_tokens(list(set(self.ner_labels)),are_special = True)
        self.model = self.model.to(self.device)

        n_gpu = torch.cuda.device_count()
        if n_gpu > 0 and "cuda" in self.device:
            self.model = self.model.cuda()
            # if n_gpu > 1:
            #     self.model = torch.nn.DataParallel(self.model)
        return found_classifier

    def save_model(self, output_folder = None):
        if output_folder is None: output_folder = self.output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        if self.base_model_name == "bert-base-uncased":
            self.save_bert(output_folder)
        elif self.base_model_name == "dbmdz/bert-base-german-cased":
            self.save_german_bert(output_folder)

    def load_bert(self, output_folder, num_labels, fusion_way, use_fea_encoder, cls_model_name = "linear", lstm_out_features = None,
                  transformers_nhid=None):
        found_classifier = False
        if os.path.exists(os.path.join(output_folder, "tokenizer.pkl")):
            output_folder = self.output_folder
            self.config = BertConfig.from_pretrained(output_folder)
            self.config.num_labels = num_labels
            self.config.cls_model_name = cls_model_name
            self.config.lstm_out_features = lstm_out_features
            self.transformers_nhid = transformers_nhid
            self.config.fusion_way = fusion_way
            self.config.use_fea_encoder = use_fea_encoder
            self.config.gradient_checkpointing = self.gradient_checkpointing
            self.tokenizer = TFAPTokenizer.load(output_folder)
            self.model = BertForActionClassification.from_pretrained(output_folder, config=self.config)
            found_classifier = self.model.load_classifier(output_folder)
        else:
            old_output_folder = output_folder
            output_folder = 'bert-base-uncased'
            print("Path {} is not exist. Now, {} will be loaded.".format(old_output_folder, output_folder))
            self.config = BertConfig.from_pretrained(output_folder)
            self.config.num_labels = num_labels
            self.config.cls_model_name = cls_model_name
            self.config.lstm_out_features = lstm_out_features
            self.transformers_nhid = transformers_nhid
            self.config.fusion_way = fusion_way
            self.config.use_fea_encoder = use_fea_encoder
            self.base_tokenizer = BertTokenizer.from_pretrained(output_folder)
            self.config.gradient_checkpointing = self.gradient_checkpointing
            self.tokenizer = TFAPTokenizer(self.base_tokenizer, output_folder, special_prefix = "[nerlabels]")
            self.model = BertForActionClassification.from_pretrained(output_folder, config=self.config)
        return found_classifier

    def load_german_bert(self, output_folder, num_labels, fusion_way, use_fea_encoder, cls_model_name = "linear", lstm_out_features = None,
                  transformers_nhid=None):
        found_classifier = False
        if os.path.exists(os.path.join(output_folder, "tokenizer.pkl")):
            output_folder = self.output_folder
            self.config = BertConfig.from_pretrained(output_folder)
            self.config.num_labels = num_labels
            self.config.cls_model_name = cls_model_name
            self.config.lstm_out_features = lstm_out_features
            self.transformers_nhid = transformers_nhid
            self.config.fusion_way = fusion_way
            self.config.use_fea_encoder = use_fea_encoder
            self.config.gradient_checkpointing = self.gradient_checkpointing
            self.tokenizer = TFAPTokenizer.load(output_folder)
            self.model = BertForActionClassification.from_pretrained(output_folder, config=self.config)
            found_classifier = self.model.load_classifier(output_folder)
        else:
            old_output_folder = output_folder
            output_folder = "dbmdz/bert-base-german-uncased"
            print("Path {} is not exist. Now, {} will be loaded.".format(old_output_folder, output_folder))
            self.config = BertConfig.from_pretrained(output_folder)
            self.config.num_labels = num_labels
            self.config.cls_model_name = cls_model_name
            self.config.lstm_out_features = lstm_out_features
            self.transformers_nhid = transformers_nhid
            self.config.fusion_way = fusion_way
            self.config.use_fea_encoder = use_fea_encoder
            self.base_tokenizer = BertTokenizer.from_pretrained(output_folder)
            self.config.gradient_checkpointing = self.gradient_checkpointing
            self.tokenizer = TFAPTokenizer(self.base_tokenizer, output_folder, special_prefix = "[nerlabels]")
            self.model = BertForActionClassification.from_pretrained(output_folder, config=self.config)

        return found_classifier
        

    def save_bert(self, output_folder):
        cur_model = self.model
        if isinstance(cur_model, torch.nn.DataParallel):
            cur_model = cur_model.module
        cur_model.save_classifier(output_folder)
        self.tokenizer.output_folder=output_folder
        TFAPTokenizer.save(self.tokenizer)
        cur_model.save_pretrained(output_folder)
        self.config.save_pretrained(output_folder)

    def save_german_bert(self, output_folder):
        cur_model = self.model
        if isinstance(cur_model, torch.nn.DataParallel):
            cur_model = cur_model.module
        cur_model.save_classifier(output_folder)
        self.tokenizer.output_folder=output_folder
        TFAPTokenizer.save(self.tokenizer)
        cur_model.save_pretrained(output_folder)
        self.config.save_pretrained(output_folder)


def get_select_mask(action_idx, none_action_idx, ig_idx, n_nearest = 5, rnd = 0.1):
    ig_mask = np.array(action_idx) < 0

    select_mask1 = np.random.rand(*action_idx.shape) <= rnd

    select_mask2 = np.zeros_like(action_idx)

    n_nearest = n_nearest
    for ith in range(len(action_idx)):
        for jth in range(len(action_idx[ith])):
            if action_idx[ith][jth] != ig_idx and action_idx[ith][jth] != none_action_idx:
                start = max([0, jth - n_nearest])
                select_mask2[ith, start:jth + n_nearest + 1] = 1

    select_mask = ((select_mask1 + select_mask2) > 0)
    select_mask[ig_mask] = False
    return select_mask



