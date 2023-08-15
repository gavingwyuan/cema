import numpy as np

from transformers import AdamW
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader
import os
from torch.nn import CrossEntropyLoss

from model.Doc2Piece import Doc2Piece
from model.CustomizedTokenizer import TokenizerWrapper

from transformers import BertTokenizer, BertConfig, BertForTokenClassification
from model.base_model.BertForNER import BertForNER
import hiddenlayer as hl
from utils.utils import compute_seq_entropy


class TFNERDataset(Doc2Piece):
    def __init__(self, sents, tokenizer = None, end_symbol=[None], max_length=510, split_way = "both_symbol_length"):
        super().__init__(sents, tokenizer = tokenizer, end_symbol=end_symbol, max_length=max_length, split_way = split_way)

    @staticmethod
    def get_tf_ner_input(sents):
        # from
        ## sents = [instance,...,...], instance = [(token, none label or true label),...,...]
        ## pred_label = [instance,...,...], instance = [label,...,...]
        # to
        ## do nothing
        sents4bertner = sents

        return sents4bertner

class TFNERTokenizer(TokenizerWrapper):
    def __init__(self, tokenizer, output_folder=""):
        """
        add labels tokenization base on SimpleTokenizer
        :param tokenizer:
        :param output_folder:
        """
        super(TFNERTokenizer, self).__init__(tokenizer, output_folder=output_folder)
        self.idx_pad_ner_label = CrossEntropyLoss().ignore_index

    def encode_labels(self, label_sents, label_mask, label2idx):
        # if label for [Shen, ##zhen] is [S-LOC, S-LOC] AND label_mask = [1, 0]
        # Then, idx_lbaels = [[label2idx[S-LOC], idx_pad_ner_label]
        # when loss is calculating the loss for idx_pad_ner_label will be ignored
        # label mask of the labels assigned to padding, [CLS], and [SEP] are 0.
        max_length = max([len(x) for x in label_sents])
        max_length = min([max_length, self.max_len_single_sentence]) # max_len_single_sentence did not consider start and end symbol
        idx_labels = []
        for ith, sent in enumerate(label_sents):
            sent = list(sent)
            sent = sent[:max_length]
            sent = [label2idx[label] if label_mask[ith][jth] else self.idx_pad_ner_label for jth, label in enumerate(sent) ]
            sent = [self.idx_pad_ner_label] + sent + [self.idx_pad_ner_label]+ [self.idx_pad_ner_label] * (max_length - len(sent))
            idx_labels.append(sent)
        return idx_labels

    def encode_label_mask(self, label_mask):
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

    def decode_labels(self, seq_list, label_mask, idx2label = None):
        assert len(seq_list) == len(label_mask)
        if idx2label is None:
            label_list = [[idx for idx in seq[1:1 + len(label_mask[ith])]] for ith, seq in
                          enumerate(seq_list)]
        else:
            label_list = [[idx2label[idx] for idx in seq[1:1+len(label_mask[ith])]] for ith, seq in enumerate(seq_list)]
        return label_list

class TFNERDataProcessor(object):
    def __init__(self, sents, label2idx, tokenizer, end_symbol=None, max_length=0):
        """
        maintain the relationship between self.tokenizer and self.german_coarse_doc
            (1) self.tokenizer.tokenize method is used in self.german_coarse_doc to tokenize word.
            Then, tokenized words and labels will be aligned in the german_coarse_doc.
            (2) self.tokenizer will be used to encode (e.g. get input_dix, attention_mask) and decode input
            (3) dataloaser will split encoded input as batch
        :param sents:
        :param label2idx:
        :param tokenizer:
        :param end_symbol:
        :param max_length:
        """
        self.label2idx = label2idx
        self.idx2label = {self.label2idx[k]: k for k in self.label2idx}
        self.tokenizer = tokenizer
        self.dataset = TFNERDataset(sents, tokenizer = self.tokenizer, end_symbol=end_symbol, max_length=max_length)
        self.collate_fn = lambda batch: TFNERDataProcessor.get_encode_data(batch, tokenizer=self.tokenizer,
                                                                        label2idx=self.label2idx)

    @staticmethod
    def get_encoded_x(token_sents, tokenizer):

        token_input = tokenizer(
            token_sents
        )

        token_input_ids = token_input['input_ids']
        token_attention_mask = token_input['attention_mask']

        return torch.LongTensor(token_input_ids), torch.LongTensor(token_attention_mask)

    @staticmethod
    def get_encoded_y(label_sents, label_mask, label2idx, tokenizer):
        idx_label_list = tokenizer.encode_labels(label_sents, label_mask, label2idx)
        return torch.LongTensor(idx_label_list)

    @staticmethod
    def get_encoded_label_mask(label_mask, tokenizer):
        label_mask = tokenizer.encode_label_mask(label_mask)
        return torch.BoolTensor(label_mask)

    @staticmethod
    def get_encode_data(batch, tokenizer = None, label2idx = None):
        data = TFNERDataset.unpackage(batch)
        if len(data) < 1:
            raise NotImplementedError("error in get_encode_data")

        token_sents = data[0]
        token_input_ids, token_attention_mask = TFNERDataProcessor.get_encoded_x(token_sents, tokenizer)

        label_mask = data[-1]

        if len(data) >= 3: # token_seq, ner_label, label_mask
            labels = data[1] # ner_label
            idx_labels = TFNERDataProcessor.get_encoded_y(labels, label_mask, label2idx, tokenizer)

            # return token_input_ids, token_attention_mask, idx_labels, label_mask
            return {"token_input_ids":token_input_ids, "token_attention_mask":token_attention_mask,
                    "idx_labels":idx_labels, "label_mask":label_mask}

        # return token_input_ids, token_attention_mask, label_mask
        return {"token_input_ids":token_input_ids, "token_attention_mask":token_attention_mask,
                "label_mask":label_mask}

    def get_dataloader4training(self, batch_size, shuffle = True):
        trainData = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn) # 处理成多个batch的形式
        return trainData

    def get_dataloader4val(self, batch_size):
        valData = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn) # 处理成多个batch的形式
        return valData

    def get_decoded_y(self, idx_label_list, label_mask, need_idx2labels = True):
        if need_idx2labels:
            idx2label = self.idx2label
        else:
            idx2label = None
        label_list = self.tokenizer.decode_labels(idx_label_list, label_mask, idx2label)
        return label_list

    def recover_from_encode_y(self, idx_label_list, label_mask, need_idx2labels = True):
        # align the predicted labels to original input shape
        # label_mask = [instance 1, instance 2, ..] where the length of the instance i is same with the no. of token in sequence i
        y_segmented = self.get_decoded_y(idx_label_list, label_mask, need_idx2labels = need_idx2labels)
        y_tokenized = self.dataset.segmented2tokenized(y_segmented, self.dataset.piece_span, is_token=False)
        mask_tokenized = self.dataset.segmented2tokenized(label_mask, self.dataset.piece_span, is_token=False)
        y_original = self.dataset.tokenized2original(y_tokenized, mask_tokenized)
        return y_original

class TFword2Embedding(object):
    @staticmethod
    def get_flat_token_embedding(hidden_states, label_mask, last_k_layers=4, previous_remain_embed=[], is_end=False):
        # if last word has multiple tokens, the last several not input in this function, and will process next time, you could set is_end is false and put it as previous_remain_embed
        # hidden_states is a tuple, [last layer, first layer, second layer, ...]
        token_embedding = torch.stack(hidden_states[-last_k_layers:]).mean(
            0)  # no. of instances, no. of tokens, no. of dimension of embedding
        # token_embedding = [sent, sent, ...] in shape no. of sent, no. of token, the size of embedding
        # sent = [token, token], in a sent, if label mask is [1, 0, 1, 0, 0, 0], then token assigned wiht first two flags 1, 0 will belong to the first word.
        # next 1, 0, 0, 0 indicated the tokens belong to the second word.
        # token = token embedding

        flat_mask = [mask for sent in label_mask for mask in sent]

        ## error
        # flat_ebd = [ebd.cpu() for sent in token_embedding for ebd in sent][
        #            1:len(flat_mask) + 1]  # remove start symbol, end symbol, and padding symbol

        ## correct
        flat_ebd = []
        for i in range(len(label_mask)):
            flat_ebd.extend(token_embedding[i][1:len(label_mask[i]) + 1]) # remove start symbol, end symbol, and padding symbol

        flat_word_embedding = []
        remain_embedding = [x for x in previous_remain_embed]
        for ith in range(len(flat_mask)):
            if flat_mask[ith] == 1:
                if len(remain_embedding) > 0:
                    word_ebd = torch.stack(remain_embedding).mean(0).cpu().tolist()
                    flat_word_embedding.append(word_ebd)
                    remain_embedding = []
            remain_embedding.append(flat_ebd[ith])

        # print("no. of words:", len(flat_word_embedding))

        if is_end:
            word_ebd = torch.stack(remain_embedding).mean(0).cpu().tolist()
            flat_word_embedding.append(word_ebd)
            remain_embedding = []

        return flat_word_embedding, remain_embedding

    @staticmethod
    def get_word_embedding(flag_word_ebd, label_mask):
        ith = 0
        all_word_ebd = []
        for sent in label_mask:
            a_sent = []
            for mask in sent:
                if mask == 1:
                    a_sent.append(flag_word_ebd[ith])
                    ith += 1
            all_word_ebd.append(a_sent)
        return all_word_ebd

    @staticmethod
    def summary_word_embedding(sent_word, sent_word_ebd):
        w2v_sum = {}
        w2v_cnt = {}
        for i in range(len(sent_word_ebd)):
            for j in range(len(sent_word_ebd[i])):
                word = sent_word[i][j]
                if word in w2v_cnt:
                    w2v_sum[word] += np.array(sent_word_ebd[i][j])
                    w2v_cnt[word] += 1
                else:
                    w2v_sum[word] = np.array(sent_word_ebd[i][j])
                    w2v_cnt[word] = 1

        for key in w2v_cnt:
            w2v_sum[key] = list(w2v_sum[key] / w2v_cnt[key])
        w2v_mean = w2v_sum
        return w2v_mean

class TransformerNER(object):
    def __init__(self,
                 label2idx,
                 learning_rate,
                 epochs,
                 batch_size,
                 device="cuda",
                 output_folder='NER',
                 base_model_name = "bert-base-uncased",
                 end_symbol = None,
                 max_length = 510,
                 gradient_checkpointing=False,
                 ):

        self.gradient_checkpointing = gradient_checkpointing

        self.output_folder = output_folder  # 模型存储的文件夹
        self.base_model_name = base_model_name

        self.max_length = max_length
        self.end_symbol = end_symbol

        self.lr = learning_rate  # 学习率
        self.epochs = epochs

        self.batch_size = batch_size  # 训练集的batch_size

        self.label2idx = label2idx
        self.idx2label = {self.label2idx[k]:k for k in self.label2idx}

        self.device = device  # "cpu", "cuda", "cuda:0"
        # this 4 parameter will be initialized in self.load_model
        self.config = None
        self.base_tokenizer = None
        self.model = None
        self.tokenizer = None
        self.load_model()
        self.predicted_batch_size = 15 * self.batch_size
        self.batch_size_embedding = 6 * self.batch_size
        self.batch_size_probability = 10 * self.batch_size

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
        nerdp = TFNERDataProcessor(sents, self.label2idx, self.tokenizer, max_length = self.max_length, end_symbol = self.end_symbol)
        trainData = nerdp.get_dataloader4training(self.batch_size)
        return trainData

    def train_on_batch(self, batch, optimizer = None, has_set_train_state = False):
        if not has_set_train_state:
            self.model = self.model.to(self.device)
            self.model.train()

        if optimizer is None:
            optimizer = AdamW(self.model.parameters(), lr=self.lr)

        input_ids = batch["token_input_ids"].to(self.device)
        attention_mask = batch["token_attention_mask"].to(self.device)
        idx_labels = batch["idx_labels"].to(self.device)
        label_mask = batch["label_mask"]

        # label_mask = batch[3]
        inputs = {"input_ids": input_ids,
                  "attention_mask": attention_mask,
                  "labels": idx_labels}
        optimizer.zero_grad()

        outputs = self.get_output(inputs)

        loss, logits = outputs.loss, outputs.logits
        step_loss = loss.item()
        logits = logits.detach().cpu().numpy()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()

        return step_loss, logits

    def train(self, sents, record_training = True, log_folder="", logger = None, save_model = True):

        if logger is None:
            class logger():
                info = None
            logger.info = print

        # labels is int sequece
        epoch_history, step_history = None, None

        apdp = TFNERDataProcessor(sents, self.label2idx, self.tokenizer, max_length = self.max_length, end_symbol = self.end_symbol)


        trainData = apdp.get_dataloader4training(self.batch_size)

        total_epoch = self.epochs

        if log_folder != "":
            epoch_history = hl.History()
            step_history = hl.History()

        self.model = self.model.to(self.device)

        optimizer = AdamW(self.model.parameters(), lr=self.lr)  # 后面看看需不需要改参数

        total_steps = len(trainData) * self.epochs

        interval = int(total_steps/100)
        interval = max([interval, 1])
        interval = interval * 10

        cur_steps = 0 # 记录已经经过的step数
        total_loss = 0
        for epoch in range(self.epochs):
            epoch_loss = 0
            logger.info('epoch: {}'.format(epoch))
            # 训练
            for step, batch in enumerate(trainData):
                self.model.train()
                input_ids = batch["token_input_ids"].to(self.device)
                attention_mask = batch["token_attention_mask"].to(self.device)
                idx_labels = batch["idx_labels"].to(self.device)
                label_mask = batch["label_mask"]

                # label_mask = batch[3]
                inputs = {"input_ids": input_ids,
                          "attention_mask": attention_mask,
                          "labels": idx_labels}

                optimizer.zero_grad()

                outputs = self.get_output(inputs)

                loss, logits = outputs.loss, outputs.logits
                total_loss += loss.item()
                epoch_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                cur_steps += 1
                if record_training and cur_steps % interval == 0:
                    self.model.eval()
                    logits = logits.detach().cpu().numpy()
                    preds_ids = logits.argmax(axis=2).tolist()
                    label_ids = idx_labels.data.cpu().tolist()
                    format_preds_ids = apdp.get_decoded_y(preds_ids, label_mask, need_idx2labels=False)
                    format_label_ids = apdp.get_decoded_y(label_ids, label_mask, need_idx2labels=False)
                    train_accuracy = self.flat_accuracy(format_label_ids, format_preds_ids, label_mask)
                    logger.info('tagger - epoch {}/{}, current total step: {}/{}, step in an epoch: {}'.format(epoch+1, total_epoch, cur_steps, total_steps, step))
                    logger.info('tagger - Train Accuracy: {}'.format(train_accuracy))
                    logger.info('tagger - total_loss/cur_steps: {}'.format(total_loss/cur_steps))
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

    def test(self, sents, logger=None):

        if logger is None:
            selected_print = print
        else:
            selected_print = logger.info

        # tokens, labels
        labels = [[x[1] for x in instance] for instance in sents]
        # 输出模型的召回率、准确率、f1-score
        preds = self.predict(sents,output_attentions=False)
        f1_score = self.cal_token_level_f1(labels, preds)
        print("**tagger token level f1**: {}".format(f1_score))
        return f1_score

    def predict(self, sents, output_attentions=False):
        end_symbol = self.end_symbol
        max_length = self.max_length
        batch_size = self.predicted_batch_size

        tfner_dp = TFNERDataProcessor( sents, self.label2idx, self.tokenizer,max_length = max_length, end_symbol = end_symbol)
        valData = tfner_dp.get_dataloader4val(batch_size)

        model = self.model.to(self.device)
        model.eval()
        all_pred_idx_labels = []
        attentions = []
        all_label_mask = []
        for i, batch in enumerate(valData):
            input_ids = batch["token_input_ids"].to(self.device)
            attention_mask = batch["token_attention_mask"].to(self.device)
            label_mask = batch["label_mask"]
            token_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            all_label_mask.extend(label_mask)
            with torch.no_grad():
                # outputs = model(**token_inputs)  # 输出loss 和 每个分类对应的输出
                outputs = self.get_output(token_inputs)
                logits = outputs[0] # logit softmax后才是预测是对应分类的概率
                logits = logits.detach().cpu().numpy()
                all_pred_idx_labels += list(np.argmax(logits, axis=2)) #
                if output_attentions:
                    _attentions = outputs.attentions
                    attentions.append([item.detach().cpu().numpy() for item in _attentions])

            torch.cuda.empty_cache()

        # 还原会原本的长度
        recovered_pred_labels = tfner_dp.recover_from_encode_y(all_pred_idx_labels, all_label_mask)

        if output_attentions:
            return recovered_pred_labels, all_label_mask, attentions
        else:
            return recovered_pred_labels

    def get_output(self, inputs):
        cur_model = self.model

        if isinstance(cur_model, torch.torch.nn.DataParallel):
            if len(inputs["input_ids"]) < torch.cuda.device_count():
                    cur_model = cur_model.module
            outputs = cur_model(**inputs, reduction="sum")  # 输出loss 和 每个分类对应的输出，softmax后才是预测是对应分类的概率
            if hasattr(outputs.loss, "sum"):
                # n_tokens = sum([len([x for x in row if x != CrossEntropyLoss().ignore_index]) for row in inputs['labels']])
                outputs.loss = outputs.loss.sum()
            if 'labels' in inputs:
                n_tokens = (inputs['labels'] != CrossEntropyLoss().ignore_index).sum()
                outputs.loss = outputs.loss/n_tokens
        else:
            outputs = cur_model(**inputs)  # 输出loss 和 每个分类对应的输出，softmax后才是预测是对应分类的概率

        return outputs


    def _get_probability(self, tfner_dp, device, batch_size):
        # to obtain the probability of assigning each label for each tokens in the sequence
        # recovered_probability is N*LENGTH*NUMBER_CLUSS
        valData = tfner_dp.get_dataloader4val(batch_size)
        model = self.model.to(device)
        model.eval()
        all_probability = []
        all_label_mask = []
        for i, batch in enumerate(valData):
            input_ids = batch["token_input_ids"].to(self.device)
            attention_mask = batch["token_attention_mask"].to(self.device)
            label_mask = batch["label_mask"]
            token_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            all_label_mask.extend(label_mask)  # in validation batch[2] is label_mask, in training batch[2] is idx_label, batch[3] is label_mask
            with torch.no_grad():
                # outputs = model(**token_inputs)  # 输出loss 和 每个分类对应的输出
                outputs = self.get_output(token_inputs)
                logits = outputs[0] # logit softmax后才是预测是对应分类的概率
                probability = torch.nn.functional.softmax(logits, dim=2)
                all_probability += probability.tolist()

            torch.cuda.empty_cache()

        # 还原会原本的长度
        recovered_probability = tfner_dp.recover_from_encode_y(all_probability, all_label_mask, need_idx2labels=False)
        return np.array(recovered_probability)

    def get_probability_core(self, sents, obtain={"marginal_probability"}):
        end_symbol = self.end_symbol
        batch_size = self.batch_size_probability
        max_length = self.max_length
        device = self.device

        tfner_dp = TFNERDataProcessor(sents, self.label2idx, self.tokenizer,max_length = max_length, end_symbol = end_symbol)
        probability = self._get_probability(tfner_dp, batch_size=batch_size, device=device)

        return_dict = {"marginal_probability":probability}

        if "entropy" in obtain:
            entropy = [compute_seq_entropy(np.array(item)) for item in probability]
            return_dict["entropy"] = entropy

        if "log_confidence" in obtain or "mnlp" in obtain or "confidence" in obtain:
            max_probability = [[max(probabilities) for probabilities in item] for item in probability]
            log_confidence = [sum([np.log(probability) if probability > 0 else 0 for probability in item]) for item in max_probability]
            confidence = [np.e ** log_probability for log_probability in log_confidence]
            length = [len(item) for item in probability]
            mnlp = [log_probability / length[ith] for ith, log_probability in enumerate(log_confidence)]

            return_dict["max_probability"] = max_probability
            return_dict["log_confidence"] = log_confidence
            return_dict["confidence"] = confidence
            return_dict["mnlp"] = mnlp

        return return_dict

    def get_probability(self, sents, obtain={"marginal_probability"}):
        batch_size = self.batch_size_probability
        max_length = self.max_length
        step_size = int(20*batch_size*max_length/len(sents[-1]))
        step_size = max([1, step_size])
        return_dict = {}

        start = 0
        while start < len(sents):
            batch_sent = sents[start:(start + step_size)]
            probabilities_dict = self.get_probability_core(batch_sent, obtain=obtain)
            if "marginal_probability" not in obtain and "marginal_probability" in probabilities_dict:
                del probabilities_dict["marginal_probability"]

            for key in probabilities_dict:
                if key not in return_dict:
                    return_dict[key] = probabilities_dict[key]
                else:
                    if isinstance(probabilities_dict[key], np.ndarray):
                        return_dict[key] = np.concatenate((return_dict[key], probabilities_dict[key]), axis=0)
                    elif isinstance(probabilities_dict[key], list):
                        return_dict[key].extend(probabilities_dict[key])
                    else:
                        raise NotImplementedError

            start += len(batch_sent)

        return return_dict

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

    def get_word_embedding(self, sents):
        end_symbol = self.end_symbol
        max_length = self.max_length
        batch_size = self.batch_size_embedding

        tfner_dp = TFNERDataProcessor(sents, self.label2idx, self.tokenizer,max_length = max_length, end_symbol = end_symbol)
        valData = tfner_dp.get_dataloader4val(batch_size)

        batch_num = len(valData)
        model = self.model.to(self.device)
        model.eval()
        all_label_mask = []
        remain_embedding = []
        all_flat_word_embedding = []
        for i, batch in enumerate(valData):
            input_ids = batch["token_input_ids"].to(self.device)
            attention_mask = batch["token_attention_mask"].to(self.device)
            label_mask = batch["label_mask"]
            token_inputs = {"input_ids": input_ids, "attention_mask": attention_mask,
                            "return_dict":True, "output_hidden_states":True}
            all_label_mask.extend(label_mask)  # in validation batch[2] is label_mask, in training batch[2] is idx_label, batch[3] is label_mask

            with torch.no_grad():
                # outputs = model(**token_inputs)  # 输出loss 和 每个分类对应的输出
                outputs = self.get_output(token_inputs)
                hidden_states = outputs['hidden_states']

                if batch_num == i + 1:
                    is_end = True
                else:
                    is_end = False
                # 从这博文的结果来看，last_k_layers差别不是特别大。实验中统一取最后一层
                # http://jalammar.github.io/illustrated-bert/
                flat_word_embedding, remain_embedding = TFword2Embedding.get_flat_token_embedding(hidden_states, label_mask,
                                                                    previous_remain_embed=remain_embedding, last_k_layers = 1, is_end = is_end)
                all_flat_word_embedding.extend(flat_word_embedding)
            torch.cuda.empty_cache()

        label_mask = [[item[2] for item in sent] for sent in tfner_dp.dataset.tokenzed_sents]
        sent_word = [[item[0] for item in sent] for sent in sents]
        word_embedding_in_sentence = TFword2Embedding.get_word_embedding(all_flat_word_embedding, label_mask)
        w2v = TFword2Embedding.summary_word_embedding(sent_word, word_embedding_in_sentence)
        return word_embedding_in_sentence, w2v

    def load_model(self, input_folder = None):
        num_labels = len(self.label2idx)
        if input_folder is None: input_folder = self.output_folder
        if self.base_model_name == "bert-base-uncased":
            self.load_bert(input_folder, num_labels)
        elif self.base_model_name == "dbmdz/bert-base-german-cased":
            self.load_german_bert(input_folder, num_labels)

        # max_position_embeddings containing start and end position, model_max_length exclude start and end position
        self.tokenizer.model_max_length = self.model.config.max_position_embeddings - 2
        self.tokenizer.add_tokens([Doc2Piece.NewLineLabel])
        self.tokenizer.add_tokens([Doc2Piece.StrangeSymbols])
        self.tokenizer.add_tokens([Doc2Piece.NoneToken])
        self.model = self.model.to(self.device)

        n_gpu = torch.cuda.device_count()
        if n_gpu > 0 and "cuda" in self.device:
            self.model = self.model.cuda()
            # if n_gpu > 1:
            #     self.model = torch.nn.DataParallel(self.model)

    def save_model(self, output_folder = None):
        if output_folder is None: output_folder = self.output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        if self.base_model_name == "bert-base-uncased":
            self.save_bert(output_folder)
        elif self.base_model_name == "dbmdz/bert-base-german-cased":
            self.save_german_bert(output_folder)

    def load_bert(self, output_folder, num_labels):
        if os.path.exists(os.path.join(output_folder, "tokenizer.pkl")):
            output_folder = self.output_folder
            self.config = BertConfig.from_pretrained(output_folder)
            self.config.num_labels = num_labels  # 设置分类模型的输出个数
            self.model = BertForNER.from_pretrained(output_folder, config=self.config)
            self.tokenizer = TFNERTokenizer.load(output_folder)
            self.model.load_classifier(output_folder)
        else:
            old_output_folder = output_folder
            output_folder = 'bert-base-uncased'
            print("Path {} is not exist. Now, {} will be loaded.".format(old_output_folder, output_folder))
            self.config = BertConfig.from_pretrained(output_folder)
            self.config.num_labels = num_labels  # 设置分类模型的输出个数
            self.base_tokenizer = BertTokenizer.from_pretrained(output_folder)  # 加载分词模型
            self.model = BertForNER.from_pretrained(output_folder, config=self.config)
            self.tokenizer = TFNERTokenizer(self.base_tokenizer, output_folder=output_folder)

    def load_german_bert(self, output_folder, num_labels):
        if os.path.exists(os.path.join(output_folder, "tokenizer.pkl")):
            output_folder = self.output_folder
            self.config = BertConfig.from_pretrained(output_folder)
            self.config.num_labels = num_labels  # 设置分类模型的输出个数
            self.model = BertForNER.from_pretrained(output_folder, config=self.config)
            self.tokenizer = TFNERTokenizer.load(output_folder)
            self.model.load_classifier(output_folder)
        else:
            old_output_folder = output_folder
            output_folder = "dbmdz/bert-base-german-uncased"
            print("Path {} is not exist. Now, {} will be loaded.".format(old_output_folder, output_folder))
            self.config = BertConfig.from_pretrained(output_folder)
            self.config.num_labels = num_labels  # 设置分类模型的输出个数
            self.base_tokenizer = BertTokenizer.from_pretrained(output_folder)  # 加载分词模型
            self.model = BertForNER.from_pretrained(output_folder, config=self.config)
            self.tokenizer = TFNERTokenizer(self.base_tokenizer, output_folder=output_folder)

    def save_bert(self, output_folder):
        self.model.save_classifier(output_folder)
        self.tokenizer.output_folder = output_folder
        TFNERTokenizer.save(self.tokenizer)
        self.model.save_pretrained(output_folder)
        self.config.save_pretrained(output_folder)

    def save_german_bert(self, output_folder):
        self.model.save_classifier(output_folder)
        self.tokenizer.output_folder = output_folder
        TFNERTokenizer.save(self.tokenizer)
        self.model.save_pretrained(output_folder)
        self.config.save_pretrained(output_folder)

    def get_entity_embedding(self, sents, entities, step_size = 100):
        max_idx_list = []
        for alist in entities:
            idx_list = [0]
            for ety in alist:
                idx_list.extend(list(ety)[1:])
            max_idx_list.append(max(idx_list) + 250)

        new_sents = []
        for ith, asent in enumerate(sents):
            new_sents.append(asent[:max_idx_list[ith]])
        sents = new_sents

        entity_ebd_dict= {}
        start = 0
        cur_idx = 0
        while start < len(sents):
            batch_sent = sents[start:(start + step_size)]
            batch_entities = entities[start:(start + step_size)]
            word_embedding_in_sentence, _ = self.get_word_embedding(batch_sent)
            del _
            for instance in batch_entities:
                ith = cur_idx - start
                for entity in instance:
                    (label, s, t) = entity
                    embedding_list = word_embedding_in_sentence[ith][s:t+1]
                    entity_ebd = np.mean(embedding_list, axis = 0)
                    key = (cur_idx, s, t)
                    entity_ebd_dict[key] = entity_ebd
                cur_idx += 1
            del batch_entities
            start += len(batch_sent)
        return entity_ebd_dict

    def get_predictions(self, sent):
        # input is a sent
        return_dict = self.get_probability([sent], obtain={"marginal_probability"})
        y_marginals = return_dict["marginal_probability"][0]
        return y_marginals

    def get_confidence(self, sent):
        # input is a sent
        return_dict = self.get_probability([sent], obtain={"mnlp"})
        confidence = return_dict["mnlp"][0]
        return [confidence]

    def get_uncertainty(self, sent):
        # input is a sent
        return_dict = self.get_probability([sent], obtain={"entropy"})
        return return_dict["entropy"][0]

    def sent2tokens(self, sent):
        # return [token for token, label in sent]
        return [item[0] for item in sent]



