import torch
import numpy as np
from copy import deepcopy

class Doc2Piece(torch.utils.data.Dataset):
    NewLineLabel = "[newline]"
    StrangeSymbols = "[strangesymbols]"
    NoneToken = "[nonetoken]"
    def __init__(self, sents, tokenizer=None, end_symbol=None, max_length=510, split_way ="both_symbol_length"):
        # split_way = both_symbol_length, length, symbol
        # end_symbol is not None and max_length is 0，document is splited by end_symbol.
        # both_symbol_length, document is splited by end_symbol and splited by specific max_length
        # sents = [instance,...,] where instance = [(token, label1, label2, ...), ...]
        if end_symbol is None:
            end_symbol = ["[NoEndSymbol]"]
        self.NewLine2Label = {"\n": Doc2Piece.NewLineLabel}

        new_end_symbol = []
        for x in end_symbol:
            if x in self.NewLine2Label:
                new_end_symbol.append(self.NewLine2Label[x])
            else:
                new_end_symbol.append(x)

        self.end_symbol = new_end_symbol
        self.max_length = max_length
        self.split_way = split_way

        # print("sents:", sents)
        # tokens and labels are aligned to follow the way the tokenizer use
        self.tokenzed_sents = self.subword_label_aligned(sents, tokenizer=tokenizer)
        # print("tokenzed_sents:", self.tokenzed_sents)
        self.piece_span = self.get_segmented_sapns()
        # print("piece_span:", self.piece_span)
        self.segmented_sents = self.cut_into_piece()
        # print("segmented_sents:", self.segmented_sents)

    def subword_label_aligned(self, sents, tokenizer=None):
        assert len(sents) > 0
        assert len(sents[0][0]) > 0
        if tokenizer is not None:
            # reimplemented this method. last elements of a item is a label
            tokenzed_sents = []
            for sent in sents:
                new_sent = []
                for item in sent:
                    item = list(item)
                    if item[0] in self.NewLine2Label:
                        item[0] = self.NewLine2Label[item[0]]
                    item_list = self.word2subword(item, tokenizer=tokenizer)
                    new_sent.extend(item_list)
                tokenzed_sents.append(new_sent)
        else:
            # reimplemented this method. last elements of a item is a label
            tokenzed_sents = []
            for sent in sents:
                new_sent = []
                for item in sent:
                    item = list(item)
                    if item[0] in self.NewLine2Label:
                        item[0] = self.NewLine2Label[item[0]]
                    label_mark = 1
                    new_item = tuple(list(item) + [label_mark])
                    new_sent.append(new_item)
                tokenzed_sents.append(new_sent)

        return tokenzed_sents

    def word2subword(self, item, tokenizer=None):
        # item = (word, label1, label2, ...)
        # Shenzhen => [Shen, ##zhen] => is_start_token = [1, 0]
        item_list = []
        if tokenizer is None:
            is_start_token = 1
            item_list.append(list(item) + [is_start_token])
        else:
            word = item[0]
            token_list = tokenizer.tokenize(word)
            if len(token_list) == 0: # '��' are strange symbols => item_list = [], then we need a token to replace '��'.
                is_start_token = 1
                item_list.append([self.StrangeSymbols] + list(item[1:]) + [is_start_token])
            else:
                is_start_token = 1
                for ith, token in enumerate(token_list):
                    if ith > 0:
                        is_start_token = 0
                    item_list.append([token] + list(item[1:]) + [is_start_token])
        return item_list

    def cal_length_cut_spans(self, seq):
        assert self.max_length > 0
        spans = []
        start = 0
        length = len(seq)
        while start < length:
            end = min([start + self.max_length - 1, length - 1])  # include last element
            spans.append((start, end))
            start = end + 1
        return spans

    def cal_symbol_cut_spans(self, seq):
        spans = []
        start = 0
        length = len(seq)
        es_len = len(self.end_symbol)
        while start < length:
            end = start
            while end < length:
                try:
                    if "".join(seq[end:end + es_len]) == "".join(self.end_symbol):
                        break
                except:
                    print("error")
                end += 1
            end = min([end + es_len - 1, length - 1])  # include last element
            spans.append((start, end))
            start = end + 1

        new_spans = []
        if "both_symbol_length" == self.split_way:
            assert self.max_length > 0
            for span in spans:
                start, t = span
                while start < t + 1:
                    end = min([start + self.max_length - 1, t])  # include last element
                    new_spans.append((start, end))
                    start = end + 1
        else:
            new_spans = spans

        return new_spans

    def get_segmented_sapns(self):
        """
        piece_span = [instance,...,] where instance=[span,...,...]，其中instance顺序和样本顺序一致
        span = (start_token_idx_of_segment, end_token_idx_of_segment)
        :return:
        """
        individual_list = Doc2Piece.unpackage(self.tokenzed_sents)
        piece_span = []
        # individual_list[0] is a token sequence
        for sent in individual_list[0]:
            if self.split_way == "length":
                piece_span_in_sent = self.cal_length_cut_spans(sent)
            else:
                piece_span_in_sent = self.cal_symbol_cut_spans(sent)
            piece_span.append(piece_span_in_sent)
        return piece_span

    def cut_into_piece(self):
        segmented_sents = Doc2Piece.get_piece(self.tokenzed_sents, self.piece_span)
        return segmented_sents

    def recover_from_spans(self, seq_list, piece_span):
        ith = 0
        new_seq_list = []
        for spans in piece_span:
            sent = []
            for span in spans:
                sent.extend(seq_list[ith])
                ith+=1
            new_seq_list.append(sent)

        return new_seq_list

    @staticmethod
    def get_instance_piece(seq, spans):
        seq_list = []
        for span in spans:
            s, t = span
            seq_list.append(seq[s:t+1])
        return seq_list

    @staticmethod
    def get_piece(seq_list, span_list):
        assert  len(seq_list) == len(span_list)
        new_seq = []
        for ith, piece_spans in enumerate(span_list):
            new_seq.extend(Doc2Piece.get_instance_piece(seq_list[ith], piece_spans))
        return new_seq

    @staticmethod
    def segmented2tokenized(seq_list, span_list, is_token=False):
        """
        the order of seq_list should be aligned to span_list
        Example input:
            seq_list = [[Shen], [##zhen]]
            span_list = [[(0,0), (1,1)]] # [(0,0), (1,1)] means the first the tokens in the same sentence
        Example output:
            new_seq = [[Shen, ##zhen]]

        :param seq_list:
        :param span_list:
        :return:
        """
        assert  len(seq_list) == sum([len(x) for x in span_list])
        new_seq = []
        cnt = 0
        for ith, piece_spans in enumerate(span_list):
            a_seq = []
            for jth in range(len(piece_spans)):
                a_seq.extend(seq_list[cnt])
                cnt += 1
            new_seq.append(a_seq)

        if is_token:
            new_seq = [["\n" if x == Doc2Piece.NewLineLabel else x for x in seq]for seq in new_seq]
        return new_seq

    @staticmethod
    def tokenized2original(seq_list, label_mask):
        assert len(seq_list) == len(label_mask)
        new_seq = []
        for ith, seq in enumerate(seq_list):
            a_seq = []
            for jth in range(len(seq)):
                if label_mask[ith][jth]:
                    a_seq.append(seq_list[ith][jth])
            new_seq.append(a_seq)
        return new_seq

    @staticmethod
    def package(individual_list):
        """
        individual_list => sents
        :param individual_list: individual_list = [token_list, label1_list, label2_list]
        :return: sents = [instance,...,] where instance = [(token, label1, label2, ...), ...]
        :Example:
             list[token_sents, or tag_sents, or..]
            a = [[1], [1, 2]]
            b = [["a"], ["b", "c"]]
            c = [["ab"], ["ab", "bc"]]
            package([a,b,c])
            output: [[(1, 'a', 'ab')], [(1, 'b', 'ab'), (2, 'c', 'bc')]]"""
        sents = []
        for alist in zip(*individual_list):
            sent = list(zip(*alist))
            sents.append(sent)
        return sents

    @staticmethod
    def unpackage(sents):
        """
        sents => individual_list
        :param sents: sents = [instance,...,] where instance = [(token, label1, label2, ...), ...]
        :return: individual_list = [token_list, label1_list, label2_list]
        :Example:
            sents = [[(1, 'a', 'ab')], [(1, 'b', 'ab'), (2, 'c', 'bc')]]
            individual_list = unpackage(sents)
            individual_list is [((1,), (1, 2)), (('a',), ('b', 'c')), (('ab',), ('ab', 'bc'))]
        """
        instance_list = []
        for sent in sents:
            instance_list.append(list(zip(*sent)))
        individual_list = list(zip(*instance_list))

        return individual_list

    def __getitem__(self, index):
        sent = self.segmented_sents[index]
        return sent

    def __len__(self):
        return len(self.segmented_sents)
