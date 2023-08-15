import pickle
import os

class TokenizerWrapper(object):
    def __init__(self, tokenizer, output_folder = "", special_prefix = "[SPECIAL]"):
        """
        copy the vocab form vocab and special_tokens_map from tokenizer
        and then designed own way of tokenization
        :param tokenizer:
        """
        self.tokenizer = tokenizer
        self.max_len_single_sentence = tokenizer.max_len_single_sentence # without consider <s> and </s>
        self.token2idx = tokenizer.get_vocab()
        self.idx2token = {self.token2idx[k]:k for k in self.token2idx}
        self.special_prefix = special_prefix
        self.new_added = set()
        self.output_folder = output_folder
        ## select part of tokens for new tokens in future
        if "bert" in self.tokenizer.name_or_path:
            self.unk_token = '[UNK]'
            self.pad_token = '[PAD]'
            self.start_token = '[CLS]'
            self.end_token = '[SEP]'
            served_as_tokens = [k for k in tokenizer.get_vocab() if "unused" in k]
            self.assumed_unsed = sorted(list(set(served_as_tokens)))
        elif "longformer" in self.tokenizer.name_or_path:
            self.unk_token = '<unk>'
            self.pad_token = '<pad>'
            self.start_token = '<s>'
            self.end_token = '</s>'
            served_as_tokens = [k.replace("Ġ", "") for k in tokenizer.get_vocab() if "Ġ" in k]
            self.assumed_unsed = sorted(["Ġ" + x for x in set(served_as_tokens).intersection((set(tokenizer.get_vocab().keys())))], reverse=True)
        else:
            raise NotImplementedError("SimpleTokenizer did not implemented for {}".format(self.tokenizer.name_or_path))

    def __call__(self, *args, **kwargs):
        return self.get_tokenization_input(*args, **kwargs)

    def add_tokens(self, tokens, are_special = False, special_prefix = None):
        added_tokens = set()
        availible = len(self.assumed_unsed)
        cnt = 0
        for token in tokens:
            if are_special:
                if special_prefix is None:
                    token = self.special_prefix + token
                else:
                    token = special_prefix + token
            if token not in self.token2idx:
                if availible <= 0:
                    print("{} tokens have been added".format(cnt))
                    print("not enough available location")
                    return

                cnt += 1

                old_token = self.assumed_unsed[0]
                idx = self.token2idx[self.assumed_unsed[0]]

                del self.token2idx[old_token]
                del self.assumed_unsed[0]
                added_tokens.add(token)
                self.new_added.add(token)
                self.tokenizer.add_tokens(token) # just for tokenize method to use

                self.token2idx[token] = idx
                self.idx2token[idx] = token

        print("{} tokens have been added: {}".format(cnt, added_tokens))

    def convert_ids_to_tokens(self, idx_sent):
        token_sent = [self.idx2token[x] if x in self.idx2token else self.unk_token for x in idx_sent]
        return token_sent

    def convert_tokens_to_ids(self, token_sent):
        idx_sent = [self.token2idx[x] if x in self.token2idx else self.token2idx[self.unk_token] for x in token_sent]
        return idx_sent

    def get_tokenization_input(self, token_sents, are_special_token = False, special_prefix = None):
        # in bert, '[CLS]', 'dog', '[SEP]', '[PAD]', ...
        # in longformer, '<s>', 'dog', '</s>', '<pad>', ...
        max_length = max([len(x) for x in token_sents])
        max_length = min([max_length, self.max_len_single_sentence]) # max_len_single_sentence did not consider start and end symbol

        idx_token_sents = []
        mask = []
        length = [] # include start and end symbol
        for sent in token_sents:
            sent = list(sent)
            new_sent = sent[:max_length]
            n_token = len(new_sent) # sent length exclude padding symbol and start and end symbol
            padding_length = max_length - n_token
            mask_len = n_token + 2
            if are_special_token:
                if special_prefix is None:
                    new_sent = [self.special_prefix + token for token in new_sent]
                else:
                    new_sent = [special_prefix + token for token in new_sent]
            new_sent = [self.start_token] + new_sent + [self.end_token] + [self.pad_token] * padding_length
            idx_token_sents.append(self.convert_tokens_to_ids(new_sent))
            mask.append([1] * mask_len + [0] * padding_length)
            length.append(n_token + 2)
        return {"input_ids":idx_token_sents, "attention_mask":mask, "length":length}

    def tokenize(self, word):
        return self.tokenizer.tokenize(word)

    @staticmethod
    def save(tokenizer):
        path = os.path.join(tokenizer.output_folder, "tokenizer.pkl")
        with open(path, "wb") as file:
            pickle.dump(tokenizer, file)

    @staticmethod
    def load(folder):
        path = os.path.join(folder, "tokenizer.pkl")
        with open(path, "rb") as file:
            return pickle.load(file)
