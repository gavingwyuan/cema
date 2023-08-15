import logging
import math

import numpy as np
import ast
import pandas as pd
import json

from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

logger = logging.getLogger()


def init_logger(log_file=None):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger

def load_data2labels_BIOES(args, vocab_size):
    labels_map_file, train_file, val_file, test_file = args.labels_map_file, args.train_file, args.val_file, args.test_file

    # if dataname == "german_doc":
    #     labels_map_file = "dataset/german_doc/label2idx_dict"
    #     train_file = "dataset/german_doc/train.txt"
    #     val_file = "dataset/german_doc/val.txt"
    #     test_file = "dataset/german_doc/test.txt"
    # else:
    #     print(dataname)
    #     raise  NotImplementedError()

    with open(labels_map_file, "r") as file:
        labels2idx = json.loads(file.read())

    all_text = []
    data_list = []
    path_list = [train_file, val_file, test_file]
    for path in path_list:
        data = pd.read_csv(path, index_col=0)
        try:
            X = data["token_seq"].map(ast.literal_eval).values.tolist()
            Y = data["label_seq"].map(ast.literal_eval).values.tolist()
        except:
            X = data["para_token_seq"].map(ast.literal_eval).values.tolist()
            Y = data["combined_token_label_seq"].map(ast.literal_eval).values.tolist()
        data_list.append(my_data2labels(X, Y, labels_map=None))

        # test text is treated as unseen part
        if path != test_file:
            all_text.extend([" ".join(x) + "\n" for x in X])

    vocab_processor = VocabularyProcessor(all_text, vocab_size)

    return data_list, labels2idx, vocab_processor

def my_data2labels(X, Y, labels_map=None):
    seq_set = []
    seq_set_label = []
    seq_set_len = []
    for row in Y:
        if labels_map is not None:
            seq_set_label.append([str(labels_map[label]) for label in row])
        else:
            seq_set_label.append([label for label in row])
        seq_set_len.append(len(row))

    for row in X:
        seq_set.append(" ".join(row))

    return [seq_set, seq_set_label, seq_set_len]


def load_data2labels(input_file, labels_map):
    seq_set = []
    seq = []
    seq_set_label = []
    seq_label = []
    seq_set_len = []
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                seq_set.append(" ".join(seq))
                seq_set_label.append(seq_label)
                seq_set_len.append(len(seq_label))
                seq = []
                seq_label = []
            else:
                tok, label = line.split()
                seq.append(tok)
                seq_label.append(labels_map[label])
    return [seq_set, seq_set_label, seq_set_len]


class VocabularyProcessor(object):
    def __init__(self, all_text, vocab_size):
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        # And finally, let's plug a decoder so we can recover from a tokenized input to the original one
        self.tokenizer.decoder = ByteLevelDecoder()
        self.trainer = BpeTrainer(vocab_size=vocab_size, show_progress=True, min_frequency=1,
                             special_tokens=["[UNK]", "[PAD]"])
        self.fit(all_text)

    def fit(self, text_list):
        tmp_name = "tmp_raw_txt4vocab"
        with open(tmp_name, "w+") as file:
            file.write("".join(text_list))
        self.tokenizer.train([tmp_name], self.trainer)

    def transform2idx(self, sent_list, is_pretokenized=False):
        results = self.tokenizer.encode_batch(sent_list, is_pretokenized=is_pretokenized)
        idx_tokens_list = [x.ids for x in results]
        return idx_tokens_list

def load_glove(file):
    """Loads GloVe vectors in numpy array.

    Args:
        file (str): a path to a glove file.

    Return:
        dict: a dict of numpy arrays.
    """
    model = {}
    with open(file, encoding="utf8", errors='ignore') as f:
        for line in f:
            line = line.split(' ')
            word = line[0]
            vector = np.array([float(val) for val in line[1:]])
            model[word] = vector
    return model

def filter_embedding(pre_w2v, vocab, max_vocab_size = 20000):
    n_dict = len(vocab)
    vocab_w2v = [None] * n_dict
    emb_size = len(list(pre_w2v.values())[0])
    # vocab_w2v[0]=np.random.uniform(-0.25,0.25,100)
    for w, i in vocab.items():
        if w in pre_w2v:
            vocab_w2v[i] = pre_w2v[w]
        elif w.lower() in pre_w2v:
            # 如果大些的没有embedding，则找小写的
            vocab_w2v[i] = pre_w2v[w.lower()]
        else:
            vocab_w2v[i] = list(np.random.uniform(-0.25, 0.25, emb_size))

    cur_i = len(vocab_w2v)
    if len(vocab_w2v) > max_vocab_size:
        logger.info("Vocabulary size is larger than {}".format(max_vocab_size))
        raise SystemExit
    while cur_i < max_vocab_size:
        cur_i += 1
        padding = [0] * emb_size
        vocab_w2v.append(padding)
    logger.info("Vocabulary {} Embedding size {}".format(n_dict, emb_size))
    return vocab_w2v

def data2sents(X, Y):
    data = []
    for i in range(len(Y)):
        sent = []
        text = X[i]
        items = text.split(" ")
        for j in range(len(Y[i])):
            sent.append((items[j], str(Y[i][j])))
        #elements in data is tuple (token,label)
        data.append(sent)
    return data

def sents2Xdata(sents):
    data=[]
    for item in sents:
        text=''
        for point in item:
            text=text+' '+point[0]
        data.append(text)
    return data

def randomKSamples(train_pool, train_pool_idx, num):
    #x_un=np.array(x_un)
    #y_un=np.array(y_un)
    random_pool=[]
    random_pool_idx=[]
    indices=np.arange(len(train_pool))
    np.random.shuffle(indices)
    queryindices=indices[0:num]
    for i in range(0,num):
        random_pool.append(train_pool[indices[i]])
        random_pool_idx.append(train_pool_idx[indices[i]])
    return random_pool, random_pool_idx, queryindices

def getEntropy(v):
    entropy=0.
    for element in v:
        p=float(element)
        if p > 0:
            entropy=entropy+ (-p)* np.log(p)
    return entropy

def compute_seq_entropy(tag_prob):
    seq_len, num_classes = tag_prob.shape
    ttk = 0.
    for i in range(seq_len):
        ent = 0.
        for y_i in tag_prob[i]:
            if y_i > 0:
                ent -= y_i * math.log(y_i, num_classes)
        ttk += ent
    return ttk

def compute_entropy(y, num_class):
    ent = 0.
    for y_i in y:
        if y_i > 0:
            ent -= y_i * math.log(y_i, num_class)
    return ent
