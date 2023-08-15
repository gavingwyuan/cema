import numpy as np
import utils.utils as utilities
from model.Doc2Piece import Doc2Piece
from model.CostEstimator import CostPerformaceEval
from sklearn.model_selection import KFold
import random

def split_data(all_x, all_y, all_lens, r):
    all_n = len(all_x)
    indices = np.random.RandomState(r).permutation(all_n)

    all_x = [all_x[idx] for idx in indices]
    all_y = [all_y[idx] for idx in indices]
    all_lens = [all_lens[idx] for idx in indices]

    training_size = int(all_n * 0.7)
    train_x, train_y, train_lens = all_x[:training_size], all_y[:training_size], all_lens[:training_size]
    test_x, test_y, test_lens = all_x[training_size:], all_y[training_size:], all_lens[training_size:]

    train_sents = utilities.data2sents(train_x, train_y)
    test_sents = utilities.data2sents(test_x, test_y)

    return train_x, train_y, train_lens, test_x, test_y, test_lens, train_sents, test_sents

def split_4test(train_x, train_y, train_lens, test_x, test_y, test_lens, init_size, r):
    init_x = train_x[:init_size]
    init_y = train_y[:init_size]
    init_sents = utilities.data2sents(init_x, init_y)
    init_lens = train_lens[:init_size]
    tmp_x = train_x[init_size:] + test_x
    tmp_y = train_y[init_size:] + test_y
    tmp_lens = train_lens[init_size:] + test_lens
    train_x, train_y, train_lens, test_x, test_y, test_lens, train_sents, test_sents = split_data(tmp_x, tmp_y, tmp_lens, r)
    train_x = init_x + train_x
    train_y = init_y + train_y
    train_sents = init_sents + train_sents
    train_lens = init_lens + train_lens
    return train_x, train_y, train_lens, test_x, test_y, test_lens, train_sents, test_sents

def split_4para(train_x, train_y, train_lens, dev_x, dev_y, dev_lens, init_size, r):
    init_x = train_x[:init_size]
    init_y = train_y[:init_size]
    init_sents = utilities.data2sents(init_x, init_y)
    init_lens = train_lens[:init_size]
    tmp_x = dev_x
    tmp_y = dev_y
    tmp_lens = dev_lens
    train_x, train_y, train_lens, test_x, test_y, test_lens, train_sents, test_sents = split_data(tmp_x, tmp_y, tmp_lens, r)
    train_x = init_x + train_x
    train_y = init_y + train_y
    train_sents = init_sents + train_sents
    train_lens = init_lens + train_lens
    return train_x, train_y, train_lens, test_x, test_y, test_lens, train_sents, test_sents

def to_segmented(sents, end_symbol = ["."], initial_num = 0, random = True, required_spans = False, is_merge=True, required_bioes=True):
    initial_sents = sents[:initial_num]
    sents = sents[initial_num:]

    # train_sents = utilities.data2sents(train_x, train_y)
    doc2piece = Doc2Piece(sents, tokenizer = None, end_symbol = end_symbol, max_length = 510, split_way = "symbol")
    segmented_sents = doc2piece.segmented_sents

    if random:
        length = len(segmented_sents)
        indices = np.random.RandomState(0).permutation(length)
        segmented_sents = [segmented_sents[idx] for idx in indices] # [[x, y, label mask], ...],...]

    data = Doc2Piece.unpackage(segmented_sents)
    X = data[0]
    Y = data[1]
    Y = [CostPerformaceEval.formatBIOES(seq, is_merge=is_merge, required_bioes=required_bioes) for seq in Y]
    assert list(map(len, X)) == list(map(len, Y))
    k = 2
    if len(sents) > 0 and len(sents[0])>0:
        k = len(sents[0][0])
    segmented_sents = Doc2Piece.package([X, Y] + data[2:k])
    segmented_sents = initial_sents + segmented_sents # [[x, y], ...],...]

    lens = [len(x) for x in X]

    if required_spans:
        return segmented_sents, X, Y, lens, doc2piece.piece_span
    else:
        return segmented_sents, X, Y, lens

def k_folds_split4test(X, Y, all_lens, init_size, n_splits=5):
    # permutation
    all_n = len(X)
    indices = np.random.RandomState(0).permutation(all_n)
    X = [X[idx] for idx in indices]
    Y = [Y[idx] for idx in indices]
    all_lens = [all_lens[idx] for idx in indices]

    init_x = X[:init_size]
    init_y = Y[:init_size]
    init_lens = all_lens[:init_size]

    tmp_x = X[init_size:]
    tmp_y = Y[init_size:]
    tmp_lens = all_lens[init_size:]

    kfolds_data = []

    kf = KFold(n_splits=n_splits)
    for train_index, test_index in kf.split(tmp_x):
        train_x = [tmp_x[idx] for idx in train_index]
        test_x = [tmp_x[idx] for idx in test_index]

        train_y = [tmp_y[idx] for idx in train_index]
        test_y = [tmp_y[idx] for idx in test_index]

        train_lens = [tmp_lens[idx] for idx in train_index]
        test_lens = [tmp_lens[idx] for idx in test_index]

        train_x = init_x + train_x
        train_y = init_y + train_y
        train_lens = init_lens + train_lens

        train_sents = utilities.data2sents(train_x, train_y)
        test_sents = utilities.data2sents(test_x, test_y)

        kfolds_data.append([train_x, train_y, train_lens, test_x, test_y, test_lens, train_sents, test_sents])
    return kfolds_data

def merge_sent2doc_controlling_revelant_sentence(sents, num_docs, percent=0.02, print_info=True):
    irrelevant_sent_list = []
    relevant_sent_list = []
    nsent_with_fea = 0
    nsent_without_fea = 0
    has_fea_list = []
    for i in range(len(sents)):
        flag = False
        for token, label in sents[i]:
            if label != "O":
                flag = True
                break

        if flag:
            nsent_with_fea += 1
            has_fea_list.append(True)
            relevant_sent_list.append(sents[i])
        else:
            nsent_without_fea += 1
            has_fea_list.append(False)
            irrelevant_sent_list.append(sents[i])

    if print_info:
        print("ratio of sentence containing features: {}/{}={}".format(nsent_with_fea, len(sents),
                                                                       nsent_with_fea / len(sents)))

    doc_start_idx = []
    relevant_sent_each_doc = [int((nsent_with_fea + num_docs - 1) / num_docs) for x in range(num_docs)]

    irrelevant_sent_each_doc = [int(x / (percent) - x) for x in relevant_sent_each_doc]

    if print_info:
        print("In new german_coarse_doc, {} sentences has features".format(sum(relevant_sent_each_doc)))
        print("In new german_coarse_doc, each documents have {} sentences containing features".format(
            sum(relevant_sent_each_doc) / num_docs))
        print("In new german_coarse_doc, each documents have {} sentences without features".format(
            sum(irrelevant_sent_each_doc) / num_docs))
        print("In new german_coarse_doc, ratio of sentence containing features: {}/{}={}".format(sum(relevant_sent_each_doc),
                                                                                       sum(relevant_sent_each_doc) + sum(
                                                                                           irrelevant_sent_each_doc),
                                                                                       sum(relevant_sent_each_doc) / (
                                                                                                   sum(relevant_sent_each_doc) + sum(
                                                                                               irrelevant_sent_each_doc))))

    start_idx_list = [0]
    i_doc = 0
    cur_n_sent_with_fea = 0
    has_fea_list = has_fea_list
    for i in range(len(has_fea_list)):

        has_fea = has_fea_list[i]
        if has_fea:
            cur_n_sent_with_fea += 1

        if i_doc < len(relevant_sent_each_doc) and cur_n_sent_with_fea > relevant_sent_each_doc[i_doc]:
            start_idx_list.append(i)
            cur_n_sent_with_fea = 1
            i_doc += 1

        if len(start_idx_list) >= len(relevant_sent_each_doc):
            break

    doc_list = []
    for i, idx in enumerate(start_idx_list):

        a_doc = []
        n_relevant = relevant_sent_each_doc[i]
        n_irrelevant = irrelevant_sent_each_doc[i]

        cur_n_relevant = 0
        cur_n_irrelevant = 0
        for ith in range(idx, len(sents)):
            has_fea = has_fea_list[ith]

            if has_fea:
                if cur_n_relevant < n_relevant:
                    a_doc.extend(sents[ith])
                    cur_n_relevant += 1
            else:
                if cur_n_irrelevant < n_irrelevant:
                    a_doc.extend(sents[ith])
                    cur_n_irrelevant += 1

            if cur_n_relevant >= n_relevant and cur_n_irrelevant >= n_irrelevant:
                break

        random.shuffle(relevant_sent_list)
        random.shuffle(irrelevant_sent_list)

        for i in range(n_relevant - cur_n_relevant):
            cur_n_relevant += 1
            asent = relevant_sent_list[i % nsent_with_fea]
            a_doc.extend(asent)

        for i in range(n_irrelevant - cur_n_irrelevant):
            cur_n_irrelevant += 1
            asent = irrelevant_sent_list[i % nsent_without_fea]
            a_doc.extend(asent)

        doc_list.append(a_doc)

    return doc_list

def get_doc_with_target_density(docs, density, end_symbol = ["\n"]):
    new_docs = []
    for i in range(len(docs)):
        doc2piece = Doc2Piece(docs[i:i+1], tokenizer = None, end_symbol = end_symbol, max_length = 10000, split_way = "symbol")
        segmented_sents = Doc2Piece.package(Doc2Piece.unpackage(doc2piece.segmented_sents)[:2])
        print_info = False
        if i == 0:
            print_info = True
        a_new_doc = merge_sent2doc_controlling_revelant_sentence(segmented_sents, 1, percent=density, print_info = print_info)
        new_docs.extend(a_new_doc)
    return new_docs
