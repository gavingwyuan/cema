import numpy as np
import pycrfsuite
import os
import random
import math

from model.Doc2Piece import Doc2Piece

class CRFTagger(object):

    def __init__(self, modelfile, unique_label_list, max_length = 0, end_symbol = None):
        # print("CRF Tagger")
        self.modelfile = modelfile
        self.name = "CRF"
        self.num_classes = len(unique_label_list)
        self.unique_label_list = unique_label_list
        self.max_length = max_length
        self.end_symbol = end_symbol

    def word2features(self, sent, i):
        word = sent[i][0]
        # postag = sent[i][1]
        features = [
            'bias',
            'word.lower=' + word.lower(),
            'word[-3:]=' + word[-3:],
            'word[-2:]=' + word[-2:],
            'word.isupper=%s' % word.isupper(),
            'word.istitle=%s' % word.istitle(),
            'word.isdigit=%s' % word.isdigit(),
            # 'postag=' + postag,
            # 'postag[:2]=' + postag[:2],
        ]
        if i > 0:
            word1 = sent[i - 1][0]
            # postag1 = sent[i - 1][1]
            features.extend([
                '-1:word.lower=' + word1.lower(),
                '-1:word.istitle=%s' % word1.istitle(),
                '-1:word.isupper=%s' % word1.isupper(),
                # '-1:postag=' + postag1,
                # '-1:postag[:2]=' + postag1[:2],
            ])
        else:
            features.append('BOS')

        if i < len(sent) - 1:
            word1 = sent[i + 1][0]
            # postag1 = sent[i + 1][1]
            features.extend([
                '+1:word.lower=' + word1.lower(),
                '+1:word.istitle=%s' % word1.istitle(),
                '+1:word.isupper=%s' % word1.isupper(),
                # '+1:postag=' + postag1,
                # '+1:postag[:2]=' + postag1[:2],
            ])
        else:
            features.append('EOS')

        return features

    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    def sent2labels(self, sent):
        # print sent
        # return [label for token, label in sent]
        return [item[1] for item in sent]

    def sent2tokens(self, sent):
        # return [token for token, label in sent]
        return [item[0] for item in sent]

    def train(self, train_sents):
        ## split into piece
        dataset = Doc2Piece(train_sents, end_symbol=self.end_symbol, max_length=self.max_length)
        train_sents = dataset.segmented_sents

        X_train = [self.sent2features(s) for s in train_sents]
        Y_train = [self.sent2labels(s) for s in train_sents]
        trainer = pycrfsuite.Trainer(verbose=False)
        for xseq, yseq in zip(X_train, Y_train):
            trainer.append(xseq, yseq)
        trainer.set_params({
            'c1': 1.0,   # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            'max_iterations': 50,  # stop earlier

            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })
        trainer.train(self.modelfile)


        '''
        if len(trainer.logparser.iterations) != 0:
            print(len(trainer.logparser.iterations), trainer.logparser.iterations[-1])
        else:
            # todo
            print(len(trainer.logparser.iterations))
            print("There is no loss to present")
        '''

    # different lens
    def _get_predictions(self, sent, tagger=None):
        if tagger is None:
            tagger = pycrfsuite.Tagger()
            if not os.path.isfile(self.modelfile):
                unc = random.random()
                return unc
            tagger.open(self.modelfile)

        x = self.sent2features(sent)
        tagger = pycrfsuite.Tagger()
        if not os.path.isfile(self.modelfile):
            y_marginals = []
            for i in range(len(x)):
                y_marginals.append([0.2] * self.num_classes)
            return y_marginals

        tagger.open(self.modelfile)
        tagger.set(x)
        y_marginals = []
        #print(tagger.labels())
        # if len(tagger.labels) < 5
        for i in range(len(x)):
            y_i = []
            for y in self.unique_label_list:
                if y in tagger.labels():
                    y_i.append(tagger.marginal(y, i))
                else:
                    y_i.append(0.)
            y_marginals.append(y_i)
        return y_marginals

    def get_predictions(self, sent):
        tagger = pycrfsuite.Tagger()
        if not os.path.isfile(self.modelfile):
            unc = random.random()
            return unc
        tagger.open(self.modelfile)

        dataset = Doc2Piece([sent], end_symbol=self.end_symbol, max_length=self.max_length)
        y_marginals = []
        segmented_sents = dataset.segmented_sents
        for new_sent in segmented_sents:
            y_marginals += self._get_predictions(new_sent, tagger=tagger)
        return y_marginals

    # use P(yseq|xseq)
    def _get_confidence(self, sent, normalized = True, use_log = False, tagger=None):
        if tagger is None:
            tagger = pycrfsuite.Tagger()
            if not os.path.isfile(self.modelfile):
                unc = random.random()
                return unc
            tagger.open(self.modelfile)

        x = self.sent2features(sent)
        tagger = pycrfsuite.Tagger()
        if not os.path.isfile(self.modelfile):
            confidence = 0.2
            return [confidence]

        tagger.open(self.modelfile)
        tagger.set(x)
        y_pred = tagger.tag()
        #avoid division by zero
        if len(y_pred) == 0:
            confidence = 0.2
            return [confidence]
        p_y_pred = tagger.probability(y_pred)
        if normalized:
            confidence = pow(p_y_pred, 1. / len(y_pred))
        else:
            confidence = p_y_pred

        if use_log:
            # +np.log(np.finfo(float).eps
            confidence = np.log(confidence)
        return confidence

    def get_confidence(self, sent, normalized = True, use_log = False):
        tagger = pycrfsuite.Tagger()
        if not os.path.isfile(self.modelfile):
            unc = random.random()
            return unc
        tagger.open(self.modelfile)

        dataset = Doc2Piece([sent], end_symbol=self.end_symbol, max_length=self.max_length)
        segmented_sents = dataset.segmented_sents
        p_y_pred = 0
        for new_sent in segmented_sents:
            p_y_pred += self._get_confidence(new_sent, normalized = False, use_log = True, tagger=tagger)

        p_y_pred = np.e ** p_y_pred
        if normalized:
            # confidence = pow(p_y_pred, 1. / len(sent))
            log_confidence = p_y_pred / len(sent)
            confidence = np.e ** log_confidence
        else:
            p_y_pred = np.e ** p_y_pred
            confidence = p_y_pred

        if use_log:
            # +np.log(np.finfo(float).eps
            confidence = np.log(confidence)

        return [confidence]

    def _get_uncertainty(self, sent, tagger = None):
        if tagger is None:
            tagger = pycrfsuite.Tagger()
            if not os.path.isfile(self.modelfile):
                unc = random.random()
                return unc
            tagger.open(self.modelfile)

        x = self.sent2features(sent)
        tagger.set(x)
        ttk = 0.
        for i in range(len(x)):
            y_probs = []
            for y in self.unique_label_list:
                if y in tagger.labels():
                    y_probs.append(tagger.marginal(y, i))
                else:
                    y_probs.append(0.)
            ent = 0.
            for y_i in y_probs:
                if y_i > 0:
                    ent -= y_i * math.log(y_i, self.num_classes)
            ttk += ent
        return ttk

    def get_uncertainty(self, sent):
        tagger = pycrfsuite.Tagger()
        if not os.path.isfile(self.modelfile):
            unc = random.random()
            return unc
        tagger.open(self.modelfile)

        dataset = Doc2Piece([sent], end_symbol=self.end_symbol, max_length=self.max_length)
        segmented_sents = dataset.segmented_sents
        ttk = 0
        for new_sent in segmented_sents:
            ttk += self._get_uncertainty(new_sent, tagger=tagger)
        return ttk

    def test(self, test_sents):

        dataset = Doc2Piece(test_sents, end_symbol=self.end_symbol, max_length=self.max_length)
        segmented_sents = dataset.segmented_sents
        y_pred = self.predict(segmented_sents)
        ## recover from merging piece
        y_pred = dataset.recover_from_spans(y_pred, dataset.piece_span)

        Y_true = [self.sent2labels(s) for s in test_sents]
        # y_pred = self.predict(test_sents)

        # token level f1 score
        # X_test = [self.sent2features(s) for s in test_sents]
        # Y_true = [self.sent2labels(s) for s in test_sents]
        # tagger = pycrfsuite.Tagger()
        # tagger.open(self.modelfile)
        # y_pred = [tagger.tag(xseq) for xseq in X_test]
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

    def predict(self, sents):
        ## split into piece
        dataset = Doc2Piece(sents, end_symbol=self.end_symbol, max_length=self.max_length)
        segmented_sents = dataset.segmented_sents

        X_features = [self.sent2features(s) for s in segmented_sents]
        tagger = pycrfsuite.Tagger()
        tagger.open(self.modelfile)
        y_pred = [tagger.tag(xseq) for xseq in X_features]

        ## recover from merging piece
        y_pred = dataset.recover_from_spans(y_pred, dataset.piece_span)

        return y_pred



