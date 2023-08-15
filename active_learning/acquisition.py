import numpy as np
from copy import deepcopy
import random

from active_learning.cosine_simlarity import CosSimWrapper
from model.CostEstimator import CostEvalFromAction
from utils.utils import init_logger
from seqeval.metrics.sequence_labeling import get_entities

from model.Doc2Piece import Doc2Piece

SEED = 0

def get_rnd_topk(sent_list, k):
    # k instances are selected randomly
    test_indices = np.random.RandomState(SEED).permutation(len(sent_list))
    top_k_indices = set(test_indices[:k])
    return list(top_k_indices)

def get_min_len_topk(sent_list, k):
    length = [len(sent) for sent in sent_list]
    indices = np.argsort(length)
    top_k_indices = indices[:k]
    return list(top_k_indices)

def get_mnlp_topk(norm_log_prob, sent_list, k):
    indices = np.argsort(norm_log_prob)
    top_k_indices = indices[:k]
    return list(top_k_indices)

def get_lc_topk(log_prob, k):
    indices = np.argsort(log_prob)
    top_k_indices = indices[:k]
    return list(top_k_indices)

def get_least_entropy_topk(entropy, k):
    indices = np.argsort(entropy)
    top_k_indices = indices[:k]
    return list(top_k_indices)

def get_least_topk(metric_values, k):
    indices = np.argsort(metric_values)
    top_k_indices = indices[:k]
    return list(top_k_indices)

class CEMA_Doc_Selctor(object):
    def __init__(self, train_index, action_labels, time_para, entity_ebd_dict, entity_sim_threshold = 0.8,
                 influence_weight = 0.5, batch_size = 512, log_preprocessed = False):
        self.set_train_idx(train_index)
        self.set_timepara(time_para)
        self.cefa = CostEvalFromAction(action_labels)
        self.total_instance_num = len(action_labels)
        self.entity_ebd_dict = entity_ebd_dict
        self.entity_sim_threshold = entity_sim_threshold
        self.influence_weight = influence_weight
        self.batch_size = batch_size # used in CosSimWrapper
        self.log_preprocessed = log_preprocessed

    def get_sim_entity_list(self, total_instance_num, entity_ebd_dict, sim_threshold, batch_size = 512):
        # instance_num is the number of instance
        # entity_ebd_dict may be empty if there is not any entity

        entity_loc_list = [key for key in entity_ebd_dict]
        entity_ebd_list = [list(entity_ebd_dict[key]) for key in entity_ebd_dict]
        if len(entity_ebd_dict) > 0:
            entity_sim_list = CosSimWrapper().cos_sim_list(entity_ebd_list, entity_ebd_list, sim_threshold, batch_size=batch_size)
        else:
            entity_sim_list = []
        # instance_num = max([key[0]+1 for key in entity_ebd_dict]) #key = (i_instance, start, end) + 1, i_instance start from 0

        instance_sim_entity = [set() for i in range(total_instance_num)]
        for s_eidx, elist in enumerate(entity_sim_list):
            ridx = entity_loc_list[s_eidx][0] # instance idx
            for jth, t_eidx in enumerate(elist):
                loc = entity_loc_list[t_eidx]
                instance_sim_entity[ridx].add(loc)
        del entity_sim_list
        return instance_sim_entity


    def update_instance_sim_entity(self, instance_sim_entity, idx_set):
        for ith in idx_set:
            key_list = list(instance_sim_entity[ith])
            for loc1 in key_list:
                for jth in range(len(instance_sim_entity)):
                    if loc1 in instance_sim_entity[jth]:
                        instance_sim_entity[jth].remove(loc1)

        for ith in range(len(instance_sim_entity)):
            key_list = list(instance_sim_entity[ith])
            for loc2 in key_list:
                if loc2[0] in idx_set:
                    instance_sim_entity[ith].remove(loc2)

        return instance_sim_entity


    def cema_sampling(self, num_instances, stop_at_zero=True, logger = "", influence_can_zero = False):
        if logger == "":
            logger = init_logger()

        total_instance_num = self.total_instance_num
        instance_sim_entity = self.get_sim_entity_list(total_instance_num, self.entity_ebd_dict, self.entity_sim_threshold, self.batch_size)

        logger.info('start of getting get_instance_correction_cost')
        annocation_cost = self.cefa.get_instance_correction_cost(time_para=self.time_para)
        logger.info('end of getting get_instance_correction_cost')
        instancce_cost = [instance["total_cost"] for instance in annocation_cost]

        new_add_idx = set()
        total_n_ety = None
        logger.info('start of getting update_instance_sim_entity')
        instance_sim_entity = self.update_instance_sim_entity(instance_sim_entity, [])
        logger.info('end of getting update_instance_sim_entity')
        if instance_sim_entity is not None:
            total_n_ety = len({xx for x in instance_sim_entity for xx in x})

        while len(new_add_idx) < num_instances:
            logger.info('start of getting update_instance_sim_entity')
            instance_sim_entity = self.update_instance_sim_entity(instance_sim_entity, self.train_index)
            logger.info('end of getting update_instance_sim_entity')
            influences = [len(x) for x in instance_sim_entity]

            cur_index = self.get_next_samples(self.train_index, instancce_cost, influences, influence_can_zero=influence_can_zero)

            n_ety_not_in_train = len({xx for x in instance_sim_entity for xx in x})

            if cur_index is None:
                break
            else:
                max_influence = max(influences)
                min_influence = min(influences)
                logger.info('min influence: {}, max influence: {}'.format(min_influence, max_influence))
                logger.info('instance {} has influence gain {}'.format(cur_index, influences[cur_index]))

            new_add_idx.add(cur_index)
            self.train_index.add(cur_index)

            logger.info("n_ety_not_in_train/total_n_ety:{}/{}".format(n_ety_not_in_train, total_n_ety))

            if n_ety_not_in_train == 0 and stop_at_zero:
                break

        return list(new_add_idx)


    def get_next_samples(self, cur_train_idx, costs, influences, influence_can_zero = False):
        norm_cost, _ = self.normalize(costs, train_idx=cur_train_idx, inf_sign = 1)
        norm_influences, _ = self.normalize(influences, train_idx=cur_train_idx, inf_sign = -1)

        w = self.influence_weight
        driven_info = []
        for i_norm_cost, i_norm_incluence in zip(norm_cost, norm_influences):
            score = (1-w)*i_norm_cost-w*i_norm_incluence
            driven_info.append(score)

        if influence_can_zero == False:
            if self.influence_weight > 0:
                # give higher priority for those instances has influences.
                # if all of those instances has not influence, one instance will be pick randomly
                max_v = max(driven_info)
                for ith in range(len(influences)):
                    if influences[ith] == 0:
                        driven_info[ith] = max_v + 1 + random.random()

        indicate_info = np.array(driven_info)

        # argmin (1-w)*i_norm_cost-w*i_norm_incluence
        indices = np.argsort(indicate_info)

        new_indices = None
        for i in range(len(indices)):
            if (indices[i] not in cur_train_idx):
                new_indices = indices[i]
                break
        return new_indices


    def normalize(self, alist, train_idx = None, inf_sign = 1):
        if self.log_preprocessed:
            alist = [np.log(x+1) for x in alist]

        # instance in train_idx will be set as inf
        if train_idx == None: train_idx = []
        new_list = [x for ith,x in enumerate(alist) if ith not in train_idx]

        max_v = max(new_list)
        min_v = min(new_list)
        flag = 1
        if max_v == 0:
            flag = 0

        if max_v == min_v:
            return [0 if ith not in train_idx else inf_sign*float("inf") for ith, _ in enumerate(alist)], flag

        intervals = max_v - min_v
        alist = [(v - min_v) / intervals if ith not in train_idx else inf_sign*float("inf") for ith,v in enumerate(alist)]
        return alist, flag


    def set_timepara(self, time_para):
        self.time_para = time_para


    def set_train_idx(self, train_idx):
        self.train_index = set(deepcopy(train_idx))

    @staticmethod
    def get_entities(pred_labels, actions = None, length = 5):
        label_entities = [get_entities(seq) for seq in pred_labels]
        actions_entities_set = [set(get_entities(seq)) for seq in actions]

        if actions is not None:
            for i in range(len(actions)):
                add_set = {entity for entity in actions_entities_set[i] if "add" in entity[0].lower()}
                new_add_set = {(entity[0], entity[1], min([entity[2]+length-1, len(actions[i])-1])) for entity in add_set}
                label_entities[i].extend(list(new_add_set))

        return label_entities

class CEMAPartialStrategy(object):

    @staticmethod
    def partial_with_ety(all_sents, pred_labels, action_labels, end_symbol,
                         refer_pred_labels = None, refer_action_labels = None, pred_action_labels = None):

        if refer_pred_labels is None:
            refer_pred_labels = pred_labels
        else:
            assert [len(x) for x in pred_labels] == [len(x) for x in refer_pred_labels]

        if action_labels is None:
            action_labels = [["O" for x in row] for row in pred_labels]

        if refer_action_labels is None:
            refer_action_labels = action_labels
        else:
            assert [len(x) for x in action_labels] == [len(x) for x in refer_action_labels]

        if pred_action_labels is not None:
            assert [len(x) for x in action_labels] == [len(x) for x in pred_action_labels]

        def has_ety(span, label_list, action_label_list):
            s, t = span
            for label in set(label_list[s:t+1]):
                if label != "O" and label != "o":
                    return True

            for label in set(action_label_list[s:t+1]):
                if "add" in label or "ADD" in label:
                    return True

            return False

        doc2piece = Doc2Piece(all_sents, tokenizer=None, end_symbol=end_symbol, max_length=510, split_way="symbol")
        piece_span = doc2piece.piece_span

        n_doc = len(all_sents)
        new_all_sents = [[] for i in range(n_doc)]
        new_pred_labels = [[] for i in range(n_doc)]
        new_action_labels = [[] for i in range(n_doc)]
        new_pred_action_labels = [[] for i in range(n_doc)]
        new_refer_action_labels = [[] for i in range(n_doc)]
        for ith, spans in enumerate(piece_span):
            for span in spans:
                s, t = span
                if has_ety(span, refer_pred_labels[ith], refer_action_labels[ith]):
                    new_all_sents[ith].extend(deepcopy(all_sents[ith][s:t + 1]))
                    new_pred_labels[ith].extend(deepcopy(pred_labels[ith][s:t + 1]))
                    new_action_labels[ith].extend(deepcopy(action_labels[ith][s:t + 1]))
                    if pred_action_labels is not None:
                        new_pred_action_labels[ith].extend(deepcopy(pred_action_labels[ith][s:t + 1]))
                    new_refer_action_labels[ith].extend(deepcopy(refer_action_labels[ith][s:t + 1]))

        all_sents_no_empty = []
        pred_labels_no_empty = []
        action_labels_no_empty = []
        pred_action_labels_no_empty = []
        refer_action_labels_no_empty = []
        idx_mapping = {}

        ith = 0
        for i in range(n_doc):
            if len(new_pred_labels[i]) > 0:
                idx_mapping[ith] = i
                all_sents_no_empty.append(new_all_sents[i])
                pred_labels_no_empty.append(new_pred_labels[i])
                action_labels_no_empty.append(new_action_labels[i])
                if pred_action_labels is not None:
                    pred_action_labels_no_empty.append(new_pred_action_labels[i])
                refer_action_labels_no_empty.append(new_refer_action_labels[i])
                ith+=1

        if pred_action_labels is None:
            return idx_mapping, all_sents_no_empty, pred_labels_no_empty, action_labels_no_empty, refer_action_labels_no_empty
        else:
            return idx_mapping, all_sents_no_empty, pred_labels_no_empty, action_labels_no_empty, refer_action_labels_no_empty, pred_action_labels_no_empty
