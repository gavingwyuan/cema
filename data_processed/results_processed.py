import pickle as pkl
import os
import sys
from copy import deepcopy
sys.path.append("../")

from model.CostEstimator import CostPerformaceEval, CostEvalFromAction
from model.TransformerNER import TFNERDataset
from seqeval.metrics.sequence_labeling import get_entities
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import numpy as np

def format_labels(data, need_unpackage = False, is_merge=True):
    if need_unpackage:
        data = TFNERDataset.unpackage(data)[1]
    data = [CostPerformaceEval.formatBIOES(list(seq), is_merge=is_merge) for seq in data]
    return data

def get_cost(results, time_para, sim_threshold):
    cost_dict = {"fma": [], "maa": []}
    action_cnt = {'confirm_cnt': [], 'revise_cnt': [], 'delete_cnt': [], 'add_cnt': [], 'read_cnt': []}
    for instance in results:
        instance_action_cnt = {'confirm_cnt': 0, 'revise_cnt': 0, 'delete_cnt': 0, 'add_cnt': 0, 'read_cnt': 0}
        # cal traing cost, train accuracy
        if len(instance["selected_sents"]) > 0:
            # get cost
            selected_true_labels = format_labels(instance["selected_sents"], need_unpackage=True, is_merge=True)
            selected_pred_labels = format_labels(instance["selected_pred_labels"], is_merge=True)
            cpe = CostPerformaceEval(selected_true_labels, selected_pred_labels, sim_threshold=sim_threshold)
            fma_cost = cpe.get_cost(time_para=time_para, way='FMA', need_verifier=True)
            maa_cost = cpe.get_cost(time_para=time_para, way='MAA', need_verifier=False)
            cost_dict["fma"].append(fma_cost["total_cost"])
            cost_dict["maa"].append(maa_cost["total_cost"])
            for aa in cpe.annotaion_actions:
                for key in aa.action_cnt:
                    instance_action_cnt[key] += aa.action_cnt[key]
            for key in instance_action_cnt:
                action_cnt[key].append(instance_action_cnt[key])
        else:
            for key in cost_dict:
                cost_dict[key].append(None)
            for key in action_cnt:
                action_cnt[key].append(None)
    return cost_dict, action_cnt

def get_pred_action_count(results):
    action_cnt = {'confirm_cnt': [], 'revise_cnt': [], 'delete_cnt': [], 'add_cnt': [], 'read_cnt': []}
    for instance in results:
        instance_action_cnt = {'confirm_cnt': 0, 'revise_cnt': 0, 'delete_cnt': 0, 'add_cnt': 0, 'read_cnt': 0}
        # cal traing cost, train accuracy
        if len(instance["selected_preds_actions"]) > 0:
            # get cost
            selected_preds_actions = format_labels(instance["selected_preds_actions"], is_merge=False)
            cefa = CostEvalFromAction(selected_preds_actions)
            for aa in cefa.action_cnt_list:
                #             print(aa.action_cnt)
                for key in aa.action_cnt:
                    instance_action_cnt[key] += aa.action_cnt[key]
            for key in instance_action_cnt:
                action_cnt[key].append(instance_action_cnt[key])
        else:
            for key in action_cnt:
                action_cnt[key].append(None)
    return action_cnt

def get_all_results_pred_action_count(paths):
    experiment_results = {}
    for ith, path in enumerate(paths):
        with open(path, "rb") as file:
            results = pkl.load(file)

        pred_action_cnt = get_pred_action_count(results)

        experiment_results[ith] = {"pred_action_cnt": pred_action_cnt,
                                   }
        return experiment_results


def get_fine_grained_acc_dict(results, sim_threshold = 5):

    def _add_fine_grained_accuracy(accuracy_list, accuracy_dict=None):
        if accuracy_dict is None:
            accuracy_dict = {"confirm_recall":[], "confirm_precision":[], "revise_recall":[], 
                                      "revise_precision":[], "delete_precision":[], "add_recall":[], 
                                      "report":[]}
        confirm_recall, confirm_precision, revise_recall, revise_precision, delete_precision, add_recall, report = accuracy_list
        accuracy_dict["confirm_recall"].append(confirm_recall)
        accuracy_dict["confirm_precision"].append(confirm_precision)
        accuracy_dict["revise_recall"].append(revise_recall)
        accuracy_dict["revise_precision"].append(revise_precision)
        accuracy_dict["delete_precision"].append(delete_precision)
        accuracy_dict["add_recall"].append(add_recall)
        accuracy_dict["report"].append(report)

        return accuracy_dict
    
    training_accuracy_dict = None
    testing_accuracy_acc_dict = None

    test_sents = results[0]['test_sents']
    test_true_labels = format_labels(test_sents, need_unpackage=True)
    for instance in results:
        # cal traing cost, train accuracy
        if len(instance["selected_sents"]) > 0:
            # get cost
            selected_true_labels = format_labels(instance["selected_sents"], need_unpackage=True, is_merge=True)
            selected_pred_labels = format_labels(instance["selected_pred_labels"], is_merge=True)
            cpe = CostPerformaceEval(selected_true_labels, selected_pred_labels, sim_threshold = sim_threshold)
            accuracy_list = cpe.get_annotation_report()
            training_accuracy_dict = _add_fine_grained_accuracy(accuracy_list, accuracy_dict=training_accuracy_dict)
        else:
            accuracy_list = [None] * len(training_accuracy_dict)
            training_accuracy_dict = _add_fine_grained_accuracy(accuracy_list, accuracy_dict=training_accuracy_dict)

        # cal test accuracy
        if type(instance["test_sent_pred_labels"]) is list and len(instance["test_sent_pred_labels"]) > 0:
            test_pred_labels = format_labels(instance["test_sent_pred_labels"], is_merge=True)
            cpe = CostPerformaceEval(test_true_labels, test_pred_labels, sim_threshold = 5)
            accuracy_list = cpe.get_annotation_report()
            testing_accuracy_acc_dict = _add_fine_grained_accuracy(accuracy_list, accuracy_dict=testing_accuracy_acc_dict)
        else:
            accuracy_list = [None] * len(testing_accuracy_acc_dict)
            testing_accuracy_acc_dict = _add_fine_grained_accuracy(accuracy_list, accuracy_dict=testing_accuracy_acc_dict)   
            
    return training_accuracy_dict, testing_accuracy_acc_dict



def get_f1_accuracy_dict(results, sim_threshold):

    def _add_f1_accuracy(accuracy_list, accuracy_dict=None):
        if accuracy_dict is None:
            accuracy_dict = {"micro_recall":[], "micro_precision":[], "micro_f1":[], "report":[]}
        micro_recall, micro_precision, micro_f1, report = accuracy_list
        accuracy_dict["micro_recall"].append(micro_recall)
        accuracy_dict["micro_precision"].append(micro_precision)
        accuracy_dict["micro_f1"].append(micro_f1)
        accuracy_dict["report"].append(report)

        return accuracy_dict
    
    training_accuracy_dict = None
    testing_accuracy_acc_dict = None

    test_sents = results[0]['test_sents']
    test_true_labels = format_labels(test_sents, need_unpackage=True)
    for instance in results:
        # cal traing cost, train accuracy
        if len(instance["selected_sents"]) > 0:
            # get cost
            selected_true_labels = format_labels(instance["selected_sents"], need_unpackage=True, is_merge=True)
            selected_pred_labels = format_labels(instance["selected_pred_labels"], is_merge=True)
            cpe = CostPerformaceEval(selected_true_labels, selected_pred_labels, sim_threshold = sim_threshold)
            accuracy_list = cpe.get_f1_report()
            training_accuracy_dict = _add_f1_accuracy(accuracy_list, accuracy_dict=training_accuracy_dict)
        else:
            accuracy_list = [None] * len(training_accuracy_dict)
            training_accuracy_dict = _add_f1_accuracy(accuracy_list, accuracy_dict=training_accuracy_dict)

        # cal test accuracy
        if type(instance["test_sent_pred_labels"]) is list and len(instance["test_sent_pred_labels"]) > 0:
            test_pred_labels = format_labels(instance["test_sent_pred_labels"], is_merge=True)
            cpe = CostPerformaceEval(test_true_labels, test_pred_labels, sim_threshold = sim_threshold)
            accuracy_list = cpe.get_f1_report()
            testing_accuracy_acc_dict = _add_f1_accuracy(accuracy_list, accuracy_dict=testing_accuracy_acc_dict)
        else:
            accuracy_list = [None] * len(testing_accuracy_acc_dict)
            testing_accuracy_acc_dict = _add_f1_accuracy(accuracy_list, accuracy_dict=testing_accuracy_acc_dict)   
            
    return training_accuracy_dict, testing_accuracy_acc_dict


def get_filtered_f1_cost(cost_dict, accuracy_acc_dict, cost_key="maa", acc_key="micro_f1", need_accumulated_cost=True):
    maa_cost = deepcopy(cost_dict[cost_key])
    micro_f1 = accuracy_acc_dict[acc_key]

    if need_accumulated_cost:
        for i in range(len(maa_cost)):
            if i < len(maa_cost) - 1:
                if maa_cost[i+1] is None:
                    print(
                        "maa_cost in get_filtered_f1_cost contain None. One of reason is that there is not enough samples for selection")
                    break
                maa_cost[i + 1] = maa_cost[i + 1] + maa_cost[i]

    results = {"step": [], cost_key: [], acc_key: []}
    for step, (cost, f1) in enumerate(zip(maa_cost, micro_f1)):
        if cost is not None and f1 is not None:
            results["step"].append(step)
            results[cost_key].append(cost)
            results[acc_key].append(f1)

    return results


def get_avg_f1_cost(experiment_results,
                    cost_key = "maa",
                    acc_key = "micro_f1",
                    cost_dict_name = 'cost_dict',
                    acc_dict_name = 'testing_f1_accuracy_acc_dict',
                    need_accumulated_cost=True):

    f1_cost_dict_list = []

    for result_id in range(len(experiment_results)):
        result = experiment_results[result_id]
        cost_dict, accuracy_acc_dict = result[cost_dict_name], result[acc_dict_name]
        f1_cost_dict = get_filtered_f1_cost(cost_dict, accuracy_acc_dict, cost_key = cost_key, acc_key = acc_key,
                                           need_accumulated_cost=need_accumulated_cost)
        f1_cost_dict_list.append(f1_cost_dict)

    common_steps = None
    for instance in f1_cost_dict_list:
        if common_steps is None:
            common_steps = set(instance["step"])
        common_steps &= set(instance["step"])

    # print(common_steps)

    step_costs = {}
    step_f1 = {}
    for instance in f1_cost_dict_list:
        for ith, step in enumerate(instance["step"]):
            if step in common_steps:
                if step in step_costs:
                    step_costs[step].append(instance[cost_key][ith])
                else:
                    step_costs[step] = [instance[cost_key][ith]]

                if step in step_f1:
                    step_f1[step].append(instance[acc_key][ith])
                else:
                    step_f1[step] = [instance[acc_key][ith]]

    step_avg_costs = {}
    for key in step_costs:
        step_avg_costs[key] = sum(step_costs[key])/len(step_costs[key])

    step_avg_f1 = {}
    for key in step_f1:
        step_avg_f1[key] = sum(step_f1[key])/len(step_f1[key])

    avg_results = {"step":[], cost_key:[], acc_key:[]}
    for step in sorted(common_steps):
        avg_results["step"].append(step)
        avg_results[cost_key].append(step_avg_costs[step])
        avg_results[acc_key].append(step_avg_f1[step])

    return avg_results


def get_training_f1_dict_with_interval(results, sim_threshold, interval=1):
    def _add_f1_accuracy(accuracy_list, accuracy_dict=None):
        if accuracy_dict is None:
            accuracy_dict = {"micro_recall": [], "micro_precision": [], "micro_f1": [], "report": []}
        micro_recall, micro_precision, micro_f1, report = accuracy_list
        accuracy_dict["micro_recall"].append(micro_recall)
        accuracy_dict["micro_precision"].append(micro_precision)
        accuracy_dict["micro_f1"].append(micro_f1)
        accuracy_dict["report"].append(report)

        return accuracy_dict

    def _merge_interval(results, interval, start=1):
        new_selected_true_labels = []
        new_selected_pred_labels = []
        tmp_true_list = []
        tmp_pred_list = []
        cnt = 0
        for instance in results[start:]:
            cnt += 1
            # cal traing cost, train accuracy
            selected_true_labels = format_labels(instance["selected_sents"], need_unpackage=True, is_merge=True)
            selected_pred_labels = format_labels(instance["selected_pred_labels"], is_merge=True)
            tmp_true_list.extend(selected_true_labels)
            tmp_pred_list.extend(selected_pred_labels)
            if cnt % interval == 0:
                new_selected_true_labels.append(tmp_true_list)
                new_selected_pred_labels.append(tmp_pred_list)
                tmp_true_list = []
                tmp_pred_list = []
                cnt = 0
        return new_selected_true_labels, new_selected_pred_labels

    new_selected_true_labels, new_selected_pred_labels = _merge_interval(results, interval, start=1)

    training_accuracy_dict = None
    for ith in range(len(new_selected_true_labels)):
        # cal traing cost, train accuracy
        # get cost
        selected_true_labels = new_selected_true_labels[ith]
        selected_pred_labels = new_selected_pred_labels[ith]

        cpe = CostPerformaceEval(selected_true_labels, selected_pred_labels, sim_threshold=sim_threshold)
        accuracy_list = cpe.get_f1_report()
        training_accuracy_dict = _add_f1_accuracy(accuracy_list, accuracy_dict=training_accuracy_dict)

    return training_accuracy_dict


def get_training_feature_cnt_with_interval(results, interval=1):
    def _add_f1_accuracy(accuracy_list, accuracy_dict=None):
        if accuracy_dict is None:
            accuracy_dict = {"micro_recall": [], "micro_precision": [], "micro_f1": [], "report": []}
        micro_recall, micro_precision, micro_f1, report = accuracy_list
        accuracy_dict["micro_recall"].append(micro_recall)
        accuracy_dict["micro_precision"].append(micro_precision)
        accuracy_dict["micro_f1"].append(micro_f1)
        accuracy_dict["report"].append(report)

        return accuracy_dict

    def _merge_interval(results, interval, start=1):
        new_selected_true_labels = []
        new_selected_pred_labels = []
        tmp_true_list = []
        tmp_pred_list = []
        cnt = 0
        for instance in results[start:]:
            cnt += 1
            # cal traing cost, train accuracy
            selected_true_labels = format_labels(instance["selected_sents"], need_unpackage=True, is_merge=True)
            selected_pred_labels = format_labels(instance["selected_pred_labels"], is_merge=True)
            tmp_true_list.extend(selected_true_labels)
            tmp_pred_list.extend(selected_pred_labels)
            if cnt % interval == 0:
                new_selected_true_labels.append(tmp_true_list)
                new_selected_pred_labels.append(tmp_pred_list)
                tmp_true_list = []
                tmp_pred_list = []
                cnt = 0
        return new_selected_true_labels, new_selected_pred_labels

    def _cnt_entity(labels):
        cnt = 0
        for row in labels:
            entities = get_entities(row)
            cnt += len(entities)
        return cnt

    new_selected_true_labels, new_selected_pred_labels = _merge_interval(results, interval, start=1)

    training_fea_cnt_dict = {"feature_cnt":[]}
    for ith in range(len(new_selected_true_labels)):
        # cal traing cost, train accuracy
        # get cost
        selected_true_labels = new_selected_true_labels[ith]
        training_fea_cnt_dict["feature_cnt"].append(_cnt_entity(selected_true_labels))

    return training_fea_cnt_dict


def get_all_results(paths, sim_threshold, time_para, interval=1):
    experiment_results = {}
    for ith, path in enumerate(paths):
        with open(path, "rb") as file:
            results = pkl.load(file)

        cost_dict, action_cnt = get_cost(results, time_para, sim_threshold)
        training_fine_grained_accuracy_dict, testing_fine_grained_accuracy_acc_dict = get_fine_grained_acc_dict(results,
                                                                                                           sim_threshold)
        training_f1_accuracy_dict, testing_f1_accuracy_acc_dict = get_f1_accuracy_dict(results, sim_threshold)

        training_f1_accuracy_dict_with_interval = get_training_f1_dict_with_interval(results, sim_threshold,
                                                                                     interval=interval)
        training_feature_cnt_dict_with_interval = get_training_feature_cnt_with_interval(results, interval=interval)

        experiment_results[ith] = {"cost_dict": cost_dict,
                                   "action_cnt": action_cnt,
                                   "training_fine_grained_accuracy_dict": training_fine_grained_accuracy_dict,
                                   "testing_fine_grained_accuracy_acc_dict": testing_fine_grained_accuracy_acc_dict,
                                   "training_f1_accuracy_dict": training_f1_accuracy_dict,
                                   "training_f1_accuracy_dict_with_interval": training_f1_accuracy_dict_with_interval,
                                   "training_feature_cnt_dict_with_interval": training_feature_cnt_dict_with_interval,
                                   "testing_f1_accuracy_acc_dict": testing_f1_accuracy_acc_dict}
    return experiment_results

def get_avg_action_cnt(experiment_results,
                       action_cnt_name='action_cnt',
                       need_accumulated_cost=True):
    exp_action_cnt = []
    for result_id in range(len(experiment_results)):
        result = experiment_results[result_id]
        action_cnt = deepcopy(result[action_cnt_name])
        key = list(action_cnt.keys())[0]
        action_cnt["step"] = []
        for i in range(len(action_cnt[key])):
            action_cnt["step"].append(i)
            if need_accumulated_cost:
                for key in action_cnt:
                    if i < len(action_cnt[key]) - 1:
                        action_cnt[key][i + 1] += action_cnt[key][i]
        exp_action_cnt.append(action_cnt)

    common_steps = None
    for instance in exp_action_cnt:
        if common_steps is None:
            common_steps = set(instance["step"])
        common_steps &= set(instance["step"])

    action_step_exp_cnt = {}
    for key in exp_action_cnt[0]:
        action_step_exp_cnt[key] = {}

    for instance in exp_action_cnt:
        for ith, step in enumerate(instance["step"]):
            if step in common_steps:
                if step in action_step_exp_cnt:
                    for key in instance:
                        action_step_exp_cnt[key][step].append(instance[key][ith])
                else:
                    for key in instance:
                        action_step_exp_cnt[key][step] = [instance[key][ith]]

    #     print(action_step_exp_cnt)

    action_step_avg_cnt = {}
    for action_name in action_step_exp_cnt:
        action_step_avg_cnt[action_name] = {}
        for step in action_step_exp_cnt[action_name]:
            action_step_avg_cnt[action_name][step] = sum(action_step_exp_cnt[action_name][step]) / len(
                action_step_exp_cnt[action_name][step])

    avg_results = {}
    for action_name in action_step_avg_cnt:
        avg_results[action_name] = []
        for step in sorted(common_steps):
            avg_results[action_name].append(action_step_avg_cnt[action_name][step])

    avg_results["step"] = list(sorted(common_steps))

    return avg_results


def get_inividual_acc(report):
    metrics = ['recall', 'precision', 'f1-score']
    acc_report = {}
    for label in report[0]:
        acc_report[label] = {}

    for instance in report:
        if instance is not None:
            for label in instance:
                for metric in instance[label]:
                    if metric not in acc_report[label]:
                        acc_report[label][metric] = [instance[label][metric]]
                    else:
                        acc_report[label][metric].append(instance[label][metric])
        else:
            # print(instance is None)
            for label in acc_report:
                for metric in metrics:
                    acc_report[label][metric].append(None)
    return acc_report


def get_common_avg_cost_f1(f1_cost_dict_list, cost_key="maa", acc_key="f1-score"):
    common_steps = None
    for instance in f1_cost_dict_list:
        if common_steps is None:
            common_steps = set(instance["step"])
        common_steps &= set(instance["step"])

    step_costs = {}
    step_f1 = {}
    for instance in f1_cost_dict_list:
        for ith, step in enumerate(instance["step"]):
            if step in common_steps:
                if step in step_costs:
                    step_costs[step].append(instance[cost_key][ith])
                else:
                    step_costs[step] = [instance[cost_key][ith]]

                if step in step_f1:
                    step_f1[step].append(instance[acc_key][ith])
                else:
                    step_f1[step] = [instance[acc_key][ith]]

    step_avg_costs = {}
    for key in step_costs:
        step_avg_costs[key] = sum(step_costs[key]) / len(step_costs[key])

    step_avg_f1 = {}
    for key in step_f1:
        step_avg_f1[key] = sum(step_f1[key]) / len(step_f1[key])

    avg_results = {"step": [], cost_key: [], acc_key: []}
    for step in sorted(common_steps):
        avg_results["step"].append(step)
        avg_results[cost_key].append(step_avg_costs[step])
        avg_results[acc_key].append(step_avg_f1[step])

    return avg_results


def get_all_label_avg_results(experiment_results, cost_dict_name='cost_dict',
                              acc_dict_name='testing_f1_accuracy_acc_dict',
                              cost_key="maa", acc_key="f1-score"):
    all_label_avg_results = {}
    label_f1_cost_dict_list = {}
    for result_id in range(len(experiment_results)):
        result = experiment_results[result_id]
        cost_dict = result[cost_dict_name]
        report = result[acc_dict_name]['report']
        individual_acc = get_inividual_acc(report)
        for label in individual_acc:
            f1_cost_dict = get_filtered_f1_cost(cost_dict, individual_acc[label], cost_key=cost_key, acc_key=acc_key)

            if label not in label_f1_cost_dict_list:
                label_f1_cost_dict_list[label] = [f1_cost_dict]
            else:
                label_f1_cost_dict_list[label].append(f1_cost_dict)

    for label in label_f1_cost_dict_list:
        label_avg_results = get_common_avg_cost_f1(label_f1_cost_dict_list[label], cost_key=cost_key, acc_key=acc_key)
        all_label_avg_results[label] = label_avg_results

    return all_label_avg_results

def get_training_feature_cnt_hist(results, interval=1):
    def _merge_interval(results, interval, start=0):
        new_selected_true_labels = []
        new_selected_pred_labels = []
        tmp_true_list = []
        tmp_pred_list = []
        cnt = 0
        for instance in results[start:]:
            cnt += 1
            # cal traing cost, train accuracy
            selected_true_labels = format_labels(instance["selected_sents"], need_unpackage=True, is_merge=True)
            selected_pred_labels = format_labels(instance["selected_pred_labels"], is_merge=True)
            tmp_true_list.extend(selected_true_labels)
            tmp_pred_list.extend(selected_pred_labels)
            if cnt % interval == 0:
                new_selected_true_labels.append(tmp_true_list)
                new_selected_pred_labels.append(tmp_pred_list)
                tmp_true_list = []
                tmp_pred_list = []
                cnt = 0
        return new_selected_true_labels, new_selected_pred_labels

    def _cnt_entity(labels):
        cnt_list = []
        fea_types = set()
        for row in labels:
            entities = get_entities(row)
            row_fea_types = {ety[0] for ety in entities} - {"O", "o"}
            fea_types = row_fea_types | fea_types
            cnt_list.append(len(row_fea_types))
        return cnt_list, fea_types

    new_selected_true_labels, new_selected_pred_labels = _merge_interval(results, interval, start=0)

    fea_types = set()
    training_fea_type_cnt_dict = {"feature_type_cnt":[]}
    for ith in range(len(new_selected_true_labels)):
        # cal traing cost, train accuracy
        # get cost
        selected_true_labels = new_selected_true_labels[ith]
        cnt_list, iter_fea_types = _cnt_entity(selected_true_labels)
        fea_types = fea_types | iter_fea_types
        training_fea_type_cnt_dict["feature_type_cnt"].append(cnt_list)

    return training_fea_type_cnt_dict, fea_types


def get_feature_type_cnt_from_results(paths, interval=1):
    experiment_results = {}
    for ith, path in enumerate(paths):
        with open(path, "rb") as file:
            results = pkl.load(file)

        training_fea_type_cnt_dict_with_interval, fea_types = get_training_feature_cnt_hist(results, interval=interval)

        max_cnt = len(fea_types)

        iteration_cnts = {}
        for k in range(max_cnt + 1):
            name = "training_feature_cnt_dict_with_interval_at_least_{}".format(k)
            iteration_cnts[name] = []

        for iter_instance in training_fea_type_cnt_dict_with_interval["feature_type_cnt"]:
            for k in range(max_cnt + 1):
                name = "training_feature_cnt_dict_with_interval_at_least_{}".format(k)
                cnt = 0
                # print(len("num_iteration", iter_instance))
                for num in iter_instance:
                    if num >= k:
                        cnt += 1
                iteration_cnts[name].append(cnt)

        experiment_results[ith] = {"training_feature_type_cnt_dict_with_interval": iteration_cnts}

    return experiment_results


def get_avg_at_least_feature_type_cnt(experiment_results,
                       name='training_feature_type_cnt_dict_with_interval',
                       need_accumulated_cost=True):
    exp_feature_type_cnt = []
    for result_id in range(len(experiment_results)):
        result = experiment_results[result_id]
        feature_type_cnt = deepcopy(result[name])
        key = list(feature_type_cnt.keys())[0]
        feature_type_cnt["step"] = []
        for i in range(len(feature_type_cnt[key])):
            feature_type_cnt["step"].append(i)
            if need_accumulated_cost:
                for key in feature_type_cnt:
                    if i < len(feature_type_cnt[key]) - 1:
                        feature_type_cnt[key][i + 1] += feature_type_cnt[key][i]
        exp_feature_type_cnt.append(feature_type_cnt)

    common_steps = None
    for instance in exp_feature_type_cnt:
        if common_steps is None:
            common_steps = set(instance["step"])
        common_steps &= set(instance["step"])

    feature_type_step_exp_cnt = {}
    for key in exp_feature_type_cnt[0]:
        feature_type_step_exp_cnt[key] = {}

    for instance in exp_feature_type_cnt:
        for ith, step in enumerate(instance["step"]):
            if step in common_steps:
                if step in feature_type_step_exp_cnt:
                    for key in instance:
                        feature_type_step_exp_cnt[key][step].append(instance[key][ith])
                else:
                    for key in instance:
                        feature_type_step_exp_cnt[key][step] = [instance[key][ith]]

    action_step_avg_cnt = {}
    for name in feature_type_step_exp_cnt:
        action_step_avg_cnt[name] = {}
        for step in feature_type_step_exp_cnt[name]:
            action_step_avg_cnt[name][step] = sum(feature_type_step_exp_cnt[name][step]) / len(
                feature_type_step_exp_cnt[name][step])

    avg_results = {}
    for name in action_step_avg_cnt:
        avg_results[name] = []
        for step in sorted(common_steps):
            avg_results[name].append(action_step_avg_cnt[name][step])

    avg_results["step"] = list(sorted(common_steps))

    return avg_results


###############################################################################################

from model.Doc2Piece import Doc2Piece

def count_para_per_doc(adoc, end_symbol=["\n"]):
    # 在GL这个数据集中，文档总是以"[\n, \n]"结尾，划分段落的时候会多出一个句子
    tokens, labels = Doc2Piece.unpackage([adoc])
    new_docs = Doc2Piece.package([tokens, labels])

    doc2piece = Doc2Piece(new_docs, tokenizer=None, end_symbol=end_symbol, max_length=10000, split_way="symbol")
    #len(doc2piece.segmented_sents)

    #print(doc2piece.segmented_sents[-2:])

    return len(doc2piece.segmented_sents)


def count_para(sents, end_symbol):
    tokens, labels = Doc2Piece.unpackage(sents)
    tokens = [["\n" if token == "[newline]" else token for token in row] for row in tokens]
    new_docs = Doc2Piece.package([tokens, labels])

    cnts = []
    for adoc in new_docs:
        cnt = count_para_per_doc(adoc, end_symbol=end_symbol)
        cnts.append(cnt)
    return cnts


def count_para_with_fea_per_doc(adoc, end_symbol=["\n"]):
    tokens, labels = Doc2Piece.unpackage([adoc])
    new_docs = Doc2Piece.package([tokens, labels])

    doc2piece = Doc2Piece(new_docs, tokenizer=None, end_symbol=end_symbol, max_length=10000, split_way="symbol")
    sent_tokens, sent_labels, q = Doc2Piece.unpackage(doc2piece.segmented_sents)
    # dd = [t for row in q for t in row ]
    # print(len(dd), sum(dd))
    cnt = 0
    for a_sent in sent_labels:
        if len(set(a_sent) - {'O'} - {'o'}) > 0:
            cnt += 1

    return cnt


def count_para_with_fea(sents, end_symbol):
    tokens, labels = Doc2Piece.unpackage(sents)
    tokens = [["\n" if token == "[newline]" else token for token in row] for row in tokens]
    new_docs = Doc2Piece.package([tokens, labels])

    cnts = []
    for adoc in new_docs:
        cnt = count_para_with_fea_per_doc(adoc, end_symbol=end_symbol)
        cnts.append(cnt)
    return cnts


def count_tokens_per_doc(adoc, end_symbol=["\n"]):
    tokens, labels = Doc2Piece.unpackage([adoc])
    new_docs = Doc2Piece.package([tokens, labels])

    doc2piece = Doc2Piece(new_docs, tokenizer=None, end_symbol=end_symbol, max_length=10000, split_way="symbol")
    sent_tokens, sent_labels, q = Doc2Piece.unpackage(doc2piece.segmented_sents)
    # dd = [t for row in q for t in row ]
    # print(len(dd), sum(dd))
    cnt = 0
    for row in sent_tokens:
        cnt += len(row)

    return cnt


def count_tokens(sents, end_symbol):
    tokens, labels = Doc2Piece.unpackage(sents)
    tokens = [["\n" if token == "[newline]" else token for token in row] for row in tokens]
    new_docs = Doc2Piece.package([tokens, labels])

    cnts = []
    for adoc in new_docs:
        cnt = count_tokens_per_doc(adoc, end_symbol=end_symbol)
        cnts.append(cnt)
    return cnts


def get_anotation_para_count(variables, end_symbol=["\n"]):

    para_count_list = []
    for ith in range(len(variables)):
        n_para_in_selected_sents = count_para(variables[ith]['selected_sents'], end_symbol=end_symbol)

        if 'orignial_sent' in variables[ith]:
            n_para_in_orignial_sents = count_para(variables[ith]['orignial_sent'], end_symbol=end_symbol)
        else:
            n_para_in_orignial_sents = n_para_in_selected_sents

        sum_n_selected_sents = sum(n_para_in_selected_sents)
        sum_n_orignial_sents = 0
        if len(n_para_in_orignial_sents) == 0:
            sum_n_orignial_sents = sum_n_selected_sents
        else:
            sum_n_orignial_sents = sum(n_para_in_orignial_sents)

        sum_n_removed_sents = sum_n_orignial_sents - sum_n_selected_sents

        para_count_list.append([sum_n_selected_sents, sum_n_removed_sents, sum_n_orignial_sents])
    return para_count_list


def get_anotation_para_with_fea_count(variables, end_symbol=["\n"]):

    para_with_fea_count_list = []
    for ith in range(len(variables)):
        n_para_with_fea_in_selected_sents = count_para_with_fea(variables[ith]['selected_sents'], end_symbol=end_symbol)

        if 'orignial_sent' in variables[ith]:
            n_para_with_fea_in_orignial_sents = count_para_with_fea(variables[ith]['orignial_sent'],
                                                                    end_symbol=end_symbol)
        else:
            n_para_with_fea_in_orignial_sents = n_para_with_fea_in_selected_sents

        sum_n_selected_sents_with_fea = sum(n_para_with_fea_in_selected_sents)
        sum_n_orignial_sents_with_fea = 0
        if len(n_para_with_fea_in_orignial_sents) == 0:
            sum_n_orignial_sents_with_fea = sum_n_selected_sents_with_fea
        else:
            sum_n_orignial_sents_with_fea = sum(n_para_with_fea_in_orignial_sents)

        sum_n_removed_sents_with_fea = sum_n_orignial_sents_with_fea - sum_n_selected_sents_with_fea

        para_with_fea_count_list.append(
            [sum_n_selected_sents_with_fea, sum_n_removed_sents_with_fea, sum_n_orignial_sents_with_fea])
    return para_with_fea_count_list


def get_anotation_tokens_count(variables, end_symbol=["\n"]):

    tokens_count_list = []
    for ith in range(len(variables)):
        n_tokens_in_selected_sents = count_tokens(variables[ith]['selected_sents'], end_symbol=end_symbol)

        if 'orignial_sent' in variables[ith]:
            n_tokens_in_orignial_sents = count_tokens(variables[ith]['orignial_sent'], end_symbol=end_symbol)
        else:
            n_tokens_in_orignial_sents = n_tokens_in_selected_sents

        sum_n_selected_sents = sum(n_tokens_in_selected_sents)
        sum_n_orignial_sents = 0
        if len(n_tokens_in_orignial_sents) == 0:
            sum_n_orignial_sents = sum_n_selected_sents
        else:
            sum_n_orignial_sents = sum(n_tokens_in_orignial_sents)

        sum_n_removed_sents = sum_n_orignial_sents - sum_n_selected_sents

        tokens_count_list.append([sum_n_selected_sents, sum_n_removed_sents, sum_n_orignial_sents])
    return tokens_count_list


def get_filtered_paragraph_characteristics(results, end_symbol):


    para_count = get_anotation_para_count(results, end_symbol)

    para_with_fea_count = get_anotation_para_with_fea_count(results, end_symbol)

    tokens_count = get_anotation_tokens_count(results, end_symbol)

    characteristics = {'n_tokens_in_selected_paragraphs': [row[0] for row in tokens_count],
                  'n_tokens_in_removed_paragraphs': [row[1] for row in tokens_count],
                  'n_tokens_in_all_paragraphs': [row[2] for row in tokens_count],

                  'n_selected_paragraphs_with_features': [row[0] for row in para_with_fea_count],
                  'n_removed_paragraphs_with_features': [row[1] for row in para_with_fea_count],
                  'n_all_paragraphs_with_features': [row[2] for row in para_with_fea_count],

                  'n_selected_paragraphs': [row[0] for row in para_count],
                  'n_removed_paragraphs': [row[1] for row in para_count],
                  'n_all_paragraphs': [row[2] for row in para_count]
                  }

    return characteristics

def get_all_results_filtered_paragraph_characteristics(paths, end_symbol):
    experiment_results = {}
    for ith, path in enumerate(paths):
        with open(path, "rb") as file:
            results = pkl.load(file)

        filtered_paragraph_characteristics = get_filtered_paragraph_characteristics(results, end_symbol)

        experiment_results[ith] = {"filtered_paragraph_characteristics": filtered_paragraph_characteristics}

    return experiment_results

###############################################################################################

def get_ap_f1_accuracy_dict(results, sim_threshold):
    def _add_f1_accuracy(accuracy_list, accuracy_dict=None):
        if accuracy_dict is None:
            # accuracy_dict = {"micro_recall": [], "micro_precision": [], "micro_f1": [], "report": []}
            accuracy_dict = {"micro_recall": [], "micro_precision": [], "micro_f1": []}
        # micro_recall, micro_precision, micro_f1, report = accuracy_list
        micro_recall, micro_precision, micro_f1 = accuracy_list
        accuracy_dict["micro_recall"].append(micro_recall)
        accuracy_dict["micro_precision"].append(micro_precision)
        accuracy_dict["micro_f1"].append(micro_f1)
        # accuracy_dict["report"].append(report)

        return accuracy_dict

    def _get_ap_f1_acc(action_labels, test_preds_actions, considered_labels):
        y_true = [x for row in action_labels for x in row]
        y_pred = [x for row in test_preds_actions for x in row]
        # considered_labels = list(set([x for x in y_true]) - {"O", "o"})
        recall = recall_score(y_true, y_pred, labels=considered_labels, average="micro")
        precision = precision_score(y_true, y_pred, labels=considered_labels, average="micro")
        # f1 = f1_score(y_true, y_pred, labels = considered_labels, average = "micro")
        if (recall == 0 and precision == 0) or recall == np.nan or precision == np.nan:
            f1 = np.nan
        else:
            f1 = 2 * recall * precision / (precision + recall)

        return recall, precision, f1

    nan_report = {'B-CONFIRMATION': {'recall': np.nan, 'precision': np.nan, 'f1-score': np.nan},
                  'B-DELETEADD': {'recall': np.nan, 'precision': np.nan, 'f1-score': np.nan},
                  'B-REVISE': {'recall': np.nan, 'precision': np.nan, 'f1-score': np.nan},
                  'B-DELETE': {'recall': np.nan, 'precision': np.nan, 'f1-score': np.nan},
                  'B-ADD': {'recall': np.nan, 'precision': np.nan, 'f1-score': np.nan}}

    training_accuracy_dict = None
    testing_accuracy_acc_dict = None
    considered_labels = ['B-CONFIRMATION', 'B-DELETEADD', 'B-REVISE', 'B-DELETE', 'B-ADD']

    test_sents = results[0]['test_sents']
    test_true_labels = format_labels(test_sents, need_unpackage=True)
    for ith, instance in enumerate(results):
        # cal traing cost, train accuracy
        if len(instance["selected_sents"]) > 0 and instance["selected_preds_actions"] is not None:
            # get cost
            selected_sents = instance["selected_sents"]
            selected_sent_true_labels = format_labels(selected_sents, need_unpackage=True, is_merge=True)
            selected_pred_labels = instance["selected_pred_labels"]
            cpe = CostPerformaceEval(selected_sent_true_labels, selected_pred_labels, sim_threshold=sim_threshold)
            selected_true_action_labels = [aa.action_labels for aa in cpe.annotaion_actions]
            selected_preds_actions = instance["selected_preds_actions"]
            accuracy_list = _get_ap_f1_acc(selected_true_action_labels, selected_preds_actions, considered_labels)
            training_accuracy_dict = _add_f1_accuracy(accuracy_list, accuracy_dict=training_accuracy_dict)
        else:
            if ith == 0:
                # accuracy_list = [np.nan, np.nan, np.nan, nan_report]
                accuracy_list = [np.nan, np.nan, np.nan]
            else:
                accuracy_list = [None] * len(training_accuracy_dict)
            training_accuracy_dict = _add_f1_accuracy(accuracy_list, accuracy_dict=training_accuracy_dict)

        # cal test accuracy
        if 'test_preds_actions' in instance:
            test_sent_pred_labels = instance['test_sent_pred_labels']
            test_preds_actions = instance['test_preds_actions']
            test_sent_true_labels = format_labels(test_sents, need_unpackage=True, is_merge=True)
            test_sent_pred_labels = format_labels(test_sent_pred_labels, is_merge=True)
            cpe = CostPerformaceEval(test_sent_true_labels, test_sent_pred_labels, sim_threshold=sim_threshold)
            action_labels = [aa.action_labels for aa in cpe.annotaion_actions]
            accuracy_list = _get_ap_f1_acc(action_labels, test_preds_actions, considered_labels)
            testing_accuracy_acc_dict = _add_f1_accuracy(accuracy_list, accuracy_dict=testing_accuracy_acc_dict)
        else:
            if ith == 0:
                # accuracy_list = [np.nan, np.nan, np.nan, nan_report]
                accuracy_list = [np.nan, np.nan, np.nan]
            else:
                accuracy_list = [None] * len(testing_accuracy_acc_dict)
            testing_accuracy_acc_dict = _add_f1_accuracy(accuracy_list, accuracy_dict=testing_accuracy_acc_dict)

    training_accuracy_dict = {"train_ap_{}".format(key): training_accuracy_dict[key] for key in training_accuracy_dict}
    testing_accuracy_acc_dict = {"test_ap_{}".format(key): testing_accuracy_acc_dict[key] for key in
                                 testing_accuracy_acc_dict}
    return training_accuracy_dict, testing_accuracy_acc_dict


def get_all_results_filtered_ap_f1_accuracy_dict(paths, sim_threshold):
    experiment_results = {}
    for ith, path in enumerate(paths):
        with open(path, "rb") as file:
            results = pkl.load(file)

        filtered_ap_f1_accuracy_dict = get_ap_f1_accuracy_dict(results, sim_threshold)

        experiment_results[ith] = {"trained_filtered_ap_f1_accuracy_dict": filtered_ap_f1_accuracy_dict[0],
                                   "test_filtered_ap_f1_accuracy_dict": filtered_ap_f1_accuracy_dict[1]}

    return experiment_results