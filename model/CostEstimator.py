from seqeval.metrics.sequence_labeling import get_entities
from copy import deepcopy

class AnnotationAction(object):
    actions = ["O", "B-CONFIRMATION", "B-DELETE", "B-ADD", "B-REVISE", "B-DELETEADD"]
    actions_name2name = {x:x for x in actions}
    action2idx = {x: ith for ith, x in enumerate(actions)}
    # 重新写一个patch文件

    def __init__(self, true_labels, pred_labels, threshold=0, sim_method="not_intersection"):
        assert len(true_labels) == len(pred_labels)
        self.threshold = threshold  # allow equal
        self.confirm_annotations = None
        self.revise_annotations = None
        self.delete_annotations = None
        self.add_annotations = None
        self.action_cnt = None
        self.action_labels = None
        self.true_labels = deepcopy(true_labels)
        self.pred_labels = deepcopy(pred_labels)
        self.length = len(true_labels)
        self.sim_method = sim_method
        self.get_4kind_annotations()
        self.get_action_cnt()
        self.get_action_sequences()

    def is_span_intersected(self, span1, span2):
        s1, e1 = span1
        s2, e2 = span2
        is_intersected = not (e2 < s1 or e1 < s2)
        return is_intersected

    def is_span_similar(self, span1, span2):
        s1, e1 = span1
        s2, e2 = span2

        total_char_if_not_intersect = (e1 - s1 + 1) + (e2 - s2 + 1)
        if self.is_span_intersected(span1, span2):
            n_not_intersect = abs(s1 - s2) + abs(e1 - e2)
        else:
            n_not_intersect = total_char_if_not_intersect

        n_intersect = (total_char_if_not_intersect - n_not_intersect) / 2
        n_union = n_intersect + n_not_intersect

        is_similar = False
        if self.sim_method == "not_intersection":
            if n_intersect > 0 and n_not_intersect <= self.threshold:
                is_similar = True
        elif self.sim_method == "jaccard":
            if n_union != 0 and n_intersect / n_union >= self.threshold:
                is_similar = True
        else:
            raise NotImplementedError("NotImplementedError in AnnoationAction.is_span_similar")

        return is_similar

    def get_oneByOne(self, dictA, refer_dict=None):
        # Only one-to-one matching of predicted and test entities is allowed

        one2one = {}
        for key in dictA:
            if len(dictA[key]) == 1:
                one2one[key] = dictA[key][0]

        if refer_dict is not None:
            for key in refer_dict:
                if len(refer_dict[key]) != 1:
                    for span in refer_dict[key]:
                        if span in one2one:
                            del one2one[span]

        return one2one

    def get_sim_one2one(self, one2one):
        new_one2one = {}
        for pred_span in one2one:
            true_span = one2one[pred_span]
            if self.is_span_similar(true_span, pred_span):
                new_one2one[pred_span] = true_span
        return new_one2one

    def get_correct_span(self, true_span_list, pred_span_list):
        # a span = (start, end)
        # 返回一个dict, key是一个预测的span, value是一个真实的span
        # 步骤
        # 1. Construction of similarity matrix
        correct2pred, pred2correct = self.get_interested_span_dict(true_span_list, pred_span_list)
        # 2. Filter many-to-many relationships
        one2one = self.get_oneByOne(pred2correct, refer_dict=correct2pred)
        # 3. Keep only the entities that are similar to each other, 1:1
        span_pred2true = self.get_sim_one2one(one2one)
        return span_pred2true

    def get_interested_span_dict(self, true_span_list, pred_span_list):
        true2pred, pred2true = {}, {}
        true_adjacency = [[] for i in range(self.length)]
        pred_adjacency = [[] for i in range(self.length)]
        for span in true_span_list:
            for i in range(span[0], span[1] + 1):
                true_adjacency[i].append(span)

        for span in pred_span_list:
            for i in range(span[0], span[1] + 1):
                pred_adjacency[i].append(span)

        for i in range(self.length):
            true_spans = true_adjacency[i]
            pred_spans = pred_adjacency[i]
            for span in true_spans:
                if span not in true2pred:
                    true2pred[span] = []
                true2pred[span].extend(pred_spans)
            for span in pred_spans:
                if span not in pred2true:
                    pred2true[span] = []
                pred2true[span].extend(true_spans)

        # remove redundant items
        for span in true2pred:
            true2pred[span] = list(set(true2pred[span]))

        for span in pred2true:
            #             print("pred_spans[span] = list(set(pred_spans[span])):", pred_spans)
            #             print("pred_spans[span] = list(set(pred_spans[span])):", span)
            pred2true[span] = list(set(pred2true[span]))

        return true2pred, pred2true

    def get_4kind_annotations(self):
        # add_annotations stores true spans,
        # confirm_annotations, revise_annotations and delete_annotations store predicted span
        confirm_annotations = set()
        revise_annotations = set()
        delete_annotations = set()
        add_annotations = set()
        # extract entities
        true_ety_list = get_entities(self.true_labels)
        pred_ety_list = get_entities(self.pred_labels)
        # span -> label mapping
        true_span2lbael = {(ety[1], ety[2]): ety[0] for ety in true_ety_list}
        pred_span2lbael = {(ety[1], ety[2]): ety[0] for ety in pred_ety_list}

        # Get the correct span dictionary, pred map to correct
        true_spans = true_span2lbael.keys()
        pred_spans = pred_span2lbael.keys()
        span_pred2true = self.get_correct_span(true_spans, pred_spans)

        # confirmation_annotations and revise_annotations
        for pred_span in span_pred2true:
            true_span = span_pred2true[pred_span]
            true_label = true_span2lbael[true_span]
            pred_label = pred_span2lbael[pred_span]
            if true_label == pred_label:
                confirm_annotations.add((pred_span, true_label))
            else:
                revise_annotations.add((pred_span, true_label, pred_label))

        # delete_annotations
        correct_pred_spans = span_pred2true.keys()
        delete_spans = set(pred_spans) - set(correct_pred_spans)
        for pred_span in delete_spans:
            pred_label = pred_span2lbael[pred_span]
            delete_annotations.add((pred_span, pred_label))

        # add_annotations
        correct_true_spans = span_pred2true.values()
        add_spans = set(true_spans) - set(correct_true_spans)
        for true_span in add_spans:
            true_label = true_span2lbael[true_span]
            add_annotations.add((true_span, true_label))

        self.confirm_annotations = confirm_annotations
        self.revise_annotations = revise_annotations
        self.delete_annotations = delete_annotations
        self.add_annotations = add_annotations

    def get_action_cnt(self):
        confirm_cnt = len(self.confirm_annotations)
        revise_cnt = len(self.revise_annotations)
        delete_cnt = len(self.delete_annotations)
        add_cnt = len(self.add_annotations)
        read_cnt = self.length
        self.action_cnt = {"confirm_cnt": confirm_cnt, "revise_cnt": revise_cnt,
                                        "delete_cnt": delete_cnt, "add_cnt": add_cnt, "read_cnt": read_cnt}

    def get_action_sequences(self):

        if self.confirm_annotations is None:
            self.get_4kind_annotations()

        none_label = AnnotationAction.actions_name2name["O"]
        action_seq = [none_label] * self.length

        for span, label in self.confirm_annotations:
            s = span[0]
            action_seq[s] = AnnotationAction.actions_name2name["B-CONFIRMATION"]

        for span, _, _ in self.revise_annotations:
            s = span[0]
            action_seq[s] = AnnotationAction.actions_name2name["B-REVISE"]

        for span, _ in self.delete_annotations:
            s = span[0]
            action_seq[s] = AnnotationAction.actions_name2name["B-DELETE"]

        for span, _ in self.add_annotations:
            s = span[0]
            #         print(s)
            if action_seq[s] == AnnotationAction.actions_name2name["B-DELETE"]:
                action_seq[s] = AnnotationAction.actions_name2name["B-DELETEADD"]
            elif action_seq[s] == none_label:
                action_seq[s] = AnnotationAction.actions_name2name["B-ADD"]
            else:
                print(action_seq[s])
                raise NotImplementedError("unexpected error")

        self.action_labels = action_seq


class ActionCountFromActions(object):
    def __init__(self, action_labels):
        self.action_labels = deepcopy(action_labels)
        self.action_cnt = None
        self.get_action_cnt_from_action()

    def get_action_cnt_from_action(self):
        action_cnt = {"confirm_cnt": 0, "revise_cnt": 0, "delete_cnt": 0, "add_cnt": 0, "read_cnt": len(self.action_labels)}

        for label in self.action_labels:
            if "confirm" in label.lower():
                action_cnt["confirm_cnt"] += 1
            if "revise" in label.lower():
                action_cnt["revise_cnt"] += 1
            if "delete" in label.lower():
                action_cnt["delete_cnt"] += 1
            if "add" in label.lower():
                action_cnt["add_cnt"] += 1

        self.action_cnt = action_cnt


class CostPerformaceEval(object):
    def __init__(self, true_label_list, pred_labels_list, sim_threshold=0, sim_method="not_intersection"):
        self.annotaion_actions = []
        for true_labels, pred_labels in zip(true_label_list, pred_labels_list):
            aa = AnnotationAction(true_labels, pred_labels, threshold=sim_threshold, sim_method=sim_method)
            self.annotaion_actions.append(aa)

        # remove prefix B-, I-, E-, S-
        self.unique_labels_without_prefix = [tag.split("-")[-1] for true_labels, pred_labels in zip(true_label_list, pred_labels_list)
                                                    for tag in(true_labels + pred_labels) if tag != "O"]

    def get_instance_correction_cost(self, time_para):
        instance_cost_list = []
        for aa in self.annotaion_actions:
            action_cnt = aa.action_cnt
            confirm_cost = action_cnt["confirm_cnt"] * time_para["confirm"]
            revise_cost = action_cnt["revise_cnt"] * time_para["revise"]
            delete_cost = action_cnt["delete_cnt"] * time_para["delete"]
            add_cost = action_cnt["add_cnt"] * time_para["add"]
            read_cost = action_cnt["read_cnt"] * time_para["read"]
            total_cost = confirm_cost + revise_cost + delete_cost + add_cost + read_cost
            instance_cost_list.append({"confirm_cost": confirm_cost, "revise_cost": revise_cost,
                                  "delete_cost": delete_cost, "add_cost": add_cost,
                                  "read_cost": read_cost, "total_cost": total_cost})
        return instance_cost_list

    def get_instance_FMA_cost(self, time_para, need_verifier=True):
        # FMA: fully manual annotation
        instance_cost_list = []
        for aa in self.annotaion_actions:
            action_cnt = aa.action_cnt
            confirm_cost = action_cnt["confirm_cnt"] * time_para["confirm"]
            revise_cost = action_cnt["revise_cnt"] * time_para["revise"]
            delete_cost = action_cnt["delete_cnt"] * time_para["delete"]
            add_cost = action_cnt["add_cnt"] * time_para["add"]
            read_cost = action_cnt["read_cnt"] * time_para["read"]
            n_ety = action_cnt["confirm_cnt"] + action_cnt["revise_cnt"] + action_cnt["add_cnt"]
            n_token = action_cnt["read_cnt"]
            # fully mannual annotaition
            annotaion_cost = n_ety * time_para["add"] + n_token * time_para["read"]
            correction_cost = 0
            if need_verifier:
                correction_cost = n_ety * time_para["confirm"] + n_token * time_para["read"]
            total_cost = annotaion_cost + correction_cost
            instance_cost_list.append({"annotaion_cost": annotaion_cost, "correction_cost": correction_cost,
                                      "total_cost": total_cost, "confirm_cost": confirm_cost, "revise_cost": revise_cost,
                                      "delete_cost": delete_cost, "add_cost": add_cost, "read_cost": read_cost})
        return instance_cost_list

    def get_cost(self, time_para, way="FMA", need_verifier=True):
        # FMA: fully_mannual_annotation
        # MAA: Machine assisted annotation. Correction Cost
        if way == "FMA":
            instance_cost_list = self.get_instance_FMA_cost(time_para, need_verifier=need_verifier)
        elif way == "MAA":
            instance_cost_list = self.get_instance_correction_cost(time_para)
        else:
            raise NotImplementedError("{} has not been implemented in CostPerformaceEval.get_cost!".format(way))

        total_cost_dict = {}
        for instance_cost in instance_cost_list:
            for key in instance_cost:
                if key not in total_cost_dict:
                    total_cost_dict[key] = instance_cost[key]
                else:
                    total_cost_dict[key] += instance_cost[key]

        return total_cost_dict

    def get_f1_report(self):
        def cal_results(tp, total_true, total_predict):
            recall = tp * 1.0 / total_true if total_true != 0 else 0
            precision = tp * 1.0 / total_predict if total_predict != 0 else 0
            f1_score = 2.0 * precision * recall / (
                    precision + recall) if precision != 0 or recall != 0 else 0
            return recall, precision, f1_score

        def cal_results_for_each_label(confirmation_predict, revise_true, revise_predict, delete_predict, unrecognized):
            keys = self.unique_labels_without_prefix
            report = {key:{"recall":0, "precision":0, "f1-score":0} for key in keys}
            for key in keys:
                tp, total_true, total_predict = 0, 0, 0
                if key in confirmation_predict: tp = confirmation_predict[key]
                total_predict += tp
                total_true += tp
                if key in revise_true: total_true += revise_true[key]
                if key in revise_predict: total_predict += revise_predict[key]
                if key in delete_predict: total_predict += delete_predict[key]
                if key in unrecognized: total_true += unrecognized[key]
                recall, precision, f1_score = cal_results(tp, total_true, total_predict)
                report[key]["recall"] = recall
                report[key]["precision"] = precision
                report[key]["f1-score"] = f1_score
            return report

        confirmation_predict = {key:0 for key in self.unique_labels_without_prefix}
        revise_true = {key: 0 for key in self.unique_labels_without_prefix}
        revise_predict = {key:0 for key in self.unique_labels_without_prefix}
        delete_predict = {key:0 for key in self.unique_labels_without_prefix}
        unrecognized = {key:0 for key in self.unique_labels_without_prefix}
        for aa in self.annotaion_actions:
            for span, label in aa.confirm_annotations:
                confirmation_predict[label] += 1
            for span, true_label, pred_label in aa.revise_annotations:
                revise_true[true_label] += 1
                revise_predict[pred_label] += 1
            for span, pred_label in aa.delete_annotations:
                delete_predict[pred_label] += 1
            for span, true_label in aa.add_annotations:
                unrecognized[true_label] += 1

        total_confirm = sum(confirmation_predict.values())
        total_revise_true = sum(revise_true.values())
        total_revise_predict = sum(revise_predict.values())
        total_true = sum(unrecognized.values()) + total_confirm + total_revise_true
        total_predict = sum(delete_predict.values()) + total_confirm + total_revise_predict

        micro_recall, micro_precision, micro_f1 = cal_results(total_confirm, total_true, total_predict)

        report = cal_results_for_each_label(confirmation_predict, revise_true, revise_predict, delete_predict, unrecognized)
        report["micro_avg"] = {}
        report["micro_avg"]["recall"] = micro_recall
        report["micro_avg"]["precision"] = micro_precision
        report["micro_avg"]["f1-score"] = micro_f1

        return micro_recall, micro_precision, micro_f1, report

    def get_annotation_report(self):
        def cal_annotation_metrics(total_confirm, total_revise_true, total_revise_predict, total_delete_predict,
                                   total_add_true, total_true, total_predict):
            confirm_divided_true = total_confirm * 1.0 / total_true if total_true != 0 else 0
            confirm_divided_pred = total_confirm * 1.0 / total_predict if total_predict != 0 else 0
            revise_true_divided_true = total_revise_true * 1.0 / total_true if total_true != 0 else 0
            revise_pred_divided_pred = total_revise_predict * 1.0 / total_predict if total_predict != 0 else 0
            delete_divided_pred = total_delete_predict * 1.0 / total_predict if total_predict != 0 else 0
            add_divided_true = total_add_true * 1.0 / total_true if total_true != 0 else 0
            return confirm_divided_true, confirm_divided_pred, revise_true_divided_true, revise_pred_divided_pred, delete_divided_pred, add_divided_true

        def cal_results_for_each_label(confirmation_predict, revise_true, revise_predict, delete_predict, unrecognized):
            keys = self.unique_labels_without_prefix
            report = {key:{"confirm_recall":0, "revise_recall":0, "confirm_precision":0, "revise_precision":0} for key in keys}
            for key in keys:
                total_confirm, total_revise_true, total_revise_predict, total_delete_predict, total_add_true = 0, 0, 0, 0, 0
                total_true, total_predict = 0, 0
                if key in confirmation_predict: total_confirm += confirmation_predict[key]
                if key in revise_true: total_revise_true += revise_true[key]
                if key in revise_predict: total_revise_predict += revise_predict[key]
                if key in delete_predict: total_delete_predict += delete_predict[key]
                if key in unrecognized: total_add_true += unrecognized[key]
                total_true += total_confirm + total_revise_true + total_add_true
                total_predict += total_confirm + total_revise_predict + total_delete_predict

                confirm_divided_true, confirm_divided_pred, revise_true_divided_true, \
                revise_pred_divided_pred, delete_divided_pred, add_divided_true = \
                                cal_annotation_metrics(total_confirm, total_revise_true, total_revise_predict,
                                                    total_delete_predict, total_add_true, total_true, total_predict)

                report[key]["confirm_divided_true"] = confirm_divided_true
                report[key]["confirm_divided_pred"] = confirm_divided_pred
                report[key]["revise_true_divided_true"] = revise_true_divided_true
                report[key]["revise_pred_divided_pred"] = revise_pred_divided_pred
                report[key]["delete_divided_pred"] = delete_divided_pred
                report[key]["add_divided_true"] = add_divided_true
            return report

        confirmation_predict = {key:0 for key in self.unique_labels_without_prefix}
        revise_true = {key: 0 for key in self.unique_labels_without_prefix}
        revise_predict = {key:0 for key in self.unique_labels_without_prefix}
        delete_predict = {key:0 for key in self.unique_labels_without_prefix}
        unrecognized = {key:0 for key in self.unique_labels_without_prefix}
        for aa in self.annotaion_actions:
            for span, label in aa.confirm_annotations:
                confirmation_predict[label] += 1
            for span, true_label, pred_label in aa.revise_annotations:
                revise_true[true_label] += 1
                revise_predict[pred_label] += 1
            for span, pred_label in aa.delete_annotations:
                delete_predict[pred_label] += 1
            for span, true_label in aa.add_annotations:
                unrecognized[true_label] += 1

        total_confirm = sum(confirmation_predict.values())
        total_revise_true = sum(revise_true.values())
        total_revise_predict = sum(revise_predict.values())
        total_delete_predict = sum(delete_predict.values())
        total_add_true = sum(unrecognized.values())
        total_true = total_confirm + total_revise_true + total_add_true
        total_predict = total_confirm + total_revise_predict + total_delete_predict
        confirm_recall, confirm_precision, revise_recall, revise_precision, delete_precision, add_recall = cal_annotation_metrics(
                                    total_confirm, total_revise_true, total_revise_predict, total_delete_predict,
                                    total_add_true, total_true, total_predict)

        report = cal_results_for_each_label(confirmation_predict, revise_true, revise_predict, delete_predict, unrecognized)
        report["micro_avg"] = {}
        report["micro_avg"]["confirm_recall"] = confirm_recall
        report["micro_avg"]["confirm_precision"] = confirm_precision
        report["micro_avg"]["revise_recall"] = revise_recall
        report["micro_avg"]["revise_precision"] = revise_precision
        report["micro_avg"]["delete_precision"] = delete_precision
        report["micro_avg"]["add_recall"] = add_recall

        return confirm_recall, confirm_precision, revise_recall, revise_precision, delete_precision, add_recall, report

    @staticmethod
    def formatBIOES(seq, is_merge = False, required_bioes=True):
        if is_merge:
            # seq = [w.split("-")[-1] for w in seq]
            seq = ["I" + w[1:] if len(w) > 1 and w[1] == "-" else w for w in seq]

        if required_bioes:
            entity = get_entities(seq)
            seq = CostPerformaceEval.entitys2seq(entity, len(seq))
        return seq

    @staticmethod
    def entitys2seq(entitys, length):

        seq = ["O"] * length
        for label, start, end in entitys:
            if start == end:
                seq[start] = "S-" + label
                continue

            cur = start + 1
            while cur < end:
                seq[cur] = "I-" + label
                cur+=1

            seq[start] = "B-" + label
            seq[end] = "E-" + label
        return seq

class CostEvalFromAction(object):
    def __init__(self, action_label_list):
        self.action_cnt_list = []
        for actions in action_label_list:
            instance_action_cnt = ActionCountFromActions(actions)
            self.action_cnt_list.append(instance_action_cnt)

    def get_instance_correction_cost(self, time_para):
        instance_cost_cost = []
        for instance in self.action_cnt_list:
            action_cnt = instance.action_cnt
            confirm_cost = action_cnt["confirm_cnt"] * time_para["confirm"]
            revise_cost = action_cnt["revise_cnt"] * time_para["revise"]
            delete_cost = action_cnt["delete_cnt"] * time_para["delete"]
            add_cost = action_cnt["add_cnt"] * time_para["add"]
            read_cost = action_cnt["read_cnt"] * time_para["read"]
            total_cost = confirm_cost + revise_cost + delete_cost + add_cost + read_cost
            instance_cost_cost.append({"confirm_cost": confirm_cost, "revise_cost": revise_cost,
                                  "delete_cost": delete_cost, "add_cost": add_cost,
                                  "read_cost": read_cost, "total_cost": total_cost})
        return instance_cost_cost

    def get_instance_FHA_cost(self, time_para, need_verifier=True):
        instance_cost_list = []
        for intance in self.action_cnt_list:
            ## instance cost
            action_cnt = intance.action_cnt
            n_ety = action_cnt["confirm_cnt"] + action_cnt["revise_cnt"] + action_cnt["add_cnt"]
            n_token = action_cnt["read_cnt"]
            # fully mannual annotaition
            annotaion_cost = n_ety * time_para["add"] + n_token * time_para["read"]
            correction_cost = 0
            if need_verifier:
                correction_cost = n_ety * time_para["confirm"] + n_token * time_para["read"]
            total_cost = annotaion_cost + correction_cost
            instance_cost_list.append({"annotaion_cost": annotaion_cost, "correction_cost": correction_cost,
                             "total_cost": total_cost})
        return instance_cost_list

    def get_cost(self, time_para, way="FMA", need_verifier=True):
        # get sum of cost, return a dict
        # FMA: fully_mannual_annotation
        # MAA: Machine assisted annotation. Correction Cost
        if way == "FMA":
            instance_cost_list = self.get_instance_FHA_cost(time_para, need_verifier=need_verifier)
        elif way == "MAA":
            instance_cost_list = self.get_instance_correction_cost(time_para)
        else:
            raise NotImplementedError("{} has not been implemented in CostPerformaceEval.get_cost!".format(way))

        total_cost_dict = {}
        for instance_cost in instance_cost_list:
            for key in instance_cost:
                if key not in total_cost_dict:
                    total_cost_dict[key] = instance_cost[key]
                else:
                    total_cost_dict[key] += instance_cost[key]

        return total_cost_dict

