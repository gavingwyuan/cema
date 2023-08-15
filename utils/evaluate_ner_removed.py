from seqeval.metrics.sequence_labeling import get_entities
from nervaluate import Evaluator as NEREvaluator
from seqeval.metrics import classification_report as ner_classification_report
import fitlog
import json
from copy import deepcopy
from collections import defaultdict
import math
import numpy

def get_cost_according_name(time_cost_dict, cost_used):
    if cost_used == "from_preannotated":
        return time_cost_dict['total_time_cost_from_preannotated']
    elif cost_used == "from_plain":
        return time_cost_dict['total_time_cost_from_plain_text']
    elif cost_used == "only_on_preannotated":
        return time_cost_dict['total_time_cost_only_on_preannotated']
    elif cost_used == "time_cost_from_preannotated_except_read_and_confirm":
        return time_cost_dict['total_time_cost_from_preannotated_except_read_and_confirm']
    elif cost_used == "time_cost_from_preannotated_except_read":
        return time_cost_dict['total_time_cost_from_preannotated_except_read']
    elif cost_used == "add_time_cost_plain":
        return time_cost_dict['total_add_time_cost_plain']
    elif cost_used == "read_time_cost_plain":
        return time_cost_dict['total_read_time_cost_plain']
    else:
        raise NotImplementedError

def get_cost(pred_y, true_y, predifined_labels, time_option, tolerance_move, cost_used, need_all = False):
    time_cost_list = []
    for py, ty in zip (pred_y, true_y):
        nermetrics = NERMatrics([py], [ty], predifined_labels, 0, "for_getting_cost",
                                    time_option=time_option, tolerance_move=tolerance_move, stage="get_label_mode")
        time_cost_dict = nermetrics.get_time_cost()
        time_cost_dict = json.loads(json.dumps(time_cost_dict))
        if need_all:
            time_cost_list.append(time_cost_dict)
        else:
            time_cost_list.append(get_cost_according_name(time_cost_dict, cost_used))
            # if cost_used == "from_preannotated":
            #     time_cost_list.append(time_cost_dict['total_time_cost_from_preannotated'])
            # elif cost_used == "from_plain":
            #     time_cost_list.append(time_cost_dict['total_time_cost_from_plain_text'])
            # elif cost_used == "only_on_preannotated":
            #     time_cost_list.append(time_cost_dict['total_time_cost_only_on_preannotated'])
            # elif cost_used == "time_cost_from_preannotated_except_read_and_confirm":
            #     time_cost_list.append(time_cost_dict['total_time_cost_from_preannotated_except_read_and_confirm'])
            # elif cost_used == "time_cost_from_preannotated_except_read":
            #     time_cost_list.append(time_cost_dict['total_time_cost_from_preannotated_except_read'])
            # elif cost_used == "add_time_cost_plain":
            #     time_cost_list.append(time_cost_dict['total_add_time_cost_plain'])
            # elif cost_used == "read_time_cost_plain":
            #     time_cost_list.append(time_cost_dict['total_read_time_cost_plain'])
            # else:
            #     raise NotImplementedError
    return time_cost_list

def cal_cost_entropy(action_dict, cost_used, time_unit, threshold=0.85, is_pred=True, option = 0):
    name_map = {'confirm_cnt': 'total_confirm_cnt_preannotated',
                'revise_cnt': 'total_revise_cnt_preannotated',
                'delete_cnt': 'total_delete_cnt_preannotated',
                'add_cnt': 'total_add_cnt_preannotated',
                'read_cnt': 'total_total_tokens_cnt_plain',
                'add_plain_cnt': 'total_add_cnt_from_plain',

                'confirm': 'total_confirm_time_cost_preannotated',
                'revise': 'total_revise_time_cost_preannotated',
                'delete': 'total_delete_time_cost_preannotated',
                'add': 'total_add_time_cost_preannotated',
                'read': 'total_read_time_cost_plain',
                'add_plain': 'total_add_time_cost_plain'}

    cost_used_map = {"from_preannotated": ['confirm', 'revise', 'delete', 'add', 'read'],
                     "from_plain": ['add_plain', 'read']}

    sum_keys = cost_used_map[cost_used]

    cost_list = []
    for key in sum_keys:
        #         if "read" in key: continue
        if is_pred:
            key2 = key
            if key == "add_plain": key2 = "add"
            cost_list.append(action_dict[key + "_cnt"] * time_unit[key2])
        else:
            key2 = key
            if key == "add_plain": key2 = "add"
            cost_list.append(action_dict[name_map[key + "_cnt"]] * time_unit[key2])

    # sum_cost = sum(cost_list)
    # norm_cost = [c/sum_cost for c in cost_list]
    # ent = 0.
    # for c in norm_cost:
    #     if c > 0:
    #         ent -= c * math.log(c, len(cost_list))
    driven_info = 0
    p = threshold
    actions_cost = cost_list
    if option == 0:
        # confirm, revise, delete, add的比例大概是200:400:400:1500
        # read大概200的样子
        sum_cost = sum(actions_cost[:-1])
        norm_cost = [0] * (len(actions_cost) - 1)
        if sum_cost > 0:
            norm_cost = [c / sum_cost for c in actions_cost[:-1]]

        driven_info = 0.
        for c in norm_cost:
            if c > 0:
                driven_info -= c * math.log(c, len(norm_cost))
        driven_info /= actions_cost[4]

    elif option == 1:
        # 200:400:500:1600
        # 权重变小也会变小
        # read大概也是200的样子。说明这种形式下，乘上系数3意义不大，类似于同比例缩放。
        # (2 - p)起到的作用是，刚开始选的样本。confirm，revise少点，delete和add多点。
        sum_cost = sum(actions_cost[:-1])
        norm_cost = [0] * (len(actions_cost) - 1)
        if sum_cost > 0:
            norm_cost = [c / sum_cost for c in actions_cost[:-1]]

        driven_info = 0.
        for c in norm_cost:
            if c > 0:
                driven_info -= c * math.log(c, len(norm_cost))
        driven_info /= 3 * ((2 - p) * actions_cost[4])

    elif option == 2:
        # 用权重的方式不能直观的控制时间比例
        # 1, 3
        # 2, 8
        # (-3 / 6 * np.log(3 / 6) - 3 / 6 * np.log(3 / 6)) - (-6 / 14 * np.log(6 / 14) - 8 / 14 * np.log(8 / 14))
        # (-2 / 5 * np.log(2 / 5) - 3 / 5 * np.log(3 / 5)) - (-4 / 12 * np.log(4 / 12) - 8 / 12 * np.log(8 / 12))
        # (-1 / 4 * np.log(1 / 4) - 3 / 4 * np.log(3 / 4)) - (-2 / 10 * np.log(2 / 10) - 8 / 10 * np.log(8 / 10))
        # (-0.5 / 3.5 * np.log(0.5 / 3.5) - 3 / 3.5 * np.log(3 / 3.5)) - (-1 / 9 * np.log(1 / 9) - 8 / 9 * np.log(8 / 9))
        # (-0.25 / 3.25 * np.log(0.25 / 3.25) - 3 / 3.25 * np.log(3 / 3.25)) - (
        #             -0.5 / 8.5 * np.log(0.5 / 8.5) - 8 / 8.5 * np.log(8 / 8.5))
        # 120:500:550:1800
        # 乘上小于1的系数，会使得cost变小
        # read cost差不多
        driven_info = 0.
        driven_info_1 = 0.
        driven_info_2 = 0.
        new_cost_list = deepcopy(actions_cost)
        new_cost_list[0] *= 0.25
        new_cost_list[1] *= 0.5

        sum_cost = sum(new_cost_list[:-1])
        norm_cost = [0] * (len(new_cost_list) - 1)
        if sum_cost > 0:
            norm_cost = [c / sum_cost for c in new_cost_list[:-1]]

        driven_info = 0.
        for c in norm_cost:
            if c > 0:
                driven_info -= c * math.log(c, len(norm_cost))
        driven_info /= 3 * ((2 - p) * actions_cost[4])
    elif option == 3:
        # 150:570:500:1500
        # 比之前confirm cost多一些了
        # revise变多了
        driven_info = 0.
        driven_info_1 = 0.
        driven_info_2 = 0.
        new_cost_list = deepcopy(actions_cost)
        if p > 0.85:
            new_cost_list[0] *= 1.25
        else:
            new_cost_list[0] *= 0.25
        new_cost_list[1] *= 0.5

        sum_cost = sum(new_cost_list[:-1])
        norm_cost = [0] * (len(new_cost_list) - 1)
        if sum_cost > 0:
            norm_cost = [c / sum_cost for c in new_cost_list[:-1]]

        driven_info = 0.
        for c in norm_cost:
            if c > 0:
                driven_info -= c * math.log(c, len(norm_cost))
        driven_info /= 3 * ((2 - p) * actions_cost[4])
    elif option == 4:
        sum_cost = sum(actions_cost)
        norm_cost = [0] * (len(actions_cost))
        if sum_cost > 0:
            norm_cost = [c / sum_cost for c in actions_cost]

        driven_info = 0.
        for c in norm_cost:
            if c > 0:
                driven_info -= c * math.log(c, len(norm_cost))
    elif option == 5:
        # 300:800:900:2600
        # read cost是450
        driven_info = 0.
        new_cost_list = deepcopy(actions_cost)
        if p > 0.85:
            new_cost_list[0] *= 1.25
        else:
            new_cost_list[0] *= 0.25
        new_cost_list[1] *= 0.5
        new_cost_list[4] *= 3 * ((2 - p) * new_cost_list[4])

        sum_cost = sum(new_cost_list)
        norm_cost = [0] * (len(new_cost_list))
        if sum_cost > 0:
            norm_cost = [c / sum_cost for c in new_cost_list]

        driven_info = 0.
        for c in norm_cost:
            if c > 0:
                driven_info -= c * math.log(c, len(norm_cost))
    ent = driven_info
    return ent, sum_keys, cost_list


def get_cost_entropy(action_dict_list, time_option, cost_used, is_pred, threshold=0.85, option = 0):
    time_unit = NERMatrics.get_time_estimate_para()[time_option]
    cost_entropy_list = [cal_cost_entropy(action_dict, cost_used, time_unit, is_pred=is_pred,
                                threshold=threshold, option = option)[0]
                                for action_dict in action_dict_list]
    return cost_entropy_list

def get_global_weight(action_dict_list, is_pred, action_costs_estimated0 = None):
    def get_actions(action_dict, is_pred=True, action_dict0=None):
        #     print(time_unit)
        name_map = {'confirm_cnt': 'total_confirm_cnt_preannotated',
                    'revise_cnt': 'total_revise_cnt_preannotated',
                    'delete_cnt': 'total_delete_cnt_preannotated',
                    'add_cnt': 'total_add_cnt_preannotated',
                    'read_cnt': 'total_total_tokens_cnt_plain',
                    'add_plain_cnt': 'total_add_cnt_from_plain',

                    'confirm': 'total_confirm_time_cost_preannotated',
                    'revise': 'total_revise_time_cost_preannotated',
                    'delete': 'total_delete_time_cost_preannotated',
                    'add': 'total_add_time_cost_preannotated',
                    'read': 'total_read_time_cost_plain',
                    'add_plain': 'total_add_time_cost_plain'}

        #     cost_used_map = {"from_preannotated":['confirm', 'revise', 'delete', 'add', 'read'],
        #                 "from_plain":['add_plain', 'read']}

        sum_keys = ['confirm', 'revise', 'delete', 'add', 'read']
        actions_cnt0 = []
        actions_cnt = []
        for key in sum_keys:
            new_key = key + "_cnt"
            if not is_pred:
                new_key = name_map[key + "_cnt"]
            key2 = key
            if key == "add_plain": key2 = "add"
            if action_dict is not None:
                actions_cnt.append(action_dict[new_key])
            if action_dict0 is not None:
                actions_cnt0.append(action_dict0[new_key])

        return [actions_cnt, actions_cnt0]

    results = [get_actions(action_dict, is_pred=is_pred, action_dict0=action_costs_estimated0[ith])
                                for ith, action_dict in enumerate(action_dict_list)]
    # print(results)
    idx = {ith for ith, actions_cnt_ in enumerate(results) if actions_cnt_[0][0] + actions_cnt_[0][1] > 0}

    sum_actions_nrdd = [sum(actions_cnt_[0][:4]) for ith, actions_cnt_ in enumerate(results) if ith in idx]
    sum_actions_all = [sum(actions_cnt_[1][:4]) for ith, actions_cnt_ in enumerate(results) if ith in idx]

    nrdd_weight = 0
    if sum(sum_actions_all) > 0:
        nrdd_weight = sum(sum_actions_nrdd) / sum(sum_actions_all)
    print("nrdd_weight: ", nrdd_weight, "nrdd: ", sum(sum_actions_nrdd), "all: ", sum(sum_actions_all))
    return nrdd_weight


def cal_cost_driven(action_dict, p, time_unit, threshold=0.85, is_pred=True, option=0,
                    action_dict0 = None, redundant_cnt = None, nrdd_weight = None, sim_action_cnt=None,
                    nsamples = None,  outside_invert_ratio = None,
                    norm_costbased = None, norm_performancebased = None,
                    performancebased_flag = None, nrdd_flag = None,
                    norm_nrdd = None, normalized_sim_entity_cnt = None,
                    sim_entity_cnt_flag = None):

    name_map = {'confirm_cnt': 'total_confirm_cnt_preannotated',
                'revise_cnt': 'total_revise_cnt_preannotated',
                'delete_cnt': 'total_delete_cnt_preannotated',
                'add_cnt': 'total_add_cnt_preannotated',
                'read_cnt': 'total_total_tokens_cnt_plain',
                'add_plain_cnt': 'total_add_cnt_from_plain',

                'confirm': 'total_confirm_time_cost_preannotated',
                'revise': 'total_revise_time_cost_preannotated',
                'delete': 'total_delete_time_cost_preannotated',
                'add': 'total_add_time_cost_preannotated',
                'read': 'total_read_time_cost_plain',
                'add_plain': 'total_add_time_cost_plain'}

    sum_keys = ['confirm', 'revise', 'delete', 'add', 'read']

    driven_info = None
    cost_based = None
    nrdd_item = None
    actions_cost0 = []
    actions_cnt0 = []
    actions_cost = []
    actions_cnt = []

    for ith, key in enumerate(sum_keys):
        new_key = key + "_cnt"
        if not is_pred:
            new_key = name_map[key + "_cnt"]

        key2 = key
        if key == "add_plain": key2 = "add"
        if action_dict is not None:
            actions_cost.append(action_dict[new_key] * time_unit[key2])
            actions_cnt.append(action_dict[new_key])
        if action_dict0 is not None:
            actions_cost0.append(action_dict0[new_key] * time_unit[key2])
            actions_cnt0.append(action_dict0[new_key])

    if p > threshold:
        flag = -1
        flag2 = 1
    else:
        flag = 1
        flag2 = -1

    # driven_info = actions_cost[0] * 5 * flag + actions_cost[1] * 5 - actions_cost[2] - actions_cost[3] - actions_cost[
    #     4] * (2 - p)
    # 先观察一个理想的比例使用mdep ccdn
    if option == 0:
        # 只考虑confirmation,前几个iteration还行。后面confirmation花费增加但是，f1并没有什么增加。说明没有带来新的信息了。
        # revise cost太少了。
        # delete cost比random少好事
        # add也基本没有
        driven_info = actions_cost[0] - actions_cost[1] - actions_cost[2] - actions_cost[3] - actions_cost[4]
    # elif option == 1:
    #     # 想要让revise和confirmation的花费大，其余花费小
    #     # 考虑revise cost比confirmation的花费大很多。revise的权重大了。然后revise的增加，read cost也增加蛮多，说明两者有一定的相关性。
    #     # delete和add都很小。
    #     driven_info = actions_cost[0] + actions_cost[1] - actions_cost[2] - actions_cost[3] - actions_cost[4]
    # elif option == 2:
    #     # (2 - p)作用在read cost上，是的read cost前期少，后期增加。也就是前期短句子，后期长句子。
    #     driven_info = actions_cost[0] + actions_cost[1] - actions_cost[2] - actions_cost[3] - actions_cost[4] * (2 - p)
    # elif option == 3:
    #     # 0.85的阈值使得confirmation cost普遍都很小。这个阈值应该设置的大点，前期还是需要confirmation cost大点的。
    #     # revise cost依然很大
    #     driven_info = actions_cost[0] * flag + actions_cost[1] - actions_cost[2] - actions_cost[3] - actions_cost[
    #             4] * (2 - p)
    # elif option == 4:
    #     # confirm增多了些，然后revise减少一点点。read cost增加了一点。
    #     # add的不够多。
    #     driven_info = 2 * actions_cost[0] * flag + actions_cost[1] - actions_cost[2] - actions_cost[3] - actions_cost[
    #             4] * (2 - p)
    # elif option == 5:
    #     # 除以read cost后，确实令read cost少了不少
    #     # confirm cost少了。后面要加大权重
    #     # revise和add变高了
    #     driven_info = (2 * actions_cost[0] * flag + actions_cost[1] - actions_cost[2] - actions_cost[3]) / actions_cost[
    #             4]
    # elif option == 6:
    #     # 效果不明显。
    #     driven_info = (2 * actions_cost[0] * flag + actions_cost[1] - actions_cost[2] - actions_cost[3]) / ((2 - p)*actions_cost[
    #             4])
    # elif option == 7:
    #     driven_info = 2*actions_cost[0] + actions_cost[1] - actions_cost[2] - actions_cost[3] - actions_cost[4]
    # elif option == 8:
    #     # 看confirm的增加
    #     # confirm cost增加了，read cost也增加了，delete增加了，revise减少了，add增加了
    #     driven_info = 4*actions_cost[0] + actions_cost[1] - actions_cost[2] - actions_cost[3] - actions_cost[4]
    # elif option == 9:
    #     # 看confirm的增加
    #     # confirm cost进一步增加了，read cost也进一步增加了，delete进一步增加了，revise进一步减少了，add增加了
    #     driven_info = 8*actions_cost[0] + actions_cost[1] - actions_cost[2] - actions_cost[3] - actions_cost[4]
    # elif option == 10:
    #     # 看revise少量增加
    #     # confirm cost增加了，read cost减少了，delete减少了，revise减少了，add减少了
    #     driven_info = 2*actions_cost[0] + 0.5*actions_cost[1] - actions_cost[2] - actions_cost[3] - actions_cost[4]
    # elif option == 11:
    #     # 看revise少量增加
    #     # confirm cost进一步增加了，read cost进一步减少了，delete进一步减少了，revise进一步减少了，add进一步减少了
    #     driven_info = 2*actions_cost[0] + 0.25*actions_cost[1] - actions_cost[2] - actions_cost[3] - actions_cost[4]
    # elif option == 12:
    #     # 看add少量减少
    #     # confirm减少了，read变化不大，revise变化不大，delete增加了，add增加了
    #     driven_info = 2*actions_cost[0] + 0.25*actions_cost[1] - actions_cost[2] - 0.5*actions_cost[3] - actions_cost[4]
    # elif option == 13:
    #     # 看add少量减少
    #     # confirm变化不大，read变化不大，revise变化不大，delete增加了，add增加了
    #     driven_info = 2*actions_cost[0] + 0.25*actions_cost[1] - actions_cost[2] - 0.25*actions_cost[3] - actions_cost[4]
    # elif option == 14:
    #     # 看add少量增加
    #     # revise减少了，delete增加了，add增加了很多
    #     driven_info = 2*actions_cost[0] + 0.25*actions_cost[1] - actions_cost[2] + 0.25*actions_cost[3] - actions_cost[4]
    # elif option == 15:
    #     # 看add少量增加
    #     # confirm减少了，read增加了一些，revise减少了，delete增加了很多，add增加了很多
    #     driven_info = 2*actions_cost[0] + 0.25*actions_cost[1] - actions_cost[2] + 0.5*actions_cost[3] - actions_cost[4]
    # elif option == 16:
    #     # 看read减少
    #     # 跟14差不多
    #     driven_info = 2*actions_cost[0] + 0.25*actions_cost[1] - actions_cost[2] + 0.25*actions_cost[3] - actions_cost[4]
    # elif option == 17:
    #     # 看read减少
    #     # read减少了，confirm减少了，revise减少了，delete减少了，add减少了一点
    #     driven_info = 2*actions_cost[0] + 0.25*actions_cost[1] - actions_cost[2] + 0.25*actions_cost[3] - 2*actions_cost[4]
    # elif option == 18:
    #     driven_info = 2*actions_cost[0] + 0.25*actions_cost[1] - actions_cost[2] + 0.25*actions_cost[3] - 4*actions_cost[4]
    # elif option == 19:
    #     #
    #     driven_info = actions_cost[0] + actions_cost[1] - actions_cost[2] + 0.2*actions_cost[3] - 3*actions_cost[4]
    # elif option == 20:
    #     # 减弱add的影响 20比21刚开始好一点后面被超越。
    #     # 综合效果比19好些了
    #     # read减少了，confirm小少些后多些。刚开始revise比19少一点，后面比19多一点。revise cost过多导致比random差。
    #     # delete比19少一些。
    #     # add比19少了很多。add过多并未给性能带来很大的提升。
    #     driven_info = actions_cost[0] + actions_cost[1] - actions_cost[2] + 0.1*actions_cost[3] - 3*actions_cost[4]
    # elif option == 21:
    #     # p越大add的影响越小。
    #     # 综合表现比19好。
    #     # read cost减少了。confirm减少了。revise跟19差不多，都比random高很多（这或许是比random差的原因）。delete比19减少了。add比19减少挺多。
    #     # 这里的2或许要换成1.5。
    #     driven_info = actions_cost[0] + actions_cost[1] - actions_cost[2] + (2-p)*0.1*actions_cost[3] - 3*actions_cost[4]
    # elif option == 22:
    #     # p越大add的影响越大
    #     # 综合表现比19差。
    #     # read cost比21增加了，跟19差不多。confirm比21多，跟19差不多。revise都差不多，比random多很多。
    #     # delete先比19和21少，后比他们多。add cost多了很多。这里的1或许要换成。0.5。
    #     driven_info = actions_cost[0] + actions_cost[1] - actions_cost[2] + (1+p)*0.1*actions_cost[3] - 3*actions_cost[4]
    # elif option == 23:
    #     # 减弱revise的影响
    #     # 综合表现比20差不少。cost比例大概是read:confirm:revise:delete:add=190:490:150:5:650.
    #     # random的cost比例大概是650:290:230:110:1380
    #     # read cost少了很多，confirm增加了很多，比random要多，20要多很多。revise跟random差不多。delete几乎为0。add cost跟20差不多，比random少很多。
    #     # performance出奇的差。
    #     # 可以以这个为base，增加实体的多样性。或者增加read和add。
    #     driven_info = actions_cost[0] + 0.25*actions_cost[1] - actions_cost[2] + 0.1*actions_cost[3] - 3*actions_cost[4]
    # elif option == 24:
    #     # 进一步减弱revise的影响
    #     # 综合表现跟23差不多。
    #     # read跟23差不多。confirmation cost比23增多了，比random多。revise cost进一步减少了，几十的样子。delete cost在5左右。add 比23更少了。
    #     # 这里说明光增加confirmation cost是不能提高性能的，还得考虑多样性。
    #     # 这里可以试试隔着一段距离取一个，先试试这个简单点。来增加多样性。或者以confirmation的token不重复来增加多样性。或者把read的减小弱化一点。
    #     driven_info = actions_cost[0] + 0.1*actions_cost[1] - actions_cost[2] + 0.1*actions_cost[3] - 3*actions_cost[4]
    # elif option == 25:
    #     # 加强confirm的影响
    #     # 比24要好一点了。
    #     # read比24多，多差不多100的样子。confirm比24多了300的样子。revise多了一点，多了5左右。
    #     # delete差不多，在0多一点。add减少了，是24的1/3的样子。
    #     driven_info = 1.5*actions_cost[0] + 0.1*actions_cost[1] - actions_cost[2] + 0.1*actions_cost[3] - 3*actions_cost[4]
    # elif option == 26:
    #     # 进一步加强confirm的影响
    #     # 综合效果比25要好。performance在刚开始提升的比较快。
    #     # read比25增多了，从290->500。confirm比25增加了，从800到1200了。revise比25增加了从10左右增加到了60左右。delete增加到了19左右。add没什么增加。
    #     driven_info = 2*actions_cost[0] + 0.1*actions_cost[1] - actions_cost[2] + 0.1*actions_cost[3] - 3*actions_cost[4]
    # elif option == 27:
    #     #这个初始比random好，后来与random有交集。主要是confirm cost太多了。
    #     # 低置信度与confirmation正相关，高置信度负相关。如果不考虑confirmation cost。会比mdepptc好。
    #     # confirmation加了反转。综合效果比26好多了。这里可能是add cost增加了的原因。
    #     # read cost比26减少了，从500->200. confirm cost 从1200降到了400左右。revise cost增加了，从70增加到了160. delete从19增加到了40. add cost从200增加到了600.
    #     driven_info = 2*actions_cost[0]*flag + 0.1*actions_cost[1] - actions_cost[2] + 0.1*actions_cost[3] - 3*actions_cost[4]
    # elif option == 28:
    #     # 与p正相关
    #     # 综合表现，刚开始那一段没有26的好，后面差不多。说明刚开始confirm还是重要的。后期差不多。read cost跟26差不多。confrimation cost也差不多。
    #     # revise cost增加了30左右。delete cost增加了几。add cost增加了几十。
    #     driven_info = (1+p)*actions_cost[0] + 0.1*actions_cost[1] - actions_cost[2] + 0.1*actions_cost[3] - 3*actions_cost[4]
    # elif option == 29:
    #     # 与p负相关
    #     # 刚开始跟26差不多，后来比26差。刚开始比28好，后来比28差。
    #     # read cost比26减少了很多，从500减到150。confirmation cost 从1500减少到550。revise cost从550增加到1000。
    #     # delete cost从18增加到了30。add cost从220增加到了700。
    #     # revise cost和add cost的增加并没有给performance带来提升，反而变差了。有种可能是add的实体比较相近，或者add的实体比较稀少。
    #     driven_info = (2-p)*actions_cost[0] + 0.1*actions_cost[1] - actions_cost[2] + 0.1*actions_cost[3] - 3*actions_cost[4]
    # elif option == 30:
    #     # 减弱read cost的影响，减弱confirmation的影响。
    #     driven_info = 1.5*actions_cost[0]*flag + 0.1*actions_cost[1] - actions_cost[2] + 0.1*actions_cost[3] - 2*actions_cost[4]
    # elif option == 31:
    #     # 测试阈值
    #     # 如果只看performance，阈值越大。但是看cost的话，read和confirm随着阈值增加也在增加。Confirm
    #     # cost超过了random很多，超过一倍的样子。
    #     # 看最后一张图的话，如果confirm取负，add和revise的权重相对会大，cost就会相应变大。但结合revise和add的cost，以及performance的变化来看。前期performance
    #     # 有提高，单不清楚哪个cost带来的影响。后期的话，尽管revise
    #     # cost和add
    #     # cost有所增加，但是整体的performance并未显著增加，说明此时cost选择的实体与performance没什么关系了。
    #     if p > 0.65:
    #         flag = -1
    #     else:
    #         flag = 1
    #     driven_info = 2 * actions_cost[0] * flag + 0.1 * actions_cost[1] - actions_cost[2] + 0.1 * actions_cost[3] - 3 * \
    #                   actions_cost[4]
    # elif option == 32:
    #     # 测试阈值
    #     if p > 0.75:
    #         flag = -1
    #     else:
    #         flag = 1
    #     driven_info = 2 * actions_cost[0] * flag + 0.1 * actions_cost[1] - actions_cost[2] + 0.1 * actions_cost[3] - 3 * \
    #                   actions_cost[4]
    # elif option == 33:
    #     # 测试阈值
    #     if p > 0.95:
    #         flag = -1
    #     else:
    #         flag = 1
    #     driven_info = 2 * actions_cost[0] * flag + 0.1 * actions_cost[1] - actions_cost[2] + 0.1 * actions_cost[3] - 3 * \
    #                   actions_cost[4]
    # elif option == 34:
    #     # confirmation的权重
    #     if p > 0.85:
    #         flag = -1
    #     else:
    #         flag = 1
    #     driven_info = actions_cost[0] * flag + 0.1 * actions_cost[1] - actions_cost[2] + 0.1 * actions_cost[3] - 3 * \
    #                   actions_cost[4]
    # elif option == 35:
    #     # confirmation的权重
    #     if p > 0.85:
    #         flag = -1
    #     else:
    #         flag = 1
    #     driven_info = 1.5 * actions_cost[0] * flag + 0.1 * actions_cost[1] - actions_cost[2] + 0.1 * actions_cost[3] - 3 * \
    #                   actions_cost[4]
    # elif option == 36:
    #     # add的效果增加反转
    #     if p > 0.85:
    #         flag = -1
    #         flag2 = -1
    #     else:
    #         flag = 1
    #         flag2 = 1
    #     driven_info = 2 * actions_cost[0] * flag + 0.1 * actions_cost[1] - actions_cost[2] + 0.1 * actions_cost[3] * flag2 - 3 * \
    #                   actions_cost[4]
    # elif option == 37:
    #     # add的效果增加反转
    #     # 37 = > 置信度高，正相关，置信度低，负相关。也说的过去，置信度高，标注成本会有所下降，这种情况添加新实体感觉会比较划算。
    #     # 从综合结果来看，37比36要好。37的作用下，read，confirm和add的花销都有所下降。
    #     if p > 0.85:
    #         flag = -1
    #         flag2 = 1
    #     else:
    #         flag = 1
    #         flag2 = -1
    #     driven_info = 2 * actions_cost[0] * flag + 0.1 * actions_cost[1] - actions_cost[2] + 0.1 * actions_cost[3] * flag2 - 3 * \
    #                   actions_cost[4]
    # elif option == 38:
    #     # 高置信度减弱对add的需要
    #     driven_info = 2 * actions_cost[0] * flag + 0.1 * actions_cost[1] - actions_cost[2] + 0.1 * actions_cost[3] * (1.85-p) - 3 * \
    #                   actions_cost[4]
    # elif option == 39:
    #     # 高置信度增加对add的需要
    #     # 综合来看38和39并没给performance带来太多改变
    #     driven_info = 2 * actions_cost[0] * flag + 0.1 * actions_cost[1] - actions_cost[2] + 0.1 * actions_cost[3] * (1.95-p) - 3 * \
    #                   actions_cost[4]
    # elif option == 40:
    #     # 在外部加等间隔抽样间隔是10
    #     driven_info = actions_cost[0] + 0.1 * actions_cost[1] - actions_cost[2] + 0.1 * actions_cost[3] - 3 * actions_cost[
    #         4]
    # elif option == 41:
    #     # 在外部加等间隔抽样间隔是20
    #     driven_info = actions_cost[0] + 0.1 * actions_cost[1] - actions_cost[2] + 0.1 * actions_cost[3] - 3 * actions_cost[
    #         4]
    # elif option == 42:
    #     # 27
    #     driven_info = 2 * actions_cost[0] * flag + 0.1 * actions_cost[1] - actions_cost[2] + 0.1 * actions_cost[3] - 3 * \
    #                   actions_cost[4]
    # elif option == 43:
    #     driven_info = 2 * actions_cost[0] * flag + 0.1 * actions_cost[1] - actions_cost[2] + 0.1 * actions_cost[3] - 3 * \
    #                   actions_cost[4]
    # elif option == 44:
    #     driven_info = 2 * actions_cost[0] * flag + 0.1 * actions_cost[1] - actions_cost[2] + 0.1 * actions_cost[3] - 3 * \
    #                   actions_cost[4]
    # elif option == 45:
    #     driven_info = 2 * actions_cost[0] * flag + 0.1 * actions_cost[1] - actions_cost[2] + 0.1 * actions_cost[
    #         3] * flag2 - 3 * actions_cost[4]
    # elif option == 46:
    #     driven_info = 2 * actions_cost[0] * flag + 0.1 * actions_cost[1] - actions_cost[2] + 0.1 * actions_cost[
    #         3] * flag2 - 3 * actions_cost[4]
    # elif option == 47:
    #     driven_info = 2 * actions_cost[0] * flag + 0.1 * actions_cost[1] - actions_cost[2] + 0.1 * actions_cost[
    #         3] * flag2 - 3 * actions_cost[4]
    # elif option == 48:
    #     driven_info = 2 * actions_cost[0] * flag + 0.1 * actions_cost[1] - actions_cost[2] + 0.1 * actions_cost[3] - 3 * \
    #                   actions_cost[4]
    # elif option == 49:
    #     driven_info = 2 * actions_cost[0] * flag + 0.1 * actions_cost[1] - actions_cost[2] + 0.1 * actions_cost[
    #         3] * flag2 - 3 * actions_cost[4]
    # elif option == 50:
    #     # gain cost + min count tag + option 27
    #     driven_info = 2 * actions_cost[0] * flag + 0.1 * actions_cost[1] - actions_cost[2] + 0.1 * actions_cost[3] - 3 * \
    #                   actions_cost[4]
    # elif option == 51:
    #     # 比较37和51，也就是有没考虑tokens和tags type的区别
    #     # gain cost + min count tag + option 37
    #     # 考虑的话，performance确实有所提升，add cost也增加了，其余cost差异不大。f1 score依然无法突破5.5左右的。
    #     # 如果回去看least confidence，mdepptc和random，他们拥有的共同特点是read cost都不小。
    #     # 猜测是read cost首先比较廉价，然后对于分类任务来说，不用标注的tokens也能起到作用对于区分是否需要标注的这个问题上。
    #     # 当然也有另一种可能，长句子可能包含某些entity在短句子是没有的。需要分别去验证这两件事。
    #     # 另一种想法是，短句子，属于时间+人名或者地名的形式，用这样的样本来训练的话，很难捕获到句子结构的特征，可能完整句子也很重要。
    #     driven_info = 2 * actions_cost[0] * flag + 0.1 * actions_cost[1] - actions_cost[2] + 0.1 * actions_cost[
    #         3] * flag2 - 3 * actions_cost[4]
    # elif option == 52:
    #     # gain cost + min count tag + no. of action per preannotated cost
    #     # 这个方法的confirm比mdepptc高，delete，revise，add和read都偏低一点点。
    #     # 如果跑的iteration再多点，不知道会不会突破瓶颈。因为目前看还在0.6的f1，然后还没到performance的拐点。
    #     driven_info = (actions_cnt[0] + actions_cnt[1] + actions_cnt[3]) / sum(actions_cost)
    # elif option == 53:
    #     # gain cost + min count tag + gain cost per preannotated cost
    #     # 综合来看54>56>53>55
    #     # 这几个方法选的样本，read cost比较小。confirm高于random大概200的样子。
    #     # revise的改变并没有给performance带来太大改变，add的改变也是[方法55]。tokens不同，并没有有效的去掉冗余。又或者说样本量太少，因为共同点还是read cost小。
    #     driven_info = (-(actions_cnt[0]*(time_unit["confirm"]-time_unit["add"])) +
    #                    - (actions_cnt[1]*(time_unit["revise"]-time_unit["add"])) - actions_cnt[2] * time_unit["delete"]
    #                    - actions_cnt[3] * time_unit["add"]) / actions_cost[4]
    # elif option == 54:
    #     # gain cost + min count tag + gain cost per preannotated cost
    #     driven_info = (-2*(actions_cnt[0]*(time_unit["confirm"]-time_unit["add"])) +
    #                    - 0.1*(actions_cnt[1]*(time_unit["revise"]-time_unit["add"])) - actions_cnt[2] * time_unit["delete"]
    #                    - 0.1*actions_cnt[3] * time_unit["add"]) / actions_cost[4]
    # elif option == 55:
    #     # gain cost + min count tag + gain cost per preannotated cost
    #     # 对比55,56,57的综合表现，performance的差异不大。add cost增加很多，而performance没有改变太多。
    #     driven_info = (-(actions_cnt[0]*(time_unit["confirm"]-time_unit["add"])) +
    #                    - (actions_cnt[1]*(time_unit["revise"]-time_unit["add"])) - actions_cnt[2] * time_unit["delete"]
    #                    + actions_cnt[3] * time_unit["add"]) / actions_cost[4]
    # elif option == 56:
    #     # gain cost + min count tag + gain cost per preannotated cost
    #     driven_info = (-2*(actions_cnt[0]*(time_unit["confirm"]-time_unit["add"])) +
    #                    - 0.1*(actions_cnt[1]*(time_unit["revise"]-time_unit["add"])) - actions_cnt[2] * time_unit["delete"]
    #                    + 0.1*actions_cnt[3] * time_unit["add"]) / actions_cost[4]
    # elif option == 57:
    #     # gain cost + min count tag + gain cost per preannotated cost 不考虑add
    #     driven_info = (-(actions_cnt[0]*(time_unit["confirm"]-time_unit["add"])) +
    #                    - (actions_cnt[1]*(time_unit["revise"]-time_unit["add"])) - actions_cnt[2] * time_unit["delete"]
    #                    ) / actions_cost[4]
    # elif option == 58:
    #     # gain cost + min count tag + gain cost 27
    #     driven_info = (-2*(actions_cnt[0]*(time_unit["confirm"]-time_unit["add"])) * flag +
    #                    -0.1*(actions_cnt[1]*(time_unit["revise"]-time_unit["add"])) - actions_cnt[2] * time_unit["delete"]
    #                    + 0.1*actions_cnt[3] * time_unit["add"]) - 3*actions_cost[4]
    # elif option == 59:
    #     # gain cost + min count tag + gain cost 37
    #     # 与51对比，58的cost整体上升，performance也突破的了0.6的瓶颈。因为cost大，综合性能不如51.
    #     driven_info = (-2*(actions_cnt[0]*(time_unit["confirm"]-time_unit["add"])) * flag +
    #                    -0.1*(actions_cnt[1]*(time_unit["revise"]-time_unit["add"])) - actions_cnt[2] * time_unit["delete"]
    #                    + 0.1*actions_cnt[3] * time_unit["add"]  * flag2) - 3*actions_cost[4]
    # elif option == 60:
    #     max_v = max([time_unit["confirm"], time_unit["revise"], time_unit["add"], ])
    #     # gain cost + min count tag + gain cost 27
    #     # 没有考虑add cost了，
    #     driven_info = (-(actions_cnt[0]*(time_unit["confirm"]-max_v)) +
    #                    -(actions_cnt[1]*(time_unit["revise"]-max_v)) - actions_cnt[2] * time_unit["delete"]
    #                    - actions_cnt[3] * (time_unit["add"]-max_v)) - 3*actions_cost[4]
    # elif option == 61:
    #     max_v = max([time_unit["confirm"], time_unit["revise"], time_unit["add"], ])
    #     # gain cost + min count tag + 27
    #     driven_info = (-(actions_cnt[0]*(time_unit["confirm"]-max_v)) +
    #                    -(actions_cnt[1]*(time_unit["revise"]-max_v)) - actions_cnt[2] * time_unit["delete"]
    #                    - actions_cnt[3] * (time_unit["add"]-max_v))/actions_cost[4]
    # elif option == 62:
    #     # 减弱read的影响
    #     driven_info = 2*actions_cost[0]*flag + 0.1*actions_cost[1] - actions_cost[2] + 0.1*actions_cost[3] - 2*actions_cost[4]
    # elif option == 63:
    #     # 减弱read的影响
    #     driven_info = 2*actions_cost[0]*flag + 0.1*actions_cost[1] - actions_cost[2] + 0.1*actions_cost[3] - 1*actions_cost[4]
    # elif option == 64:
    #     # 减弱read的影响
    #     driven_info = 2*actions_cost[0]*flag + 0.1*actions_cost[1] - actions_cost[2] + 0.1*actions_cost[3] - 0.5*actions_cost[4]
    # elif option == 65:
    #     # 减弱read的影响
    #     driven_info = 2*actions_cost[0]*flag + 0.1*actions_cost[1] - actions_cost[2] + 0.1*actions_cost[3] - 0.1*actions_cost[4]
    # elif option == 66:
    #     # 增强read的影响
    #     driven_info = 2*actions_cost[0]*flag + 0.1*actions_cost[1] - actions_cost[2] + 0.1*actions_cost[3] + 0.1*actions_cost[4]
    # elif option == 67:
    #     # 增强read的影响
    #     driven_info = 2*actions_cost[0]*flag + 0.1*actions_cost[1] - actions_cost[2] + 0.1*actions_cost[3] + 0.25*actions_cost[4]
    # elif option == 68:
    #     # 增强read的影响
    #     driven_info = 2*actions_cost[0]*flag + 0.1*actions_cost[1] - actions_cost[2] + 0.1*actions_cost[3] + 0.5*actions_cost[4]
    # elif option == 69:
    #     # 增强read的影响
    #     driven_info = 2*actions_cost[0]*flag + 0.1*actions_cost[1] - actions_cost[2] + 0.1*actions_cost[3] + 1*actions_cost[4]
    # elif option == 70:
    #     # 增强read的影响
    #     driven_info = 2*actions_cost[0]*flag + 0.1*actions_cost[1] - actions_cost[2] + 0.1*actions_cost[3] + 2*actions_cost[4]
    # elif option == 71:
    #     # 增强read的影响
    #     driven_info = 2*actions_cost[0]*flag + 0.1*actions_cost[1] - actions_cost[2] + 0.1*actions_cost[3] + 4*actions_cost[4]
    # elif option ==72:
    #     # 希望confirm和revise多，delete少。根据cost大小调权重。不考虑read cost。
    #     driven_info = actions_cnt[0] + 0.2*actions_cnt[1] - actions_cnt[2]
    # elif option ==73:
    #     # 希望confirm和revise多，delete少。根据cost大小调权重。希望read cost越小越好
    #     # read,add和revise都随着read而减少一些，相比于72。performance都差不多。
    #     # 除了add cost，其余cost都比random多。
    #     driven_info = actions_cnt[0] + 0.2*actions_cnt[1] - actions_cnt[2] - 0.1*actions_cost[4]
    # elif option ==74:
    #     # 希望confirm和revise多，delete少。根据cost大小调权重。希望read cost越大越好
    #     # cost普遍有所增加，performance也有所增加。综合表现比72差一点。
    #     driven_info = actions_cnt[0] + 0.2*actions_cnt[1] - actions_cnt[2] + 0.1*actions_cost[4]
    # elif option ==75:
    #     # 希望confirm和revise新实体多旧实体少，delete少。根据cost大小调权重。不考虑read cost。
    #     # read，confirm少一些，add多一些，revise少一些。performance差一些。总体和72差不多。
    #     driven_info = 2*actions_cnt[0] - actions_cnt0[0] + 0.2*(2*actions_cnt[1] - actions_cnt0[1]) - actions_cnt[2]
    # elif option ==76:
    #     # 希望confirm和revise新实体多旧实体少，delete少。根据cost大小调权重。read cost要少。
    #     # cost普遍少一些了，performance也好一点点。综合效果比75好一点点。
    #     driven_info = 2*actions_cnt[0] - actions_cnt0[0] + 0.2*(2*actions_cnt[1] - actions_cnt0[1]) - actions_cnt[2] - 0.1*actions_cost[4]
    # elif option ==77:
    #     # 希望confirm和revise新实体多旧实体少，delete少。根据cost大小调权重。read cost要多。
    #     # cost更多了，而performance差不多。综合效果比75差。
    #     driven_info = 2*actions_cnt[0] - actions_cnt0[0] + 0.2*(2*actions_cnt[1] - actions_cnt0[1]) - actions_cnt[2] + 0.1*actions_cost[4]
    # elif option ==78:
    #     # 希望confirm和revise新实体多旧实体少，delete少。根据cost大小调权重。希望add的少。
    #     # read和confirm差不多，revise相比75变多了。add少了很多。performance好一点。
    #     # 综合比75要好。
    #     driven_info = 2*actions_cnt[0] - actions_cnt0[0] + 0.2*(2*actions_cnt[1] - actions_cnt0[1]) - actions_cnt0[2] - actions_cnt0[3]
    # elif option ==79:
    #     # 希望confirm和revise新实体多旧实体少，delete少。根据cost大小调权重。希望新add多（0.05权重），旧add少。
    #     # 78,79,80,81的add cost逐步增加，performance也逐步增加。但是cost的增加远大于performance的增加，因此综合表现比78要差。
    #     driven_info = 2*actions_cnt[0] - actions_cnt0[0] + 0.2*(2*actions_cnt[1] - actions_cnt0[1]) - actions_cnt0[2] - (actions_cnt0[3]-actions_cnt[3]) + 0.05*actions_cnt[3]
    # elif option ==80:
    #     # 希望confirm和revise新实体多旧实体少，delete少。根据cost大小调权重。希望新add多（0.1权重），旧add少。
    #     driven_info = 2*actions_cnt[0] - actions_cnt0[0] + 0.2*(2*actions_cnt[1] - actions_cnt0[1]) - actions_cnt0[2] - (actions_cnt0[3]-actions_cnt[3]) + 0.1*actions_cnt[3]
    # elif option ==81:
    #     # 希望confirm和revise新实体多旧实体少，delete少。根据cost大小调权重。希望新add多（0.2权重），旧add少。
    #     driven_info = 2*actions_cnt[0] - actions_cnt0[0] + 0.2*(2*actions_cnt[1] - actions_cnt0[1]) - actions_cnt0[2] - (actions_cnt0[3]-actions_cnt[3]) + 0.2*actions_cnt[3]
    # elif option ==82:
    #     # 只考虑去冗余
    #     driven_info = (actions_cnt[0] - actions_cnt0[0]) + (actions_cnt[1] - actions_cnt0[1]) - actions_cnt[2] + (actions_cnt[3] - actions_cnt0[3]) - 0.1 * redundant_cnt
    # elif option ==83:
    #     # 只考虑去冗余
    #     driven_info = (actions_cnt[0] - actions_cnt0[0]) + (actions_cnt[1] - actions_cnt0[1]) - actions_cnt[2] + (actions_cnt[3] - actions_cnt0[3]) - 0.3 * redundant_cnt
    # elif option ==84:
    #     # 只考虑去冗余
    #     driven_info = (actions_cnt[0] - actions_cnt0[0]) + (actions_cnt[1] - actions_cnt0[1]) - actions_cnt[2] + (actions_cnt[3] - actions_cnt0[3]) - 0.3*(2*redundant_cnt-actions_cost[4])
    # elif option ==85:
    #     # 78 - 冗余read
    #     driven_info = 2*actions_cnt[0] - actions_cnt0[0] + 0.2*(2*actions_cnt[1] - actions_cnt0[1]) - actions_cnt0[2] - actions_cnt0[3] - 0.1 * redundant_cnt
    # elif option ==86:
    #     # 78 - 冗余read
    #     driven_info = 2*actions_cnt[0] - actions_cnt0[0] + 0.2*(2*actions_cnt[1] - actions_cnt0[1]) - actions_cnt0[2] - actions_cnt0[3] - 0.3 * redundant_cnt
    # elif option ==87:
    #     # 78 - read
    #     driven_info = 2*actions_cnt[0] - actions_cnt0[0] + 0.2*(2*actions_cnt[1] - actions_cnt0[1]) - actions_cnt0[2] - actions_cnt0[3] - 0.3 * actions_cnt0[4]
    # elif option ==88:
    #     # 78 - read
    #     driven_info = 2*actions_cnt[0] - actions_cnt0[0] + 0.2*(2*actions_cnt[1] - actions_cnt0[1]) - actions_cnt0[2] - actions_cnt0[3] - 0.3*(2*redundant_cnt-actions_cost[4])
    # elif option ==89:
    #     # 52+不考虑tokens和tags
    #     driven_info = (actions_cnt[0] + actions_cnt[1] + actions_cnt[3]) / sum(actions_cost)
    # elif option ==90:
    #     # 78 - read + flag
    #     driven_info = actions_cnt[0]+actions_cnt[0]*flag - actions_cnt0[0] + 0.2*(2*actions_cnt[1] - actions_cnt0[1]) - actions_cnt0[2] - actions_cnt0[3] - 0.3*(2*redundant_cnt-actions_cost[4])
    # elif option ==91:
    #     # 88+不考虑tokens和tags
    #     driven_info = actions_cnt[0] + 0.2*actions_cnt[1] - actions_cnt[2] - actions_cnt[3] - 0.3*actions_cost[4]
    # elif option ==92:
    #     # 78 - read + flag 强化read
    #     driven_info = actions_cnt[0]+actions_cnt[0]*flag - actions_cnt0[0] + 0.2*(2*actions_cnt[1] - actions_cnt0[1]) - actions_cnt0[2] - actions_cnt0[3] - 0.5*(2*redundant_cnt-actions_cost[4])
    # elif option ==93:
    #     # 78 - read + flag 弱化read
    #     driven_info = actions_cnt[0]+actions_cnt[0]*flag - actions_cnt0[0] + 0.2*(2*actions_cnt[1] - actions_cnt0[1]) - actions_cnt0[2] - actions_cnt0[3] - 0.1*(2*redundant_cnt-actions_cost[4])
    # elif option ==94:
    #     # 78 - read 不使用tag_count
    #     driven_info = 2*actions_cnt[0] - actions_cnt0[0] + 0.2*(2*actions_cnt[1] - actions_cnt0[1]) - actions_cnt0[2] - actions_cnt0[3] - 0.3 * redundant_cnt
    # elif option ==95:
    #     # DT篇章级别
    #     driven_info = (2 * actions_cnt[0] - actions_cnt0[0]) - 0.25 * (2*actions_cnt[1] - actions_cnt0[1]) - actions_cnt0[2] - actions_cnt0[3] - 0.021 * redundant_cnt
    # elif option ==96:
    #     # DT句子级别
    #     driven_info = (2 * actions_cnt[0] - actions_cnt0[0]) - 0.25 * (2*actions_cnt[1] - actions_cnt0[1]) - actions_cnt0[2] - actions_cnt0[3] - 0.052 * redundant_cnt
    # elif option ==97:
    #     # 78 - 冗余read + ebd相似度
    #     driven_info = 2*actions_cnt[0] - actions_cnt0[0] + 0.2*(2*actions_cnt[1] - actions_cnt0[1]) - actions_cnt0[2] - actions_cnt0[3] - 0.3 * redundant_cnt
    # elif option ==98:
    #     # 78 - 冗余read 不使用tag_count + ebd相似度
    #     driven_info = 2 * actions_cnt[0] - actions_cnt0[0] + 0.2 * (2 * actions_cnt[1] - actions_cnt0[1]) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - 0.3 * redundant_cnt
    # elif option == 99:
    #     # 下一个实验加个控制，
    #     # min([非冗余add + 非冗余revise] * weight, 冗余tokens)
    #     # weight尝试取0.5或1.5
    #     weight = 0.5
    #     bound = ((actions_cnt0[0] - actions_cnt[0]) + (actions_cnt0[1] - actions_cnt[1])) * weight
    #     driven_info = 2 * actions_cnt[0] - actions_cnt0[0] + 0.2 * (2 * actions_cnt[1] - actions_cnt0[1]) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound ,0.3 * redundant_cnt])
    # elif option ==100:
    #     weight = 1.5
    #     bound = ((actions_cnt0[0] - actions_cnt[0]) + (actions_cnt0[1] - actions_cnt[1])) * weight
    #     driven_info = 2 * actions_cnt[0] - actions_cnt0[0] + 0.2 * (2 * actions_cnt[1] - actions_cnt0[1]) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound ,0.3 * redundant_cnt])
    # elif option ==101:
    #     weight = 6
    #     bound = ((actions_cnt0[0] - actions_cnt[0]) + (actions_cnt0[1] - actions_cnt[1])) * weight
    #     driven_info = 2 * actions_cnt[0] - actions_cnt0[0] + 0.2 * (2 * actions_cnt[1] - actions_cnt0[1]) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound ,0.3 * redundant_cnt])
    # elif option ==102:
    #     weight = 12
    #     bound = ((actions_cnt0[0] - actions_cnt[0]) + (actions_cnt0[1] - actions_cnt[1])) * weight
    #     driven_info = 2 * actions_cnt[0] - actions_cnt0[0] + 0.2 * (2 * actions_cnt[1] - actions_cnt0[1]) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound ,0.3 * redundant_cnt])
    # elif option ==103:
    #     weight = 0.5
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = 2 * actions_cnt[0] - actions_cnt0[0] + 0.25 * (2 * actions_cnt[1] - actions_cnt0[1]) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound ,0.3 * redundant_cnt])
    # elif option ==104:
    #     weight = 6
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = 2 * actions_cnt[0] - actions_cnt0[0] + 0.25 * (2 * actions_cnt[1] - actions_cnt0[1]) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound ,0.3 * redundant_cnt])
    # elif option ==105:
    #     # DT篇章级别
    #     weight = 0.5
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = 2 * actions_cnt[0] - actions_cnt0[0] + 0.25 * (2 * actions_cnt[1] - actions_cnt0[1]) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound, 0.021 * redundant_cnt])
    # elif option ==106:
    #     # DT篇章级别
    #     weight = 6
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = 2 * actions_cnt[0] - actions_cnt0[0] + 0.25 * (2 * actions_cnt[1] - actions_cnt0[1]) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound, 0.021 * redundant_cnt])
    # elif option ==107:
    #     # 一方面改boundt的权重
    #     weight = 12
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = 2 * actions_cnt[0] - actions_cnt0[0] + 0.25 * (2 * actions_cnt[1] - actions_cnt0[1]) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound ,0.3 * redundant_cnt])
    # elif option ==108:
    #     # 一方面改redundant_cnt的权重
    #     weight = 6
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = 2 * actions_cnt[0] - actions_cnt0[0] + 0.25 * (2 * actions_cnt[1] - actions_cnt0[1]) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound ,0.6 * redundant_cnt])
    # elif option ==109:
    #     # 一方面改redundant_cnt的权重
    #     weight = 6
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = 2 * actions_cnt[0] - actions_cnt0[0] + 0.25 * (2 * actions_cnt[1] - actions_cnt0[1]) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound ,0.5 * redundant_cnt])
    # elif option ==110:
    #     # 一方面改redundant_cnt的权重
    #     weight = 6
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = 2 * actions_cnt[0] - actions_cnt0[0] + 0.25 * (2 * actions_cnt[1] - actions_cnt0[1]) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound ,0.4 * redundant_cnt])
    # elif option ==111:
    #     # weight
    #     weight = 12
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = 2 * actions_cnt[0] - actions_cnt0[0] + 0.25 * (2 * actions_cnt[1] - actions_cnt0[1]) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound ,0.4 * redundant_cnt])
    # elif option ==112:
    #     # weight
    #     weight = 3
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = 2 * actions_cnt[0] - actions_cnt0[0] + 0.25 * (2 * actions_cnt[1] - actions_cnt0[1]) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound ,0.4 * redundant_cnt])
    # elif option ==113:
    #     # weight
    #     weight = 1
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = 2 * actions_cnt[0] - actions_cnt0[0] + 0.25 * (2 * actions_cnt[1] - actions_cnt0[1]) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound ,0.4 * redundant_cnt])
    # elif option ==114:
    #     # DT篇章级别
    #     driven_info = (2 * actions_cnt[0] - actions_cnt0[0]) - 0.25 * (2*actions_cnt[1] - actions_cnt0[1]) - actions_cnt0[2] - actions_cnt0[3] - 0.006 * redundant_cnt
    # elif option ==115:
    #     # weight
    #     weight = 6
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = 2 * actions_cnt[0] - actions_cnt0[0] + 0.25 * (2 * actions_cnt[1] - actions_cnt0[1]) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound ,0.8 * redundant_cnt])
    # elif option ==116:
    #     # weight
    #     weight = 6
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = 2 * actions_cnt[0] - actions_cnt0[0] + 0.25 * (2 * actions_cnt[1] - actions_cnt0[1]) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound ,redundant_cnt])
    # elif option ==117:
    #     # weight
    #     weight = 3
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = 2 * actions_cnt[0] - actions_cnt0[0] + 0.25 * (2 * actions_cnt[1] - actions_cnt0[1]) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound ,0.8 * redundant_cnt])
    # elif option ==118:
    #     # weight
    #     weight = 1
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = 2 * actions_cnt[0] - actions_cnt0[0] + 0.25 * (2 * actions_cnt[1] - actions_cnt0[1]) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound ,0.8 * redundant_cnt])
    # elif option ==119:
    #     # weight 只考虑非 redundancy
    #     weight = 3
    #     rdd_w = 0
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = (actions_cnt[0] - rdd_w*(actions_cnt0[0]-actions_cnt[0])) + \
    #                   0.25 * (actions_cnt[1] - rdd_w*(actions_cnt0[1]-actions_cnt[1])) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound, 0.8 * redundant_cnt])
    # elif option ==120:
    #     # weight 只考虑非 redundancy
    #     weight = 3
    #     rdd_w = 0.4
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = (actions_cnt[0] - rdd_w*(actions_cnt0[0]-actions_cnt[0])) + \
    #                   0.25 * (actions_cnt[1] - rdd_w*(actions_cnt0[1]-actions_cnt[1])) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound, 0.8 * redundant_cnt])
    # elif option ==121:
    #     # weight 只考虑非 redundancy
    #     weight = 3
    #     rdd_w = 0.8
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = (actions_cnt[0] - rdd_w*(actions_cnt0[0]-actions_cnt[0])) + \
    #                   0.25 * (actions_cnt[1] - rdd_w*(actions_cnt0[1]-actions_cnt[1])) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound, 0.8 * redundant_cnt])
    # elif option ==122:
    #     # weight 只考虑非 redundancy
    #     weight = 3
    #     rdd_w = 1
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = (actions_cnt[0] - rdd_w*(actions_cnt0[0]-actions_cnt[0])) + \
    #                   0.25 * (actions_cnt[1] - rdd_w*(actions_cnt0[1]-actions_cnt[1])) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound, 0.6 * redundant_cnt])
    # elif option ==123:
    #     # weight 只考虑非 redundancy
    #     weight = 1
    #     rdd_w = 1
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = (actions_cnt[0] - rdd_w*(actions_cnt0[0]-actions_cnt[0])) + \
    #                   0.25 * (actions_cnt[1] - rdd_w*(actions_cnt0[1]-actions_cnt[1])) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound, 0.6 * redundant_cnt])
    # elif option ==124:
    #     # weight 只考虑非 redundancy
    #     weight = 0.5
    #     rdd_w = 1
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = (actions_cnt[0] - rdd_w*(actions_cnt0[0]-actions_cnt[0])) + \
    #                   0.25 * (actions_cnt[1] - rdd_w*(actions_cnt0[1]-actions_cnt[1])) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound, 0.6 * redundant_cnt])
    # elif option ==125:
    #     # weight 只考虑非 redundancy
    #     weight = 2
    #     rdd_w = 1
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = (actions_cnt[0] - rdd_w*(actions_cnt0[0]-actions_cnt[0])) + \
    #                   0.25 * (actions_cnt[1] - rdd_w*(actions_cnt0[1]-actions_cnt[1])) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound, 0.6 * redundant_cnt])
    # elif option ==126:
    #     # weight 只考虑非 redundancy
    #     weight = 1.5
    #     rdd_w = 1
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = (actions_cnt[0] - rdd_w*(actions_cnt0[0]-actions_cnt[0])) + \
    #                   0.25 * (actions_cnt[1] - rdd_w*(actions_cnt0[1]-actions_cnt[1])) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound, 0.6 * redundant_cnt])
    # elif option ==127:
    #     # weight 只考虑非 redundancy
    #     weight = 0.8
    #     rdd_w = 1
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = (actions_cnt[0] - rdd_w*(actions_cnt0[0]-actions_cnt[0])) + \
    #                   0.25 * (actions_cnt[1] - rdd_w*(actions_cnt0[1]-actions_cnt[1])) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound, 0.6 * redundant_cnt])
    # elif option ==128:
    #     # weight 只考虑非 redundancy
    #     weight = 0.6
    #     rdd_w = 1
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = (actions_cnt[0] - rdd_w*(actions_cnt0[0]-actions_cnt[0])) + \
    #                   0.25 * (actions_cnt[1] - rdd_w*(actions_cnt0[1]-actions_cnt[1])) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound, 0.6 * redundant_cnt])
    # elif option ==129:
    #     # weight 只考虑非 redundancy
    #     weight = 0.7
    #     rdd_w = 1
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = (actions_cnt[0] - rdd_w*(actions_cnt0[0]-actions_cnt[0])) + \
    #                   0.25 * (actions_cnt[1] - rdd_w*(actions_cnt0[1]-actions_cnt[1])) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound, 0.6 * redundant_cnt])
    # elif option ==130:
    #     # weight 只考虑非 redundancy
    #     weight = 0.9
    #     rdd_w = 1
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = (actions_cnt[0] - rdd_w*(actions_cnt0[0]-actions_cnt[0])) + \
    #                   0.25 * (actions_cnt[1] - rdd_w*(actions_cnt0[1]-actions_cnt[1])) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound, 0.6 * redundant_cnt])
    # elif option ==131:
    #     # weight 只考虑非 redundancy
    #     weight = 1.5
    #     rdd_w = 1
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = (actions_cnt[0] - rdd_w*(actions_cnt0[0]-actions_cnt[0])) + \
    #                   0.25 * (actions_cnt[1] - rdd_w*(actions_cnt0[1]-actions_cnt[1])) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound, 0.4 * redundant_cnt])
    # elif option ==132:
    #     # weight 只考虑非 redundancy
    #     weight = 1.5
    #     rdd_w = 1
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = (actions_cnt[0] - rdd_w*(actions_cnt0[0]-actions_cnt[0])) + \
    #                   0.25 * (actions_cnt[1] - rdd_w*(actions_cnt0[1]-actions_cnt[1])) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound, 0.2 * redundant_cnt])
    # elif option ==133:
    #     # weight 只考虑非 redundancy
    #     weight = 1.5
    #     rdd_w = 1
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = (actions_cnt[0] - rdd_w*(actions_cnt0[0]-actions_cnt[0])) + \
    #                   0.25 * (actions_cnt[1] - rdd_w*(actions_cnt0[1]-actions_cnt[1])) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound, 0.1 * redundant_cnt])
    # elif option == 134:
    #     weight = 1
    #     if nrdd_weight is not None:
    #         weight = nrdd_weight*1.5
    #     rdd_w = 1
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = (actions_cnt[0] - rdd_w*(actions_cnt0[0]-actions_cnt[0])) + \
    #                   0.25 * (actions_cnt[1] - rdd_w*(actions_cnt0[1]-actions_cnt[1])) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound, 0.6 * redundant_cnt])
    # elif option == 135:
    #     weight = 1
    #     if nrdd_weight is not None:
    #         weight = nrdd_weight*2
    #     rdd_w = 1
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = (actions_cnt[0] - rdd_w*(actions_cnt0[0]-actions_cnt[0])) + \
    #                   0.25 * (actions_cnt[1] - rdd_w*(actions_cnt0[1]-actions_cnt[1])) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound, 0.6 * redundant_cnt])
    # elif option == 136:
    #     weight = 1
    #     if nrdd_weight is not None:
    #         weight = nrdd_weight*2.5
    #     rdd_w = 1
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = (actions_cnt[0] - rdd_w*(actions_cnt0[0]-actions_cnt[0])) + \
    #                   0.25 * (actions_cnt[1] - rdd_w*(actions_cnt0[1]-actions_cnt[1])) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound, 0.6 * redundant_cnt])
    # elif option == 137:
    #     weight = 1
    #     if nrdd_weight is not None:
    #         if nrdd_weight > 0.7:
    #             weight = (nrdd_weight-0.7)*5 + 1
    #         else:
    #             weight = max((nrdd_weight-0.7)*5 + 1, 0)
    #     rdd_w = 1
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = (actions_cnt[0] - rdd_w*(actions_cnt0[0]-actions_cnt[0])) + \
    #                   0.25 * (actions_cnt[1] - rdd_w*(actions_cnt0[1]-actions_cnt[1])) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound, 0.6 * redundant_cnt])
    # elif option == 138:
    #     weight = 1
    #     if nrdd_weight is not None:
    #         if nrdd_weight > 0.7:
    #             weight = (nrdd_weight-0.7)*5 + 1
    #         if nrdd_weight > 0.7:
    #             weight = max((nrdd_weight-0.7)*3 + 1, 0)
    #     rdd_w = 1
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = (actions_cnt[0] - rdd_w*(actions_cnt0[0]-actions_cnt[0])) + \
    #                   0.25 * (actions_cnt[1] - rdd_w*(actions_cnt0[1]-actions_cnt[1])) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound, 0.6 * redundant_cnt])
    # elif option == 139:
    #     weight = 1
    #     if nrdd_weight is not None:
    #         if nrdd_weight > 0.7:
    #             weight = (nrdd_weight-0.7)*5 + 1
    #         if nrdd_weight > 0.7:
    #             weight = max((nrdd_weight-0.7)*1 + 1, 0)
    #     rdd_w = 1
    #     bound = (actions_cnt[0] + actions_cnt[1]) * weight
    #     driven_info = (actions_cnt[0] - rdd_w*(actions_cnt0[0]-actions_cnt[0])) + \
    #                   0.25 * (actions_cnt[1] - rdd_w*(actions_cnt0[1]-actions_cnt[1])) - \
    #                   actions_cnt0[2] - actions_cnt0[3] - min([bound, 0.6 * redundant_cnt])
    # elif option == 140:
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     w = 0
    #     nredundant_actions_cost = deepcopy(actions_cost)
    #     # nredundant_actions_cost[-1] = actions_cost0[-1] - redundant_cost
    #     nredundant_actions_cost[-1] = actions_cost0[-1] - 0
    #     nredundant_actions_cost[3] = actions_cost0[3] - 0
    #     total_cost = sum(actions_cost0)
    #     total_rdd_cost = total_cost - sum(nredundant_actions_cost)
    #     driven_info = total_cost + w * total_rdd_cost
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1
    # elif option == 141:
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     w = 0.25
    #     nredundant_actions_cost = deepcopy(actions_cost)
    #     nredundant_actions_cost[-1] = actions_cost0[-1] - 0
    #     nredundant_actions_cost[3] = actions_cost0[3] - 0
    #     total_cost = sum(actions_cost0)
    #     total_rdd_cost = total_cost - sum(nredundant_actions_cost)
    #     driven_info = total_cost + w * total_rdd_cost
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1
    # elif option == 142:
    #     # delete如何解释呢？
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     w = 1
    #     nredundant_actions_cost = deepcopy(actions_cost)
    #     nredundant_actions_cost[-1] = actions_cost0[-1] - 0
    #     nredundant_actions_cost[3] = actions_cost0[3] - 0
    #     total_cost = sum(actions_cost0)
    #     total_rdd_cost = total_cost - sum(nredundant_actions_cost)
    #     driven_info = total_cost + w * total_rdd_cost
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1
    # elif option == 143:
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     w = 4
    #     nredundant_actions_cost = deepcopy(actions_cost)
    #     nredundant_actions_cost[-1] = actions_cost0[-1] - 0
    #     nredundant_actions_cost[3] = actions_cost0[3] - 0
    #     total_cost = sum(actions_cost0)
    #     total_rdd_cost = total_cost - sum(nredundant_actions_cost)
    #     driven_info = total_cost + w * total_rdd_cost
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1
    # elif option == 144:
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     nredundant_actions_cnt = deepcopy(actions_cnt)
    #     nredundant_actions_cnt[-1] = actions_cnt0[-1] - redundant_cnt
    #     nredundant_actions_cnt[3] = actions_cnt0[3] - 0
    #     driven_info = 0
    #     for i in range(len(actions_cost0)):
    #         if nredundant_actions_cnt[i] > 0:
    #             driven_info += actions_cost0[i] * actions_cnt0[i] / nredundant_actions_cnt[i]
    #         else:
    #             if actions_cnt0[i] > 0:
    #                 driven_info += actions_cnt0[i] * time_unit["add"]
    #             else:
    #                 driven_info += 0
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1
    # elif option == 145:
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     nredundant_actions_cnt = deepcopy(actions_cnt)
    #     # 不考虑read的redundant_cnt
    #     nredundant_actions_cnt[-1] = actions_cnt0[-1] - 0
    #     nredundant_actions_cnt[3] = actions_cnt0[3] - 0
    #     driven_info = 0
    #     for i in range(len(actions_cost0)):
    #         if nredundant_actions_cnt[i] > 0:
    #             driven_info += actions_cost0[i] * actions_cnt0[i] / nredundant_actions_cnt[i]
    #         else:
    #             if actions_cnt0[i] > 0:
    #                 driven_info += actions_cnt0[i] * time_unit["add"]
    #             else:
    #                 driven_info += 0
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1
    # elif option == 146:
    #     efficient_cnt = actions_cnt0[0] + actions_cnt0[1] + actions_cnt0[3]
    #     w = 1
    #     nredundant_actions_cost = deepcopy(actions_cost)
    #     # redundant_cost不考虑
    #     nredundant_actions_cost[-1] = actions_cost0[-1] - redundant_cost
    #     nredundant_actions_cost[3] = actions_cost0[3] - 0
    #     total_cost = sum(actions_cost0)
    #     total_rdd_cost = total_cost - sum(nredundant_actions_cost)
    #     driven_info = total_cost + w * total_rdd_cost
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1
    # elif option == 147:
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     w = 16
    #     nredundant_actions_cost = deepcopy(actions_cost)
    #     nredundant_actions_cost[-1] = actions_cost0[-1] - 0
    #     nredundant_actions_cost[3] = actions_cost0[3] - 0
    #     total_cost = sum(actions_cost0)
    #     total_rdd_cost = total_cost - sum(nredundant_actions_cost)
    #     driven_info = total_cost + w * total_rdd_cost
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1
    # elif option == 148:
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     w = 64
    #     nredundant_actions_cost = deepcopy(actions_cost)
    #     nredundant_actions_cost[-1] = actions_cost0[-1] - 0
    #     nredundant_actions_cost[3] = actions_cost0[3] - 0
    #     total_cost = sum(actions_cost0)
    #     total_rdd_cost = total_cost - sum(nredundant_actions_cost)
    #     driven_info = total_cost + w * total_rdd_cost
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1
    # elif option == 149:
    #     w = 0
    #     efficient_cnt = actions_cnt[0]+actions_cnt[1]+actions_cnt[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     # 不算read和delete
    #     nredundant_actions_cost = deepcopy(actions_cost)
    #     nredundant_actions_cost[2] = 0
    #     nredundant_actions_cost[4] = 0
    #
    #     # 不算read和delete
    #     redundant_actions_cost = [actions_cost0[0] - nredundant_actions_cost[0],
    #                               actions_cost0[1] - nredundant_actions_cost[1],
    #                               0, # delete
    #                               actions_cost0[3] - nredundant_actions_cost[3],
    #                               0 # add
    #                               ]
    #     driven_info = cost_based + w * (sum(redundant_actions_cost)-sum(nredundant_actions_cost))
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 150:
    #     w = 0.25
    #     efficient_cnt = actions_cnt[0]+actions_cnt[1]+actions_cnt[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     # 不算read和delete
    #     nredundant_actions_cost = deepcopy(actions_cost)
    #     nredundant_actions_cost[2] = 0
    #     nredundant_actions_cost[4] = 0
    #
    #     # 不算read和delete
    #     redundant_actions_cost = [actions_cost0[0] - nredundant_actions_cost[0],
    #                               actions_cost0[1] - nredundant_actions_cost[1],
    #                               0, # delete
    #                               actions_cost0[3] - nredundant_actions_cost[3],
    #                               0 # add
    #                               ]
    #     driven_info = cost_based + w * (sum(redundant_actions_cost)-sum(nredundant_actions_cost))
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 151:
    #     w = 1
    #     efficient_cnt = actions_cnt[0]+actions_cnt[1]+actions_cnt[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     # 不算read和delete
    #     nredundant_actions_cost = deepcopy(actions_cost)
    #     nredundant_actions_cost[2] = 0
    #     nredundant_actions_cost[4] = 0
    #
    #     # 不算read和delete
    #     redundant_actions_cost = [actions_cost0[0] - nredundant_actions_cost[0],
    #                               actions_cost0[1] - nredundant_actions_cost[1],
    #                               0, # delete
    #                               actions_cost0[3] - nredundant_actions_cost[3],
    #                               0 # add
    #                               ]
    #     driven_info = cost_based + w * (sum(redundant_actions_cost)-sum(nredundant_actions_cost))
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 152:
    #     w = 4
    #     efficient_cnt = actions_cnt[0]+actions_cnt[1]+actions_cnt[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     # 不算read和delete
    #     nredundant_actions_cost = deepcopy(actions_cost)
    #     nredundant_actions_cost[2] = 0
    #     nredundant_actions_cost[4] = 0
    #
    #     # 不算read和delete
    #     redundant_actions_cost = [actions_cost0[0] - nredundant_actions_cost[0],
    #                               actions_cost0[1] - nredundant_actions_cost[1],
    #                               0, # delete
    #                               actions_cost0[3] - nredundant_actions_cost[3],
    #                               0 # add
    #                               ]
    #     driven_info = cost_based + w * (sum(redundant_actions_cost)-sum(nredundant_actions_cost))
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 153:
    #     w = 16
    #     efficient_cnt = actions_cnt[0]+actions_cnt[1]+actions_cnt[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     # 不算read和delete
    #     nredundant_actions_cost = deepcopy(actions_cost)
    #     nredundant_actions_cost[2] = 0
    #     nredundant_actions_cost[4] = 0
    #
    #     # 不算read和delete
    #     redundant_actions_cost = [actions_cost0[0] - nredundant_actions_cost[0],
    #                               actions_cost0[1] - nredundant_actions_cost[1],
    #                               0, # delete
    #                               actions_cost0[3] - nredundant_actions_cost[3],
    #                               0 # add
    #                               ]
    #     driven_info = cost_based + w * (sum(redundant_actions_cost)-sum(nredundant_actions_cost))
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 154:
    #     # w = 0.01
    #     # efficient_cnt = actions_cnt[0]+actions_cnt[1]+actions_cnt[3]
    #     # total_future_saving_cost = sum(future_saving_cost)
    #     # cur_cost = sum(actions_cost0)
    #     #
    #     # driven_info = cur_cost*cur_cost/(cur_cost+w*total_future_saving_cost)
    #     #
    #     # if efficient_cnt == 0:
    #     #     driven_info = float("inf")
    #     # driven_info *= -1  # 因为外面用的maximize
    #     w1 = 0.25
    #     w2 = 0.01
    #     efficient_cnt = actions_cnt[0]+actions_cnt[1]+actions_cnt[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     # 不算read和delete
    #     nredundant_actions_cost = deepcopy(actions_cost)
    #     nredundant_actions_cost[2] = 0
    #     nredundant_actions_cost[4] = 0
    #
    #     # 不算read和delete
    #     redundant_actions_cost = [actions_cost0[0] - nredundant_actions_cost[0],
    #                               actions_cost0[1] - nredundant_actions_cost[1],
    #                               0, # delete
    #                               actions_cost0[3] - nredundant_actions_cost[3],
    #                               0 # add
    #                               ]
    #     item2 = (sum(redundant_actions_cost)-sum(nredundant_actions_cost))
    #     total_future_saving_cost = sum(future_saving_cost)
    #     driven_info = cost_based + w1 * item2 - w2 * total_future_saving_cost
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 155:
    #     w1 = 0.25
    #     w2 = 0.1
    #     efficient_cnt = actions_cnt[0]+actions_cnt[1]+actions_cnt[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     # 不算read和delete
    #     nredundant_actions_cost = deepcopy(actions_cost)
    #     nredundant_actions_cost[2] = 0
    #     nredundant_actions_cost[4] = 0
    #
    #     # 不算read和delete
    #     redundant_actions_cost = [actions_cost0[0] - nredundant_actions_cost[0],
    #                               actions_cost0[1] - nredundant_actions_cost[1],
    #                               0, # delete
    #                               actions_cost0[3] - nredundant_actions_cost[3],
    #                               0 # add
    #                               ]
    #     item2 = (sum(redundant_actions_cost)-sum(nredundant_actions_cost))
    #     total_future_saving_cost = sum(future_saving_cost)
    #     driven_info = cost_based + w1 * item2 - w2 * total_future_saving_cost
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 156:
    #     w1 = 0.25
    #     w2 = 1
    #     efficient_cnt = actions_cnt[0]+actions_cnt[1]+actions_cnt[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     # 不算read和delete
    #     nredundant_actions_cost = deepcopy(actions_cost)
    #     nredundant_actions_cost[2] = 0
    #     nredundant_actions_cost[4] = 0
    #
    #     # 不算read和delete
    #     redundant_actions_cost = [actions_cost0[0] - nredundant_actions_cost[0],
    #                               actions_cost0[1] - nredundant_actions_cost[1],
    #                               0, # delete
    #                               actions_cost0[3] - nredundant_actions_cost[3],
    #                               0 # add
    #                               ]
    #     item2 = (sum(redundant_actions_cost)-sum(nredundant_actions_cost))
    #     total_future_saving_cost = sum(future_saving_cost)
    #     driven_info = cost_based + w1 * item2 - w2 * total_future_saving_cost
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 157:
    #     w1 = 0.25
    #     w2 = 10
    #     efficient_cnt = actions_cnt[0]+actions_cnt[1]+actions_cnt[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     # 不算read和delete
    #     nredundant_actions_cost = deepcopy(actions_cost)
    #     nredundant_actions_cost[2] = 0
    #     nredundant_actions_cost[4] = 0
    #
    #     # 不算read和delete
    #     redundant_actions_cost = [actions_cost0[0] - nredundant_actions_cost[0],
    #                               actions_cost0[1] - nredundant_actions_cost[1],
    #                               0, # delete
    #                               actions_cost0[3] - nredundant_actions_cost[3],
    #                               0 # add
    #                               ]
    #     item2 = (sum(redundant_actions_cost)-sum(nredundant_actions_cost))
    #     total_future_saving_cost = sum(future_saving_cost)
    #     driven_info = cost_based + w1 * item2 - w2 * total_future_saving_cost
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 158:
    #     w = 0.05
    #     efficient_cnt = actions_cnt[0]+actions_cnt[1]+actions_cnt[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     # 不算read和delete
    #     nredundant_actions_cost = deepcopy(actions_cost)
    #     nredundant_actions_cost[2] = 0
    #     nredundant_actions_cost[4] = 0
    #
    #     # 不算read和delete
    #     redundant_actions_cost = [actions_cost0[0] - nredundant_actions_cost[0],
    #                               actions_cost0[1] - nredundant_actions_cost[1],
    #                               0, # delete
    #                               actions_cost0[3] - nredundant_actions_cost[3],
    #                               0 # add
    #                               ]
    #     driven_info = cost_based + w * (sum(redundant_actions_cost)-sum(nredundant_actions_cost))
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 159:
    #     w = 0.1
    #     efficient_cnt = actions_cnt[0]+actions_cnt[1]+actions_cnt[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     # 不算read和delete
    #     nredundant_actions_cost = deepcopy(actions_cost)
    #     nredundant_actions_cost[2] = 0
    #     nredundant_actions_cost[4] = 0
    #
    #     # 不算read和delete
    #     redundant_actions_cost = [actions_cost0[0] - nredundant_actions_cost[0],
    #                               actions_cost0[1] - nredundant_actions_cost[1],
    #                               0, # delete
    #                               actions_cost0[3] - nredundant_actions_cost[3],
    #                               0 # add
    #                               ]
    #     driven_info = cost_based + w * (sum(redundant_actions_cost)-sum(nredundant_actions_cost))
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 160:
    #     w = 0.15
    #     efficient_cnt = actions_cnt[0]+actions_cnt[1]+actions_cnt[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     # 不算read和delete
    #     nredundant_actions_cost = deepcopy(actions_cost)
    #     nredundant_actions_cost[2] = 0
    #     nredundant_actions_cost[4] = 0
    #
    #     # 不算read和delete
    #     redundant_actions_cost = [actions_cost0[0] - nredundant_actions_cost[0],
    #                               actions_cost0[1] - nredundant_actions_cost[1],
    #                               0, # delete
    #                               actions_cost0[3] - nredundant_actions_cost[3],
    #                               0 # add
    #                               ]
    #     driven_info = cost_based + w * (sum(redundant_actions_cost)-sum(nredundant_actions_cost))
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 161:
    #     w = 0.2
    #     efficient_cnt = actions_cnt[0]+actions_cnt[1]+actions_cnt[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     # 不算read和delete
    #     nredundant_actions_cost = deepcopy(actions_cost)
    #     nredundant_actions_cost[2] = 0
    #     nredundant_actions_cost[4] = 0
    #
    #     # 不算read和delete
    #     redundant_actions_cost = [actions_cost0[0] - nredundant_actions_cost[0],
    #                               actions_cost0[1] - nredundant_actions_cost[1],
    #                               0, # delete
    #                               actions_cost0[3] - nredundant_actions_cost[3],
    #                               0 # add
    #                               ]
    #     driven_info = cost_based + w * (sum(redundant_actions_cost)-sum(nredundant_actions_cost))
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 162:
    #     w = 0.25 # w待定，相似度阈值设置为0.6
    #     efficient_cnt = actions_cnt[0]+actions_cnt[1]+actions_cnt[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     # 不算read和delete
    #     nredundant_actions_cost = deepcopy(actions_cost)
    #     nredundant_actions_cost[2] = 0
    #     nredundant_actions_cost[4] = 0
    #
    #     # 不算read和delete
    #     redundant_actions_cost = [actions_cost0[0] - nredundant_actions_cost[0],
    #                               actions_cost0[1] - nredundant_actions_cost[1],
    #                               0, # delete
    #                               actions_cost0[3] - nredundant_actions_cost[3],
    #                               0 # add
    #                               ]
    #     driven_info = cost_based + w * (sum(redundant_actions_cost)-sum(nredundant_actions_cost))
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 163:
    #     w = 0.25 # w待定，相似度阈值设置为0.7
    #     efficient_cnt = actions_cnt[0]+actions_cnt[1]+actions_cnt[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     # 不算read和delete
    #     nredundant_actions_cost = deepcopy(actions_cost)
    #     nredundant_actions_cost[2] = 0
    #     nredundant_actions_cost[4] = 0
    #
    #     # 不算read和delete
    #     redundant_actions_cost = [actions_cost0[0] - nredundant_actions_cost[0],
    #                               actions_cost0[1] - nredundant_actions_cost[1],
    #                               0, # delete
    #                               actions_cost0[3] - nredundant_actions_cost[3],
    #                               0 # add
    #                               ]
    #     driven_info = cost_based + w * (sum(redundant_actions_cost)-sum(nredundant_actions_cost))
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 164:
    #     w = 0.25 # # w待定，相似度阈值设置为0.9
    #     efficient_cnt = actions_cnt[0]+actions_cnt[1]+actions_cnt[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     # 不算read和delete
    #     nredundant_actions_cost = deepcopy(actions_cost)
    #     nredundant_actions_cost[2] = 0
    #     nredundant_actions_cost[4] = 0
    #
    #     # 不算read和delete
    #     redundant_actions_cost = [actions_cost0[0] - nredundant_actions_cost[0],
    #                               actions_cost0[1] - nredundant_actions_cost[1],
    #                               0, # delete
    #                               actions_cost0[3] - nredundant_actions_cost[3],
    #                               0 # add
    #                               ]
    #     driven_info = cost_based + w * (sum(redundant_actions_cost)-sum(nredundant_actions_cost))
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 165:
    #     w = 2
    #     efficient_cnt = actions_cnt[0]+actions_cnt[1]+actions_cnt[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     # 不算read和delete
    #     nredundant_actions_cost = deepcopy(actions_cost)
    #     nredundant_actions_cost[2] = 0
    #     nredundant_actions_cost[4] = 0
    #
    #     # 不算read和delete
    #     redundant_actions_cost = [actions_cost0[0] - nredundant_actions_cost[0],
    #                               actions_cost0[1] - nredundant_actions_cost[1],
    #                               0, # delete
    #                               actions_cost0[3] - nredundant_actions_cost[3],
    #                               0 # add
    #                               ]
    #     driven_info = cost_based + w * (sum(redundant_actions_cost)-sum(nredundant_actions_cost))
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 166:
    #     w = 3
    #     efficient_cnt = actions_cnt[0]+actions_cnt[1]+actions_cnt[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     # 不算read和delete
    #     nredundant_actions_cost = deepcopy(actions_cost)
    #     nredundant_actions_cost[2] = 0
    #     nredundant_actions_cost[4] = 0
    #
    #     # 不算read和delete
    #     redundant_actions_cost = [actions_cost0[0] - nredundant_actions_cost[0],
    #                               actions_cost0[1] - nredundant_actions_cost[1],
    #                               0, # delete
    #                               actions_cost0[3] - nredundant_actions_cost[3],
    #                               0 # add
    #                               ]
    #     driven_info = cost_based + w * (sum(redundant_actions_cost)-sum(nredundant_actions_cost))
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 167:
    #     w = 0.25
    #     efficient_cnt = actions_cnt[0]+actions_cnt[1]+actions_cnt[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     # 不算read和delete
    #     nredundant_actions_cnt= deepcopy(actions_cnt)
    #     nredundant_actions_cnt[2] = 0
    #     nredundant_actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(nredundant_actions_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 168:
    #     w = 1
    #     efficient_cnt = actions_cnt[0]+actions_cnt[1]+actions_cnt[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     # 不算read和delete
    #     nredundant_actions_cnt= deepcopy(actions_cnt)
    #     nredundant_actions_cnt[2] = 0
    #     nredundant_actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(nredundant_actions_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 169:
    #     w = 4
    #     efficient_cnt = actions_cnt[0]+actions_cnt[1]+actions_cnt[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     # 不算read和delete
    #     nredundant_actions_cnt= deepcopy(actions_cnt)
    #     nredundant_actions_cnt[2] = 0
    #     nredundant_actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(nredundant_actions_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 170:
    #     w = 16
    #     efficient_cnt = actions_cnt[0]+actions_cnt[1]+actions_cnt[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     # 不算read和delete
    #     nredundant_actions_cnt= deepcopy(actions_cnt)
    #     nredundant_actions_cnt[2] = 0
    #     nredundant_actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(nredundant_actions_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 171:
    #     w = 0.25 # 把reading time改成原来的1/3
    #     efficient_cnt = actions_cnt[0]+actions_cnt[1]+actions_cnt[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     # 不算read和delete
    #     nredundant_actions_cost = deepcopy(actions_cost)
    #     nredundant_actions_cost[2] = 0
    #     nredundant_actions_cost[4] = 0
    #
    #     # 不算read和delete
    #     redundant_actions_cost = [actions_cost0[0] - nredundant_actions_cost[0],
    #                               actions_cost0[1] - nredundant_actions_cost[1],
    #                               0, # delete
    #                               actions_cost0[3] - nredundant_actions_cost[3],
    #                               0 # add
    #                               ]
    #     driven_info = cost_based + w * (sum(redundant_actions_cost)-sum(nredundant_actions_cost))
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 172:
    #     w = 0
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 173:
    #     w = 0.25
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 174:
    #     w = 1
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 175:
    #     w = 4
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 176:
    #     w = 16
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 177:
    #     w = 0.01
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 178:
    #     w = 0.1
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 179:
    #     w = 1
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 180:
    #     w = 10
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 181:
    #     w = 1.5
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 182:
    #     w = 2
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 183:
    #     w = 2.5
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 184:
    #     w = 3
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 185:
    #     w = 3.5
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 186:
    #     w = 0.005
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 187:
    #     w = 0.05
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 188:
    #     w = 0.5
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 189:
    #     w = 5
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 190:
    #     w = 50
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 191:
    #     w = 0.3 # threshold 0.8
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 192:
    #     w = 0.3 # threshold 0.6
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 193:
    #     w = 0.3 # threshold 0.7
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 194:
    #     w = 0.3 # threshold 0.9
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 195:
    #     w = 0.3 # 待定 加入random 0.01
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 196:
    #     w = 0.3 # 待定 加入random 0.05
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 197:
    #     w = 0.3 # 待定 加入random  0.1
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 198:
    #     w = 0.3 # 待定 加入random  0.2
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 199:
    #     w = 0.3 # 不加入random, sim = 0.8
    #     efficient_cnt = sim_action_cnt[0]+sim_action_cnt[1]+sim_action_cnt[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 200:
    #     w = 0.3 # 待定 加入random 比例30
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 201:
    #     w = 0.3 # 待定 加入least confidence 比例5
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 202:
    #     w = 0.3 # 待定 加入least confidence 比例20
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 203:
    #     w = 0.3 # 将长度改成0.85
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     actions_cost0 = deepcopy(actions_cost0)
    #     actions_cost0[4] *= 0.85
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 204:
    #     w = 0.3 # 待定 加入mnlp 比例5
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     actions_cost0 = deepcopy(actions_cost0)
    #     actions_cost0[4] *= 0.85
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 205:
    #     w = 0.3 # 待定 加入mnlp 比例20
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     actions_cost0 = deepcopy(actions_cost0)
    #     actions_cost0[4] *= 0.85
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 206:
    #     w = 0.3 # 将长度改成0.3
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     actions_cost0 = deepcopy(actions_cost0)
    #     actions_cost0[4] *= 0.3
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 206:
    #     w = 0.3 # threshold 1
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 207:
    #     # DTK
    #     w = 0.3 # reading tokens 变成0.35 方法待定 混合阈值待定
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3] # 方法待定
    #     actions_cost0 = deepcopy(actions_cost0)
    #     actions_cost0[4] *= 0.35
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    elif option == 208: # 只考虑mincost
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add', 'read']
        driven_info = cost_based
        driven_info *= -1  # 因为外面用的maximize
    # elif option == 209:
    #     w = 0.3 # threshold 0.8
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 210:
    #     # DTK
    #     w = 0.3 # reading tokens 变成0.35 方法待定 混合阈值待定
    #     actions_cost0 = deepcopy(actions_cost0)
    #     actions_cost0[4] *= 0.35
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 211:
    #     w = 0.03 # threshold 0.8
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 212:
    #     w = 3 # threshold 0.8
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 213:
    #     w = 30 # threshold 0.8
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 214:
    #     w = 300 # threshold 0.8
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 215:
    #     w = 0.01 # threshold 0.8
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 216:
    #     w = 0.1 # threshold 0.8
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 217:
    #     w = 1 # threshold 0.8
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 218:
    #     w = 10 # threshold 0.8
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 219:
    #     w = 30 # threshold 0.8
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 220:
    #     w = 50 # threshold 0.8
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 221:
    #     w = 70 # threshold 0.8
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 222:
    #     w = 90 # threshold 0.8
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 223:
    #     w = 3 # threshold 0.8
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 224:
    #     w = 5 # threshold 0.8
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 225:
    #     w = 7 # threshold 0.8
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 226:
    #     w = 9 # threshold 0.8
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 227:
    #     w = 1 # rnd threshold 1
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 228:
    #     w = 1 # rnd threshold 2
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 229:
    #     w = 1 # rnd threshold 3
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 230:
    #     w = 1 # rnd threshold 5%
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 231:
    #     w = 1 # rnd threshold 5% sim theshold 0.6
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 232:
    #     w = 1 # rnd threshold 5% sim theshold 0.7
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 233:
    #     w = 1 # rnd threshold 5% sim theshold 0.9
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 234:
    #     # 试一下反过来排序
    #     w = 1 # threshold 0.8
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     driven_info *= 1  # 因为外面用的minimize
    # elif option == 235:
    #     w = 0.01 # threshold 0.8
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 236:
    #     w = 0.1 # threshold 0.8
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 237:
    #     w = 1 # threshold 0.8
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 238:
    #     w = 10 # threshold 0.8
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 239:
    #     w = 100 # threshold 0.8
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 239:
    #     w = 100 # threshold 0.8
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 240:
    #     w = 2 # threshold 0.8
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 241:
    #     w = 4 # threshold 0.8
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 242:
    #     w = 6 # threshold 0.8
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 243:
    #     w = 8 # threshold 0.8
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 244:
    #     w = 1.5 # threshold 0.8
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 245:
    #     w = 2.5 # threshold 0.8
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 246:
    #     w = 3 # threshold 0.8
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 247:
    #     w = 3.5 # threshold 0.8
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 248:
    #     w = 2 # threshold 0.8 rd 0.1
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 249:
    #     w = 2 # threshold 0.8 rd 0.2
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 250:
    #     w = 2 # threshold 0.8 rd 0.3
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 251:
    #     w = 0.4 # threshold 0.8 rd 0.05
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 252:
    #     w = 0.8 # threshold 0.8 rd 0.05
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 253:
    #     w = 2 # threshold 0.8 rd 0.05
    #     actions_cost0 = deepcopy(actions_cost0)
    #     actions_cost0[4] *= 0.35
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 254:
    #     w = 1.25 # threshold 0.8 rd 0.05
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 255:
    #     w = 1.75 # threshold 0.8 rd 0.05
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 256:
    #     w = 2.25 # threshold 0.8 rd 0.05
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 257:
    #     w = 2.5 # threshold 0.8 rd 0.05
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 258:
    #     w = 2.75 # threshold 0.8 rd 0.05
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 259:
    #     w = 1.5 # threshold 0.8 rd 0.05
    #     efficient_cnt = actions_cnt0[0] + actions_cnt0[1] + actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 260:
    #     w = 1.5 # threshold 0.8 rd 0.05
    #     efficient_cnt = actions_cnt[0] + actions_cnt[1] + actions_cnt[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add', 'read']
    #     actions_cnt[4] = 0
    #     driven_info = cost_based - w * sum(actions_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 261:
    #     w = 0.5 # threshold 0.8 rd 0.05
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 262:
    #     w = 0.5 # threshold 0.8
    #     efficient_cnt = actions_cnt0[0]+actions_cnt0[1]+actions_cnt0[3]
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     if efficient_cnt == 0:
    #         driven_info = float("inf")
    #     driven_info *= -1 # 因为外面用的maximize
    # elif option == 263:
    #     w = 0.25 # threshold 0.8 rd 0.05
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 264:
    #     w = 0.75 # threshold 0.8 rd 0.05
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 265:
    #     w = 0.2 # threshold 0.8 rd 0.05
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    # elif option == 266:
    #     w = 0.3 # threshold 0.8 rd 0.05
    #     cost_based = sum(actions_cost0)
    #     # ['confirm', 'revise', 'delete', 'add']
    #     driven_info = cost_based - w * sum(sim_action_cnt)
    #     driven_info *= -1  # 因为外面用的maximize
    elif option == 267:
        assert nsamples is not None
        w = 0.5*(10**3) # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        driven_info = cost_based - w / nsamples * sum(sim_action_cnt)
        driven_info *= -1  # 因为外面用的maximize
    elif option == 268:
        assert nsamples is not None
        w = 1*(10**3) # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        driven_info = cost_based - w / nsamples * sum(sim_action_cnt)
        driven_info *= -1  # 因为外面用的maximize
    elif option == 269:
        assert nsamples is not None
        w = 2*(10**3) # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        driven_info = cost_based - w / nsamples * sum(sim_action_cnt)
        driven_info *= -1  # 因为外面用的maximize
    elif option == 270:
        assert nsamples is not None
        w = 4*(10**3) # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        driven_info = cost_based - w / nsamples * sum(sim_action_cnt)
        driven_info *= -1  # 因为外面用的maximize
    elif option == 271:
        assert nsamples is not None
        w = 8*(10**3) # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        driven_info = cost_based - w / nsamples * sum(sim_action_cnt)
        driven_info *= -1  # 因为外面用的maximize
    elif option == 272:
        assert nsamples is not None
        w = 5*(10**3) # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        driven_info = cost_based - w / nsamples * sum(sim_action_cnt)
        driven_info *= -1  # 因为外面用的maximize
    ######################
    elif option == 273:
        assert nsamples is not None
        w = 0.5*(10**3) # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        driven_info = cost_based - w / nsamples * sum(sim_action_cnt) * outside_invert_ratio
        driven_info *= -1  # 因为外面用的maximize
    elif option == 274:
        assert nsamples is not None
        w = 1*(10**3) # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        driven_info = cost_based - w / nsamples * sum(sim_action_cnt) * outside_invert_ratio
        driven_info *= -1  # 因为外面用的maximize
    elif option == 275:
        assert nsamples is not None
        w = 2*(10**3) # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        driven_info = cost_based - w / nsamples * sum(sim_action_cnt) * outside_invert_ratio
        driven_info *= -1  # 因为外面用的maximize
    elif option == 276:
        assert nsamples is not None
        w = 4*(10**3) # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        driven_info = cost_based - w / nsamples * sum(sim_action_cnt) * outside_invert_ratio
        driven_info *= -1  # 因为外面用的maximize
    elif option == 277:
        assert nsamples is not None
        w = 8*(10**3) # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        driven_info = cost_based - w / nsamples * sum(sim_action_cnt) * outside_invert_ratio
        driven_info *= -1  # 因为外面用的maximize
    ######################
    elif option == 278:
        # norm_costbased = None, norm_performancebased = None
        assert nsamples is not None
        w = 0.1 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_performancebased is not None:
            driven_info = (1-w)*norm_costbased - w*norm_performancebased
            driven_info *= -1  # 因为外面用的maximize
    elif option == 279:
        assert nsamples is not None
        w = 0.3 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_performancebased is not None:
            driven_info = (1-w)*norm_costbased - w*norm_performancebased
            driven_info *= -1  # 因为外面用的maximize
    elif option == 280:
        assert nsamples is not None
        w = 0.5 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_performancebased is not None:
            driven_info = (1-w)*norm_costbased - w*norm_performancebased
            driven_info *= -1  # 因为外面用的maximize
    elif option == 281:
        assert nsamples is not None
        w = 0.7 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_performancebased is not None:
            driven_info = (1-w)*norm_costbased - w*norm_performancebased
            driven_info *= -1  # 因为外面用的maximize
    elif option == 282:
        assert nsamples is not None
        w = 0.9 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_performancebased is not None:
            driven_info = (1-w)*norm_costbased - w*norm_performancebased
            driven_info *= -1  # 因为外面用的maximize
###########################
    elif option == 283:
        assert nsamples is not None
        w = 0.125*(10**3) # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        driven_info = cost_based - w / nsamples * sum(sim_action_cnt) * outside_invert_ratio
        driven_info *= -1  # 因为外面用的maximize
    elif option == 284:
        assert nsamples is not None
        w = 0.25*(10**3) # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        driven_info = cost_based - w / nsamples * sum(sim_action_cnt) * outside_invert_ratio
        driven_info *= -1  # 因为外面用的maximize
#########################################
    elif option == 285:
        assert nsamples is not None
        w = 0.2  # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_performancebased is not None:
            driven_info = (1 - w) * norm_costbased - w * norm_performancebased
            driven_info *= -1  # 因为外面用的maximize
    elif option == 286:
        assert nsamples is not None
        w = 0.4  # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_performancebased is not None:
            driven_info = (1 - w) * norm_costbased - w * norm_performancebased
            driven_info *= -1  # 因为外面用的maximize
    elif option == 287:
        assert nsamples is not None
        w = 0.6  # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_performancebased is not None:
            driven_info = (1 - w) * norm_costbased - w * norm_performancebased
            driven_info *= -1  # 因为外面用的maximize
    elif option == 288:
        assert nsamples is not None
        w = 0.8  # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_performancebased is not None:
            driven_info = (1 - w) * norm_costbased - w * norm_performancebased
            driven_info *= -1  # 因为外面用的maximize
########################################
    elif option == 289:
        # norm_costbased = None, norm_performancebased = None
        assert nsamples is not None
        w = 0.1 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_performancebased is not None:
            driven_info = (1-w)*norm_costbased - w*norm_performancebased
            if performancebased_flag == 0:
                driven_info = numpy.random.rand()
            elif sum(sim_action_cnt) == 0:
                driven_info = float("inf")
            driven_info *= -1  # 因为外面用的maximize
    elif option == 290:
        assert nsamples is not None
        w = 0.3 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_performancebased is not None:
            driven_info = (1-w)*norm_costbased - w*norm_performancebased
            if performancebased_flag == 0:
                driven_info = numpy.random.rand()
            elif sum(sim_action_cnt) == 0:
                driven_info = float("inf")
            driven_info *= -1  # 因为外面用的maximize
    elif option == 291:
        assert nsamples is not None
        w = 0.5 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_performancebased is not None:
            driven_info = (1-w)*norm_costbased - w*norm_performancebased
            if performancebased_flag == 0:
                driven_info = numpy.random.rand()
            elif sum(sim_action_cnt) == 0:
                driven_info = float("inf")
            driven_info *= -1  # 因为外面用的maximize
    elif option == 292:
        assert nsamples is not None
        w = 0.7 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_performancebased is not None:
            driven_info = (1-w)*norm_costbased - w*norm_performancebased
            if performancebased_flag == 0:
                driven_info = numpy.random.rand()
            elif sum(sim_action_cnt) == 0:
                driven_info = float("inf")
            driven_info *= -1  # 因为外面用的maximize
    elif option == 293:
        assert nsamples is not None
        w = 0.9 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_performancebased is not None:
            driven_info = (1-w)*norm_costbased - w*norm_performancebased
            if performancebased_flag == 0:
                driven_info = numpy.random.rand()
            elif sum(sim_action_cnt) == 0:
                driven_info = float("inf")
            driven_info *= -1  # 因为外面用的maximize
#######################################
    elif option == 294:
        assert nsamples is not None
        w = 5*(10**3) # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        driven_info = cost_based - w / nsamples * sum(sim_action_cnt)
        if performancebased_flag == 0:
            driven_info = numpy.random.rand()
        elif sum(sim_action_cnt) == 0:
            driven_info = float("inf")
        driven_info *= -1  # 因为外面用的maximize
#######################################
    elif option == 295:
        # norm_costbased = None, norm_performancebased = None
        assert nsamples is not None
        w = 0.1 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        actions_cnt[4] = 0
        nrdd_item = sum(actions_cnt)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_nrdd is not None:
            driven_info = (1-w)*norm_costbased - w*norm_nrdd
            if nrdd_flag == 0:
                driven_info = numpy.random.rand()
            elif nrdd_item == 0:
                driven_info = float("inf")
            driven_info *= -1  # 因为外面用的maximize
    elif option == 296:
        # norm_costbased = None, norm_performancebased = None
        assert nsamples is not None
        w = 0.3 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        actions_cnt[4] = 0
        nrdd_item = sum(actions_cnt)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_nrdd is not None:
            driven_info = (1-w)*norm_costbased - w*norm_nrdd
            if nrdd_flag == 0:
                driven_info = numpy.random.rand()
            elif nrdd_item == 0:
                driven_info = float("inf")
            driven_info *= -1  # 因为外面用的maximize
    elif option == 297:
        # norm_costbased = None, norm_performancebased = None
        assert nsamples is not None
        w = 0.5 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        actions_cnt[4] = 0
        nrdd_item = sum(actions_cnt)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_nrdd is not None:
            driven_info = (1-w)*norm_costbased - w*norm_nrdd
            if nrdd_flag == 0:
                driven_info = numpy.random.rand()
            elif nrdd_item == 0:
                driven_info = float("inf")
            driven_info *= -1  # 因为外面用的maximize
    elif option == 298:
        # norm_costbased = None, norm_performancebased = None
        assert nsamples is not None
        w = 0.7 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        actions_cnt[4] = 0
        nrdd_item = sum(actions_cnt)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_nrdd is not None:
            driven_info = (1-w)*norm_costbased - w*norm_nrdd
            if nrdd_flag == 0:
                driven_info = numpy.random.rand()
            elif nrdd_item == 0:
                driven_info = float("inf")
            driven_info *= -1  # 因为外面用的maximize
    elif option == 299:
        # norm_costbased = None, norm_performancebased = None
        assert nsamples is not None
        w = 0.9 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        actions_cnt[4] = 0
        nrdd_item = sum(actions_cnt)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_nrdd is not None:
            driven_info = (1-w)*norm_costbased - w*norm_nrdd
            if nrdd_flag == 0:
                driven_info = numpy.random.rand()
            elif nrdd_item == 0:
                driven_info = float("inf")
            driven_info *= -1  # 因为外面用的maximize
#######################################
    elif option == 300:
        assert nsamples is not None
        w = 0.5*(10**3) # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        driven_info = cost_based - w / nsamples * sum(sim_action_cnt)
        if performancebased_flag == 0:
            driven_info = numpy.random.rand()
        elif sum(sim_action_cnt) == 0:
            driven_info = float("inf")
        driven_info *= -1  # 因为外面用的maximize
    elif option == 301:
        assert nsamples is not None
        w = 1*(10**3) # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        driven_info = cost_based - w / nsamples * sum(sim_action_cnt)
        if performancebased_flag == 0:
            driven_info = numpy.random.rand()
        elif sum(sim_action_cnt) == 0:
            driven_info = float("inf")
        driven_info *= -1  # 因为外面用的maximize
    elif option == 302:
        assert nsamples is not None
        w = 2*(10**3) # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        driven_info = cost_based - w / nsamples * sum(sim_action_cnt)
        if performancebased_flag == 0:
            driven_info = numpy.random.rand()
        elif sum(sim_action_cnt) == 0:
            driven_info = float("inf")
        driven_info *= -1  # 因为外面用的maximize
    elif option == 303:
        assert nsamples is not None
        w = 4*(10**3) # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        driven_info = cost_based - w / nsamples * sum(sim_action_cnt)
        if performancebased_flag == 0:
            driven_info = numpy.random.rand()
        elif sum(sim_action_cnt) == 0:
            driven_info = float("inf")
        driven_info *= -1  # 因为外面用的maximize
    elif option == 304:
        assert nsamples is not None
        w = 8*(10**3) # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        driven_info = cost_based - w / nsamples * sum(sim_action_cnt)
        if performancebased_flag == 0:
            driven_info = numpy.random.rand()
        elif sum(sim_action_cnt) == 0:
            driven_info = float("inf")
        driven_info *= -1  # 因为外面用的maximize
#######################################
    elif option == 305:
        assert nsamples is not None
        w = 0.0015625*(10**3) # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        driven_info = cost_based - w / nsamples * sum(sim_action_cnt)
        if performancebased_flag == 0:
            driven_info = numpy.random.rand()
        elif sum(sim_action_cnt) == 0:
            driven_info = float("inf")
        driven_info *= -1  # 因为外面用的maximize
    elif option == 306:
        assert nsamples is not None
        w = 0.003125*(10**3) # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        driven_info = cost_based - w / nsamples * sum(sim_action_cnt)
        if performancebased_flag == 0:
            driven_info = numpy.random.rand()
        elif sum(sim_action_cnt) == 0:
            driven_info = float("inf")
        driven_info *= -1  # 因为外面用的maximize
    elif option == 307:
        assert nsamples is not None #跑一下
        w = 0.00625*(10**3) # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        driven_info = cost_based - w / nsamples * sum(sim_action_cnt)
        if performancebased_flag == 0:
            driven_info = numpy.random.rand()
        elif sum(sim_action_cnt) == 0:
            driven_info = float("inf")
        driven_info *= -1  # 因为外面用的maximize
    elif option == 308:
        assert nsamples is not None
        w = 0.0125*(10**3) # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        driven_info = cost_based - w / nsamples * sum(sim_action_cnt)
        if performancebased_flag == 0:
            driven_info = numpy.random.rand()
        elif sum(sim_action_cnt) == 0:
            driven_info = float("inf")
        driven_info *= -1  # 因为外面用的maximize
    elif option == 309:
        assert nsamples is not None
        w = 0.025*(10**3) # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        driven_info = cost_based - w / nsamples * sum(sim_action_cnt)
        if performancebased_flag == 0:
            driven_info = numpy.random.rand()
        elif sum(sim_action_cnt) == 0:
            driven_info = float("inf")
        driven_info *= -1  # 因为外面用的maximize
#########################################
    elif option == 310:
        assert nsamples is not None
        w = 1*(10**3) # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        driven_info = -sum(sim_action_cnt)
        if performancebased_flag == 0:
            driven_info = numpy.random.rand()
        elif sum(sim_action_cnt) == 0:
            driven_info = float("inf")
        driven_info *= -1  # 因为外面用的maximize
    elif option == 311:
        assert nsamples is not None
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        driven_info = cost_based
        if performancebased_flag == 0:
            driven_info = numpy.random.rand()
        elif sum(sim_action_cnt) == 0:
            driven_info = float("inf")
        driven_info *= -1  # 因为外面用的maximize
    elif option == 312:
        assert nsamples is not None
        w = 1 * (10 ** 3)  # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        driven_info = cost_based - w / nsamples * sum(sim_action_cnt)
        if performancebased_flag == 0:
            driven_info = numpy.random.rand()
        driven_info *= -1  # 因为外面用的maximize
    elif option == 313:
        assert nsamples is not None
        w = 1 * (10 ** 3)  # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        driven_info = cost_based - w / nsamples * sum(sim_action_cnt)
        if sum(sim_action_cnt) == 0:
            driven_info = float("inf")
        driven_info *= -1  # 因为外面用的maximize
##########################################
    elif option == 314:
        assert nsamples is not None
        w = 1*(10**3) # threshold 0.7 查看实体数下降速度
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        driven_info = cost_based - w / nsamples * sum(sim_action_cnt)
        if performancebased_flag == 0:
            driven_info = numpy.random.rand()
        elif sum(sim_action_cnt) == 0:
            driven_info = float("inf")
        driven_info *= -1  # 因为外面用的maximize
    elif option == 315:
        assert nsamples is not None
        w = 1*(10**3) # threshold 0.7 查看实体数下降速度
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        driven_info = cost_based - w / nsamples * sum(sim_action_cnt)
        if performancebased_flag == 0:
            driven_info = numpy.random.rand()
        elif sum(sim_action_cnt) == 0:
            driven_info = float("inf")
        driven_info *= -1  # 因为外面用的maximize
##########################################
    elif option == 316:
        # norm_costbased = None, norm_performancebased = None
        assert nsamples is not None
        w = 0.1 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_performancebased is not None:
            driven_info = (1-w)*norm_costbased - w*norm_performancebased
            driven_info *= -1  # 因为外面用的maximize
            if performancebased_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 317:
        assert nsamples is not None
        w = 0.2 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_performancebased is not None:
            driven_info = (1-w)*norm_costbased - w*norm_performancebased
            driven_info *= -1  # 因为外面用的maximize
            if performancebased_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 318:
        assert nsamples is not None
        w = 0.3 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_performancebased is not None:
            driven_info = (1-w)*norm_costbased - w*norm_performancebased
            driven_info *= -1  # 因为外面用的maximize
            if performancebased_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 319:
        assert nsamples is not None
        w = 0.4 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_performancebased is not None:
            driven_info = (1-w)*norm_costbased - w*norm_performancebased
            driven_info *= -1  # 因为外面用的maximize
            if performancebased_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 320:
        assert nsamples is not None
        w = 0.5 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_performancebased is not None:
            driven_info = (1-w)*norm_costbased - w*norm_performancebased
            driven_info *= -1  # 因为外面用的maximize
            if performancebased_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 321:
        assert nsamples is not None
        w = 0.6 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_performancebased is not None:
            driven_info = (1-w)*norm_costbased - w*norm_performancebased
            driven_info *= -1  # 因为外面用的maximize
            if performancebased_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 322:
        assert nsamples is not None
        w = 0.7 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_performancebased is not None:
            driven_info = (1-w)*norm_costbased - w*norm_performancebased
            driven_info *= -1  # 因为外面用的maximize
            if performancebased_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 323:
        assert nsamples is not None
        w = 0.8 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_performancebased is not None:
            driven_info = (1-w)*norm_costbased - w*norm_performancebased
            driven_info *= -1  # 因为外面用的maximize
            if performancebased_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 324:
        assert nsamples is not None
        w = 0.9 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_performancebased is not None:
            driven_info = (1-w)*norm_costbased - w*norm_performancebased
            driven_info *= -1  # 因为外面用的maximize
            if performancebased_flag == 0:
                driven_info = numpy.random.rand()
##########################################
    elif option == 325:
        # norm_costbased = None, norm_performancebased = None
        assert nsamples is not None
        w = 0.025 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_performancebased is not None:
            driven_info = (1-w)*norm_costbased - w*norm_performancebased
            driven_info *= -1  # 因为外面用的maximize
            if performancebased_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 326:
        assert nsamples is not None
        w = 0.05 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_performancebased is not None:
            driven_info = (1-w)*norm_costbased - w*norm_performancebased
            driven_info *= -1  # 因为外面用的maximize
            if performancebased_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 326:
        assert nsamples is not None
        w = 0.05 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_performancebased is not None:
            driven_info = (1-w)*norm_costbased - w*norm_performancebased
            driven_info *= -1  # 因为外面用的maximize
            if performancebased_flag == 0:
                driven_info = numpy.random.rand()
##########################################
    elif option == 327:
        assert nsamples is not None
        w = 0.1 # threshold 0.8 rd 0.05
        actions_cost0 = deepcopy(actions_cost0)
        actions_cost0[4] *= 0.35
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_performancebased is not None:
            driven_info = (1-w)*norm_costbased - w*norm_performancebased
            driven_info *= -1  # 因为外面用的maximize
            if performancebased_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 328:
        assert nsamples is not None
        w = 0.2 # threshold 0.8 rd 0.05
        actions_cost0 = deepcopy(actions_cost0)
        actions_cost0[4] *= 0.35
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_performancebased is not None:
            driven_info = (1-w)*norm_costbased - w*norm_performancebased
            driven_info *= -1  # 因为外面用的maximize
            if performancebased_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 329:
        assert nsamples is not None
        w = 0.3 # threshold 0.8 rd 0.05
        actions_cost0 = deepcopy(actions_cost0)
        actions_cost0[4] *= 0.35
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_performancebased is not None:
            driven_info = (1-w)*norm_costbased - w*norm_performancebased
            driven_info *= -1  # 因为外面用的maximize
            if performancebased_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 330:
        assert nsamples is not None
        w = 0.4 # threshold 0.8 rd 0.05
        actions_cost0 = deepcopy(actions_cost0)
        actions_cost0[4] *= 0.35
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_performancebased is not None:
            driven_info = (1-w)*norm_costbased - w*norm_performancebased
            driven_info *= -1  # 因为外面用的maximize
            if performancebased_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 331:
        assert nsamples is not None
        w = 0.5 # threshold 0.8 rd 0.05
        actions_cost0 = deepcopy(actions_cost0)
        actions_cost0[4] *= 0.35
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_performancebased is not None:
            driven_info = (1-w)*norm_costbased - w*norm_performancebased
            driven_info *= -1  # 因为外面用的maximize
            if performancebased_flag == 0:
                driven_info = numpy.random.rand()
##########################################
    elif option == 332:
        # norm_costbased = None, norm_performancebased = None
        assert nsamples is not None
        w = 0.05 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_performancebased is not None:
            driven_info = (1-w)*norm_costbased - w*norm_performancebased
            driven_info *= -1  # 因为外面用的maximize
            if performancebased_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 333:
        # norm_costbased = None, norm_performancebased = None
        assert nsamples is not None
        w = 0 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if norm_performancebased is not None:
            driven_info = (1-w)*norm_costbased - w*norm_performancebased
            driven_info *= -1  # 因为外面用的maximize
            if performancebased_flag == 0:
                driven_info = numpy.random.rand()
##########################################
    elif option == 334:
        # norm_costbased = None, norm_performancebased = None
        assert nsamples is not None
        w = 0 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if normalized_sim_entity_cnt is not None:
            driven_info = (1-w)*norm_costbased - w*normalized_sim_entity_cnt
            driven_info *= -1  # 因为外面用的maximize
            if sim_entity_cnt_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 335:
        assert nsamples is not None
        w = 0.1 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if normalized_sim_entity_cnt is not None:
            driven_info = (1-w)*norm_costbased - w*normalized_sim_entity_cnt
            driven_info *= -1  # 因为外面用的maximize
            if sim_entity_cnt_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 336:
        assert nsamples is not None
        w = 0.2 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if normalized_sim_entity_cnt is not None:
            driven_info = (1-w)*norm_costbased - w*normalized_sim_entity_cnt
            driven_info *= -1  # 因为外面用的maximize
            if sim_entity_cnt_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 337:
        assert nsamples is not None
        w = 0.3 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if normalized_sim_entity_cnt is not None:
            driven_info = (1-w)*norm_costbased - w*normalized_sim_entity_cnt
            driven_info *= -1  # 因为外面用的maximize
            if sim_entity_cnt_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 338:
        assert nsamples is not None
        w = 0.4 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if normalized_sim_entity_cnt is not None:
            driven_info = (1-w)*norm_costbased - w*normalized_sim_entity_cnt
            driven_info *= -1  # 因为外面用的maximize
            if sim_entity_cnt_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 339:
        assert nsamples is not None
        w = 0.5 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if normalized_sim_entity_cnt is not None:
            driven_info = (1-w)*norm_costbased - w*normalized_sim_entity_cnt
            driven_info *= -1  # 因为外面用的maximize
            if sim_entity_cnt_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 340:
        assert nsamples is not None
        w = 0.6 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if normalized_sim_entity_cnt is not None:
            driven_info = (1-w)*norm_costbased - w*normalized_sim_entity_cnt
            driven_info *= -1  # 因为外面用的maximize
            if sim_entity_cnt_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 341:
        assert nsamples is not None
        w = 0.7 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if normalized_sim_entity_cnt is not None:
            driven_info = (1-w)*norm_costbased - w*normalized_sim_entity_cnt
            driven_info *= -1  # 因为外面用的maximize
            if sim_entity_cnt_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 342:
        assert nsamples is not None
        w = 0.8 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if normalized_sim_entity_cnt is not None:
            driven_info = (1-w)*norm_costbased - w*normalized_sim_entity_cnt
            driven_info *= -1  # 因为外面用的maximize
            if sim_entity_cnt_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 343:
        assert nsamples is not None
        w = 0.9 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if normalized_sim_entity_cnt is not None:
            driven_info = (1-w)*norm_costbased - w*normalized_sim_entity_cnt
            driven_info *= -1  # 因为外面用的maximize
            if sim_entity_cnt_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 344:
        assert nsamples is not None
        w = 1 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if normalized_sim_entity_cnt is not None:
            driven_info = (1-w)*norm_costbased - w*normalized_sim_entity_cnt
            driven_info *= -1  # 因为外面用的maximize
            if sim_entity_cnt_flag == 0:
                driven_info = numpy.random.rand()
##########################################
    elif option == 345:
        assert nsamples is not None
        w = 0.3 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if normalized_sim_entity_cnt is not None:
            driven_info = (1-w)*norm_costbased - w*normalized_sim_entity_cnt
            driven_info *= -1  # 因为外面用的maximize
            if sim_entity_cnt_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 346:
        assert nsamples is not None
        w = 0.4 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if normalized_sim_entity_cnt is not None:
            driven_info = (1-w)*norm_costbased - w*normalized_sim_entity_cnt
            driven_info *= -1  # 因为外面用的maximize
            if sim_entity_cnt_flag == 0:
                driven_info = numpy.random.rand()
    ##########################################
    elif option == 347:
        assert nsamples is not None
        w = 0.15  # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if normalized_sim_entity_cnt is not None:
            driven_info = (1 - w) * norm_costbased - w * normalized_sim_entity_cnt
            driven_info *= -1  # 因为外面用的maximize
            if sim_entity_cnt_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 348:
        assert nsamples is not None
        w = 0.25  # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if normalized_sim_entity_cnt is not None:
            driven_info = (1 - w) * norm_costbased - w * normalized_sim_entity_cnt
            driven_info *= -1  # 因为外面用的maximize
            if sim_entity_cnt_flag == 0:
                driven_info = numpy.random.rand()
    ##########################################
    elif option == 349:
        # norm_costbased = None, norm_performancebased = None
        assert nsamples is not None
        w = 0 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if normalized_sim_entity_cnt is not None:
            driven_info = (1-w)*norm_costbased - w*normalized_sim_entity_cnt
            driven_info *= -1  # 因为外面用的maximize
            if sim_entity_cnt_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 350:
        assert nsamples is not None
        w = 0.1 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if normalized_sim_entity_cnt is not None:
            driven_info = (1-w)*norm_costbased - w*normalized_sim_entity_cnt
            driven_info *= -1  # 因为外面用的maximize
            if sim_entity_cnt_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 351:
        assert nsamples is not None
        w = 0.2 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if normalized_sim_entity_cnt is not None:
            driven_info = (1-w)*norm_costbased - w*normalized_sim_entity_cnt
            driven_info *= -1  # 因为外面用的maximize
            if sim_entity_cnt_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 352:
        assert nsamples is not None
        w = 0.3 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if normalized_sim_entity_cnt is not None:
            driven_info = (1-w)*norm_costbased - w*normalized_sim_entity_cnt
            driven_info *= -1  # 因为外面用的maximize
            if sim_entity_cnt_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 353:
        assert nsamples is not None
        w = 0.4 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if normalized_sim_entity_cnt is not None:
            driven_info = (1-w)*norm_costbased - w*normalized_sim_entity_cnt
            driven_info *= -1  # 因为外面用的maximize
            if sim_entity_cnt_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 354:
        assert nsamples is not None
        w = 0.5 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if normalized_sim_entity_cnt is not None:
            driven_info = (1-w)*norm_costbased - w*normalized_sim_entity_cnt
            driven_info *= -1  # 因为外面用的maximize
            if sim_entity_cnt_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 355:
        assert nsamples is not None
        w = 0.6 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if normalized_sim_entity_cnt is not None:
            driven_info = (1-w)*norm_costbased - w*normalized_sim_entity_cnt
            driven_info *= -1  # 因为外面用的maximize
            if sim_entity_cnt_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 356:
        assert nsamples is not None
        w = 0.7 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if normalized_sim_entity_cnt is not None:
            driven_info = (1-w)*norm_costbased - w*normalized_sim_entity_cnt
            driven_info *= -1  # 因为外面用的maximize
            if sim_entity_cnt_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 357:
        assert nsamples is not None
        w = 0.8 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if normalized_sim_entity_cnt is not None:
            driven_info = (1-w)*norm_costbased - w*normalized_sim_entity_cnt
            driven_info *= -1  # 因为外面用的maximize
            if sim_entity_cnt_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 358:
        assert nsamples is not None
        w = 0.9 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if normalized_sim_entity_cnt is not None:
            driven_info = (1-w)*norm_costbased - w*normalized_sim_entity_cnt
            driven_info *= -1  # 因为外面用的maximize
            if sim_entity_cnt_flag == 0:
                driven_info = numpy.random.rand()
    elif option == 359:
        assert nsamples is not None
        w = 1 # threshold 0.8 rd 0.05
        cost_based = sum(actions_cost0)
        # ['confirm', 'revise', 'delete', 'add']
        if normalized_sim_entity_cnt is not None:
            driven_info = (1-w)*norm_costbased - w*normalized_sim_entity_cnt
            driven_info *= -1  # 因为外面用的maximize
            if sim_entity_cnt_flag == 0:
                driven_info = numpy.random.rand()
    else:
        raise NotImplementedError()

    if sim_action_cnt is not None:
        actions_cnt.extend(sim_action_cnt)
        actions_cnt0.extend(sim_action_cnt)

    return actions_cost0, [actions_cnt, actions_cnt0], sum_keys, nrdd_item, cost_based, driven_info

def normalize(alist):
    max_v = max(alist)
    min_v = min(alist)
    flag = 1
    if max_v == 0:
        flag = 0

    if max_v == min_v:
        return [0 for x in alist], flag

    intervals = max_v - min_v
    alist = [(v-min_v)/intervals for v in alist]
    return alist, flag


def get_cost_driven(action_dict_list, p_list, time_option, is_pred, threshold=0.85, option=0,
                    action_costs_estimated0 = None,
                    redundant_tokens_cnt = None,
                    sim_action_cnt = None,
                    nsamples = None,
                    outside_invert_ratio=None,
                    sim_entity_cnt = None):

    nrdd_weight =None
    if action_costs_estimated0 is not None and action_dict_list is not None:
        nrdd_weight = get_global_weight(action_dict_list, is_pred, action_costs_estimated0=action_costs_estimated0)

    n = None
    if action_costs_estimated0 is not None:
        n = len(action_costs_estimated0)

    if action_dict_list is not None:
        n = len(action_dict_list)

    if action_dict_list is None:
        action_dict_list = [None] * n

    if action_costs_estimated0 is None:
        action_costs_estimated0 = [None] * n

    if redundant_tokens_cnt is None:
        redundant_tokens_cnt = [None] * n

    if sim_action_cnt is None:
        sim_action_cnt = [None] * n

    time_unit = NERMatrics.get_time_estimate_para()[time_option]

    results_ = [cal_cost_driven(action_dict_list[ith], p_list[ith], time_unit, threshold=threshold,
                        is_pred=is_pred, option=option, action_dict0 = action_costs_estimated0[ith],
                                   redundant_cnt = redundant_tokens_cnt[ith], nrdd_weight=nrdd_weight,
                               sim_action_cnt = sim_action_cnt[ith],
                               nsamples = nsamples,
                               outside_invert_ratio=outside_invert_ratio)
                   for ith in range(n)]

    cost_list = [results_[ith][-2] for ith, action_dict in enumerate(action_dict_list)]
    if cost_list is not None and cost_list[0] is not None:
        normalized_costs, _ = normalize(cost_list)
    else:
        normalized_costs = [None for ith, action_dict in enumerate(action_dict_list)]

    nrdd_list = [results_[ith][-3] for ith, action_dict in enumerate(action_dict_list)]
    if nrdd_list is None or nrdd_list[0] is None:
        normalized_nrdd, nrdd_flag = [None] * n, None
    else:
        normalized_nrdd, nrdd_flag = normalize(nrdd_list)

    if sim_action_cnt is None or sim_action_cnt[0] is None:
        normalized_nentities = [None] * n
        performancebased_flag = None
    else:
        nentities = [sum(cnt_list) for cnt_list in sim_action_cnt]
        normalized_nentities, performancebased_flag = normalize(nentities)

    if sim_entity_cnt is None:
        normalized_sim_entity_cnt = [None] * n
        sim_entity_cnt_flag = None
    else:
        normalized_sim_entity_cnt, sim_entity_cnt_flag = normalize(sim_entity_cnt)


    results = [cal_cost_driven(action_dict_list[ith], p_list[ith], time_unit, threshold=threshold,
                        is_pred=is_pred, option=option, action_dict0 = action_costs_estimated0[ith],
                                   redundant_cnt = redundant_tokens_cnt[ith], nrdd_weight=nrdd_weight,
                               sim_action_cnt = sim_action_cnt[ith],
                               nsamples = nsamples,
                               outside_invert_ratio=outside_invert_ratio,
                               norm_costbased = normalized_costs[ith],
                               norm_performancebased = normalized_nentities[ith],
                               performancebased_flag = performancebased_flag,
                               nrdd_flag = nrdd_flag,
                               norm_nrdd = normalized_nrdd[ith],
                               normalized_sim_entity_cnt = normalized_sim_entity_cnt[ith],
                               sim_entity_cnt_flag = sim_entity_cnt_flag
                               )
                   for ith in range(n)]

    action_cnts = [results[ith][1] for ith, action_dict in enumerate(action_dict_list)]
    driven_info = [results[ith][-1] for ith, action_dict in enumerate(action_dict_list)]
    return driven_info, action_cnts

def cal_entity_driven(action_dict, p, time_unit, threshold=0.85, is_pred=True, option=0):

    name_map = {'confirm_cnt': 'total_confirm_cnt_preannotated',
                'revise_cnt': 'total_revise_cnt_preannotated',
                'delete_cnt': 'total_delete_cnt_preannotated',
                'add_cnt': 'total_add_cnt_preannotated',
                'read_cnt': 'total_total_tokens_cnt_plain',
                'add_plain_cnt': 'total_add_cnt_from_plain',

                'confirm': 'total_confirm_time_cost_preannotated',
                'revise': 'total_revise_time_cost_preannotated',
                'delete': 'total_delete_time_cost_preannotated',
                'add': 'total_add_time_cost_preannotated',
                'read': 'total_read_time_cost_plain',
                'add_plain': 'total_add_time_cost_plain'}

    #     cost_used_map = {"from_preannotated":['confirm', 'revise', 'delete', 'add', 'read'],
    #                 "from_plain":['add_plain', 'read']}

    sum_keys = ['confirm', 'revise', 'delete', 'add', 'read']

    actions_cost = []
    actions_cnt = []
    for key in sum_keys:
        new_key = key + "_cnt"
        if not is_pred:
            new_key = name_map[key + "_cnt"]

        key2 = key
        if key == "add_plain": key2 = "add"
        actions_cost.append(action_dict[new_key] * time_unit[key2])
        actions_cnt.append(action_dict[new_key])

    driven_info = (actions_cnt[0] + actions_cnt[1] + actions_cnt[3]) / sum(actions_cost)

    return actions_cost, actions_cnt, sum_keys, driven_info

class NERMatrics():
    def __init__(self, batch_predict_seq, batch_correct_seq, tags, step, model_name,
                 time_option = 2, tolerance_move = 3, stage="test", k4preannotated=float("inf"), annotation_decisions = None,
                 is_merge=True, consider_tagtype = None):
        self.batch_predict_seq, self.batch_correct_seq, self.tags, self.step, self.model_name, self.time_option, self.tolerance_move, self.stage = \
                                            batch_predict_seq, batch_correct_seq, tags, step, model_name, time_option, tolerance_move, stage
        self.is_merge = is_merge
        self.consider_tagtype = consider_tagtype
        self.k4preannotated = k4preannotated

        self.tags_remove_prefix = [tag.split("-")[-1] for tag in tags if tag != 'O']

        label_preffix = {"B-", "I-", "E-", "S-", "O"}
        for seq in self.batch_predict_seq:
            for label in seq:
                assert label[:2] in label_preffix
        for label in self.batch_correct_seq:
            for label in seq:
                assert label[:2] in label_preffix

        if self.consider_tagtype is not None:
            self.batch_predict_seq = [[tag if tag.split("-")[-1] in self.consider_tagtype else "O" for tag in row] for row in self.batch_predict_seq]
            self.batch_correct_seq = [[tag if tag.split("-")[-1] in self.consider_tagtype else "O" for tag in row] for row in self.batch_correct_seq]

        self.batch_predict_seq = [ NERMatrics.formatBIOES(row, is_merge = is_merge) for row in self.batch_predict_seq ]
        self.batch_correct_seq = [ NERMatrics.formatBIOES(row, is_merge = is_merge) for row in self.batch_correct_seq ]

        self.annotation_decisions = annotation_decisions
        if self.annotation_decisions is not None:
            assert len(self.annotation_decisions) == len(batch_predict_seq)

    @staticmethod
    def formatBIOES(seq, is_merge = False):
        if is_merge:
            # seq = [w.split("-")[-1] for w in seq]
            seq = ["I" + w[1:] if len(w) > 1 and w[1] == "-" else w for w in seq]

        ## 调整格式为BIOES格式
        ## 'I-b', 'E-b', 'E-b' = > ('b', 0, 1), ('b', 2, 2)
        entity = get_entities(seq)
        # >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        # >>> get_entities(seq)
        # [('PER', 0, 1), ('LOC', 3, 3)]
        seq = NERMatrics.entitys2seq(entity, len(seq))
        return seq

    @staticmethod
    def entitys2seq(entitys, length):
        # entitys格式是 [('PER', 0, 1), ('LOC', 3, 3)]
        # 从标签计数中恢复标签，并调整为BIOES格式
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

    # 定义相似
    @staticmethod
    def _is_similar(entity1, entity2, k, need_fea_match = False):
        # 这个例子就没考虑到，如果一个预测的entity同时包含了多个真实entity的情况。因此需要多考虑两种情况，一对多的情况。
        # 假设一个实体与多个不同的预测实体存在交集，那么预测实体当做是错的实体。
        # 如果多个预测的实体对应一个真实的实体，那么认为预测实体是错的实体。
        # test_target_seq   = ['B-a', 'E-a', 'O', 'O', 'S-b', 'S-c', 'O', 'O', 'O', 'O', 'B-a', 'E-a']
        # test_target_list = _get_entitys(test_target_seq)
        # test_origin_seql = ['S-a', 'O'  , 'O', 'O', 'B-d', 'E-d', 'O', 'O', 'B-a', 'I-a', 'I-a', 'E-a']
        # test_origin_list = _get_entitys(test_origin_seql)

        s1, e1, tags1 = entity1['start'], entity1['end'], entity1["tags"][0].split("-")[-1]
        s2, e2, tags2 = entity2['start'], entity2['end'], entity2["tags"][0].split("-")[-1]
        # 使得2实体不在1实体前面
        if s1 > s2:
            tmp_s, tmp_e = s1, e1
            s1, e1 = s2, e2
            s2, e2 = tmp_s, tmp_e

        intersection = None
        total_char = abs(e1 - s1) + 1 + abs(e2 - s2) + 1
        flag = False
        if s2 < e1 or s2 == e1:  # 存在交集
            # 因为是包含起点和终点的，所以加上1
            not_intersection = abs(s1 - s2) + abs(e1 - e2)
    #         print(not_intersection)
            intersection = (total_char - not_intersection) / 2
            total_unique_words = intersection + not_intersection
            if not_intersection > k:
                flag = False
            else:
                flag = True
                if need_fea_match and tags1 != tags2:
                    flag = False
        else:
            intersection = 0
            not_intersection = total_char
            total_unique_words = intersection + not_intersection
            flag = False

        return flag, intersection, total_unique_words

    @staticmethod
    def k_similar_entities(origin_entity_list, target_entity_list, k, need_fea_match=True):
        # 假设右边是正确的实体序列
        # 相似定义是，存在交集，非交集部分不超过k个单词，记录交集的单词数，非交集的单词数
        # （后面再改进，使用jaccard相似度，交集部分占的比例除以并集。jaccard需要大于k）
        # 相似的entity
        #     特征也一样，那么完全正确
        #     如果特征不一样，则需要修改
        # 不相似的entity
        #     出现在origin_entity_list，但不出现在target_entity_list，则为归类为标注错误的entity
        #     出现在target_entity_list，但不出现在origin_entity_list，则为归类为忘记标注的entity
        # 预测的：['O', 'O', 'O', 'O', 'B-ORG', 'B-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'S-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
        # 真实的：['O', 'O', 'O', 'O', 'B-LOC', 'E-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'S-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
        # 如果换别的算法，需要是stable sort，保证外部调用不会乱。
        origin_entity_list = sorted(origin_entity_list, key=lambda x: x['start'])
        target_entity_list = sorted(target_entity_list, key=lambda x: x['start'])

        #     print(origin_entity_list)
        #     print(target_entity_list)
        #     print("=========================================")

        origin_found_list = [[] for i in range(len(origin_entity_list))]
        target_found_list = [[] for i in range(len(target_entity_list))]
        for ith, origin_entity in enumerate(origin_entity_list):
            for jth, target_entity in enumerate(target_entity_list):
                flag, intersection_cnt, total_unique_words = NERMatrics._is_similar(origin_entity, target_entity, k, need_fea_match=need_fea_match)
                if flag:
                    origin_found_list[ith].append([jth, target_entity])
                    target_found_list[jth].append([ith, origin_entity])

        # 存放target实体， 存放origin实体
        return origin_found_list, target_found_list, origin_entity_list, target_entity_list

    @staticmethod
    def my_get_entitys(tags_seq):
        # 输入一个label序列，将一个完成label抽出来。比如：['B-fea1','E-fea1'] => {"tags":['B-fea1','E-fea1'], "start":0, "end":1}
        entity_list = []
        try:
            # 需要是BIEOS格式
            for i, a_tag in enumerate(tags_seq):
                prefix = a_tag[:2]
                if prefix == "B-":
                    an_entity = {"tags": [a_tag], "start":i, "end":None} # start and end inclusive
                elif prefix == "I-":
                    an_entity["tags"].append(a_tag)
                elif prefix == "E-":
                    an_entity["tags"].append(a_tag)
                    an_entity["end"] = i
                    entity_list.append(an_entity)
                elif prefix == "S-":
                    an_entity = {"tags":[a_tag], "start":i, "end":i} # start and end inclusive
                    entity_list.append(an_entity)
        except:
            print("err in _get_entitys: ", tags_seq)
        return entity_list

    def get_IIIscore(self):
        total_pred_correct = 0
        total_correct_preded = 0
        total_pred = 0
        total_correct = 0

        for predict_seq, correct_seq in zip(self.batch_predict_seq, self.batch_correct_seq):
            predict_entity_list = NERMatrics.my_get_entitys(predict_seq)
            correct_entity_list = NERMatrics.my_get_entitys(correct_seq)
            # origin_found_list, target_found_list, origin_entity_list, target_entity_list
            origin_found_list, target_found_list, origin_entity_list, target_entity_list = NERMatrics.k_similar_entities(predict_entity_list,
                                                                                    correct_entity_list, self.tolerance_move,
                                                                                    need_fea_match=True)
            # 预测的是正确的
            total_pred_correct += len([alist for alist in origin_found_list if len(alist) > 0])
            # 正确的被预测出来了
            total_correct_preded += len([alist for alist in target_found_list if len(alist) > 0])
            total_pred += len(predict_entity_list)
            total_correct += len(correct_entity_list)

        kfuzzy_recall, kfuzzy_precision, kfuzzy_f1_score = 0, 0, 0
        if total_correct == total_pred and total_correct == 0:
            kfuzzy_recall, kfuzzy_precision, kfuzzy_f1_score = 1, 1, 1
        else:
            kfuzzy_recall = total_correct_preded * 1.0 / total_correct if total_correct != 0 else 0
            kfuzzy_precision = total_pred_correct * 1.0 / total_pred if total_pred != 0 else 0
            kfuzzy_f1_score = 2.0 * kfuzzy_precision * kfuzzy_recall / (
                        kfuzzy_precision + kfuzzy_recall) if kfuzzy_precision != 0 or kfuzzy_recall != 0 else 0
        return kfuzzy_recall, kfuzzy_precision, kfuzzy_f1_score

    def get_formatseqs(self):
        return self.batch_predict_seq, self.batch_correct_seq

    def get_Iscore(self):
        # I 类指标
        nem_tp1, nem_tp2, total_entity, total_predict = \
            self.evaluate_batch_ner_seq(self.batch_predict_seq, self.batch_correct_seq, is_exact_match=False)

        nem_recall = nem_tp1 * 1.0 / total_entity if total_entity != 0 else 0
        nem_precision = nem_tp2 * 1.0 / total_predict if total_predict != 0 else 0
        nem_f1_score = 2.0 * nem_precision * nem_recall / (
                    nem_precision + nem_recall) if nem_precision != 0 or nem_recall != 0 else 0
        return nem_recall, nem_precision, nem_f1_score

    def get_fuzzy_report(self):
        def cal_results(tp, total_predict, total_correct):
            recall = tp * 1.0 / total_correct if total_correct != 0 else 0
            precision = tp * 1.0 / total_predict if total_predict != 0 else 0
            f1_score = 2.0 * precision * recall / (
                    precision + recall) if precision != 0 or recall != 0 else 0
            return recall, precision, f1_score

        def cal_results_for_each_tag(confirm_cnt_dict, predict_entity_cnt_dict, true_entity_cnt_dict):
            keys = set(predict_entity_cnt_dict.keys()) | set(true_entity_cnt_dict.keys())
            report = {key:{"recall":0, "precision":0, "f1-score":0} for key in keys}
            for key in keys:
                tp, total_predict, total_correct = 0, 0, 0
                if key in confirm_cnt_dict: tp = confirm_cnt_dict[key]
                if key in predict_entity_cnt_dict: total_predict = predict_entity_cnt_dict[key]
                if key in true_entity_cnt_dict: total_correct = true_entity_cnt_dict[key]
                recall, precision, f1_score = cal_results(tp, total_predict, total_correct)
                report[key]["recall"] = recall
                report[key]["precision"] = precision
                report[key]["f1-score"] = f1_score
            return report

        batch_confirm_cnt_dict, batch_predict_entity_cnt_dict, batch_true_entity_cnt_dict = \
            self.evaluate_batch_ner_seq_for_each_tag(self.batch_predict_seq, self.batch_correct_seq, self.tolerance_move)

        micro_recall, micro_precision, micro_f1 = cal_results(sum(batch_confirm_cnt_dict.values()),
                                                                sum(batch_predict_entity_cnt_dict.values()),
                                                                sum(batch_true_entity_cnt_dict.values()))

        report = cal_results_for_each_tag(batch_confirm_cnt_dict, batch_predict_entity_cnt_dict, batch_true_entity_cnt_dict)
        report["micro_avg"] = {}
        report["micro_avg"]["recall"] = micro_recall
        report["micro_avg"]["precision"] = micro_precision
        report["micro_avg"]["f1-score"] = micro_f1

        return micro_recall, micro_precision, micro_f1, report

    def get_IIscore(self):
        em_tp1, em_tp2, total_entity, total_predict = \
            self.evaluate_batch_ner_seq(self.batch_predict_seq, self.batch_correct_seq, is_exact_match=True)

        em_recall = em_tp1 * 1.0 / total_entity if total_entity != 0 else 0
        em_precision = em_tp2 * 1.0 / total_predict if total_predict != 0 else 0
        em_f1_score = 2.0 * em_precision * em_recall / (
                    em_precision + em_recall) if em_precision != 0 or em_recall != 0 else 0
        return em_recall, em_precision, em_f1_score

    def get_time_cost(self):

        time_cost_dict = self.cal_batch_time_cost(self.batch_predict_seq, self.batch_correct_seq, optional=self.time_option,
                                                  k=self.tolerance_move, k4preannotated=self.k4preannotated)

        return time_cost_dict

    def get_score_cost(self, ner_seq_result = None, time_cost_dict = None):

        if ner_seq_result is None:
            nem_recall, nem_precision, nem_f1_score = self.get_Iscore()
            em_recall, em_precision, em_f1_score = self.get_IIscore()
            kfuzzy_recall, kfuzzy_precision, kfuzzy_f1_score = self.get_IIIscore()

            ner_seq_result = {"nem_recall": nem_recall, "nem_precision": nem_precision, "nem_f1_score": nem_f1_score,
                              "em_recall": em_recall, "em_precision": em_precision, "em_f1_score": em_f1_score,
                              "kfuzzy_recall": kfuzzy_recall, "kfuzzy_precision": kfuzzy_precision,
                              "kfuzzy_f1_score": kfuzzy_f1_score}

        if time_cost_dict is None:
            time_cost_dict = self.get_time_cost()

        score_per_cost = {}

        # 需要处理0值
        for key in ner_seq_result:
            score_per_cost[key + "_per_token_plain"] = float("inf")
            score_per_cost[key + "_per_token_preannotated"] = float("inf")
            score_per_cost[key + "_per_cost_from_preannotated"] = float("inf")
            score_per_cost[key + "_per_cost_only_on_preannotated_without_add"] = float("inf")
            score_per_cost[key + "_per_cost_only_on_preannotated"] = float("inf")
            score_per_cost[key + "_per_cost_on_highlighted"] = float("inf")
            score_per_cost[key + "_per_cost_from_plain_text"] = float("inf")
            if time_cost_dict["total_total_tokens_cnt_plain"] > 0:
                score_per_cost[key + "_per_token_plain"] = ner_seq_result[key] / time_cost_dict["total_total_tokens_cnt_plain"]
            if time_cost_dict["total_total_tokens_cnt_preannotated"] > 0:
                score_per_cost[key + "_per_token_preannotated"] = ner_seq_result[key] / time_cost_dict["total_total_tokens_cnt_preannotated"]
            if time_cost_dict["total_time_cost_from_preannotated"] > 0:
                score_per_cost[key + "_per_cost_from_preannotated"] = ner_seq_result[key] / time_cost_dict["total_time_cost_from_preannotated"]
            if time_cost_dict["total_time_cost_only_on_preannotated_without_add"] > 0:
                score_per_cost[key + "_per_cost_only_on_preannotated_without_add"] = ner_seq_result[key] / time_cost_dict["total_time_cost_only_on_preannotated_without_add"]
            if time_cost_dict["total_time_cost_only_on_preannotated"] > 0:
                score_per_cost[key + "_per_cost_only_on_preannotated"] = ner_seq_result[key] / time_cost_dict["total_time_cost_only_on_preannotated"]
            if time_cost_dict["total_time_cost_on_highlighted"] > 0:
                score_per_cost[key + "_per_cost_on_highlighted"] = ner_seq_result[key] / time_cost_dict["total_time_cost_on_highlighted"]
            if time_cost_dict["total_time_cost_from_plain_text"] > 0:
                score_per_cost[key + "_per_cost_from_plain_text"] = ner_seq_result[key] / time_cost_dict["total_time_cost_from_plain_text"]

        return score_per_cost

    def get_ner_classification_report_score(self, need_report = False):
        ner_classification_report_results = self.batch_ner_classification_report(self.batch_predict_seq,
                                                                                 self.batch_correct_seq)
        em_recall = ner_classification_report_results["micro_avg"]["recall"]
        em_precision = ner_classification_report_results["micro_avg"]["precision"]
        em_f1_score = ner_classification_report_results["micro_avg"]["f1-score"]
        if need_report:
            return em_recall, em_precision, em_f1_score, ner_classification_report_results
        else:
            return em_recall, em_precision, em_f1_score

    def evaluate(self):
        nem_recall, nem_precision, nem_f1_score = self.get_Iscore()
        em_recall, em_precision, em_f1_score = self.get_IIscore()
        kfuzzy_recall, kfuzzy_precision, kfuzzy_f1_score = self.get_IIIscore()

        ner_seq_result = {"nem_recall":nem_recall,"nem_precision":nem_precision,"nem_f1_score":nem_f1_score,
         "em_recall":em_recall,"em_precision":em_precision,"em_f1_score":em_f1_score,
        "kfuzzy_recall":kfuzzy_recall, "kfuzzy_precision":kfuzzy_precision, "kfuzzy_f1_score":kfuzzy_f1_score}

        # II 类指标
        ner_valuator_results, ner_valuator_per_tag = self.batch_ner_valuator(self.batch_predict_seq, self.batch_correct_seq)
        
        ### III 类指标
        ner_classification_report_results = self.batch_ner_classification_report(self.batch_predict_seq, self.batch_correct_seq)

        # time cost
        time_cost_dict = self.get_time_cost()

        # time cost per tokens
        score_per_cost = self.get_score_cost(ner_seq_result=ner_seq_result, time_cost_dict=time_cost_dict)

        return ner_seq_result, ner_valuator_results, ner_valuator_per_tag, ner_classification_report_results, time_cost_dict, score_per_cost
    
    def log_evaluate(self):
        ner_seq_result, ner_valuator_results, ner_valuator_per_tag, ner_classification_report_results, \
                        time_cost_dict, score_per_cost = self.evaluate()

        prefix = self.stage + "+" + self.model_name + "_" + "step_" + str(self.step) + "+"
        
        fitlog.add_other(json.dumps(ner_seq_result), name=prefix + "ner_seq_result")
        fitlog.add_other(json.dumps(ner_valuator_results), name=prefix + "ner_valuator_results")
        fitlog.add_other(json.dumps(ner_valuator_per_tag), name=prefix + "ner_valuator_per_tag")
        fitlog.add_other(json.dumps(ner_classification_report_results), 
                         name=prefix + "ner_classification_report_results")

        # I 类指标
        for key in ner_seq_result:
            name = self.stage + "+" + "I+" + self.model_name + "+" + key
            fitlog.add_metric(ner_seq_result[key], name=name, step=self.step)
        
        # II 类指标
        results, results_per_tag = ner_valuator_results, ner_valuator_per_tag

        for cn in results:
            for idx in results[cn]:
                name = self.stage + "+" + "II+" + self.model_name + "+" + "+".join([cn,idx])
                fitlog.add_metric(results[cn][idx], name=name, step=self.step)

        for tag in results_per_tag:
            for cn in results_per_tag[tag]:
                for idx in results_per_tag[tag][cn]:
                    name = self.stage + "+" + "II+" + self.model_name + "+" + "+".join([tag, cn, idx])
                    fitlog.add_metric(results_per_tag[tag][cn][idx], name=name, step=self.step)
        
        
        ### III 类指标
        results2 = ner_classification_report_results

        for cn in results2:
            for idx in results2[cn]:
                name = self.stage + "+" + "III+" + self.model_name + "+" + "+".join([cn,idx])
                fitlog.add_metric(results2[cn][idx], name=name, step=self.step)

        # 记录标注时间
        for key in time_cost_dict:
            name = self.stage + "+" + self.model_name + "+" + key
            fitlog.add_metric(time_cost_dict[key], name=name, step=self.step)

        # 记录指标每单位消耗
        for key in score_per_cost:
            name = self.stage + "+" + self.model_name + "+" + key
            fitlog.add_metric(score_per_cost[key], name=name, step=self.step)

    def log_metrics(self, score, key):
        name = self.stage + "+" + "I+" + self.model_name + "+" + key
        fitlog.add_metric(score, name=name, step=self.step)

    def log_loss(self, loss_dict):
        loss_dict = deepcopy(loss_dict)
        for key in loss_dict:
            loss_dict[key] = float(loss_dict[key])
        for key in loss_dict:
            fitlog.add_loss(loss_dict[key], name=self.model_name + "+" + key, step=self.step)

    def log_other(self, recommended_samples=None, pre_tags=None, timer=None):
        recommended_samples = deepcopy(recommended_samples)
        pre_tags = deepcopy(pre_tags)
        prefix = self.model_name + "_" + "step_" + str(self.step) + "+"
        if recommended_samples is not None:
            # 避免json在dumps时候报错。TypeError: Object of type ‘int64’ is not JSON serializable。
            for key in recommended_samples:
                if isinstance(recommended_samples[key], list) or isinstance(recommended_samples[key], set):
                    recommended_samples[key] = [float(w) if "." in str(w) else int(w) for w in
                                                 recommended_samples[key]]
            fitlog.add_other(json.dumps(recommended_samples), name=prefix + "recommended_samples")

        if pre_tags is not None:
            fitlog.add_other(json.dumps(pre_tags), name=prefix + "pre_tags")

        if timer is not None:
            # 避免json在dumps时候报错。TypeError: Object of type ‘int64’ is not JSON serializable。
            for key in timer:
                timer[key] = [float(w) if "." in str(w) else int(w) for w in
                                                 timer[key]]
            fitlog.add_other(json.dumps(timer), name=prefix + "timer")


        if recommended_samples is not None:
            # 避免json在dumps时候报错。TypeError: Object of type ‘int64’ is not JSON serializable。
            for key in recommended_samples:
                recommended_samples[key] = [float(w) if "." in str(w) else int(w) for w in
                                                 recommended_samples[key]]
            fitlog.add_other(json.dumps(recommended_samples), name=prefix + "recommended_samples")

    def evaluate_ner_seq(self , predict_seq, correct_seq, is_exact_match = True):
        # predict_seq, correct_seq 各是一个laebl序列，laebl序列里头的每个元素是一个标签。这两个laebl序列长度相同因为对应同一个样本
        # label必须是"B-","I-","E-","S-"开头
        # cal according to https://stackoverrun.com/cn/q/353940
        # 精确匹配：边界和类别都得正确。模糊匹配：类别正确，有交集就行
        # true positives（TP）：NER能正确识别实体
        # false positives（FP）：NER识别出的实体，但实体不正确
        # false negatives（FN）：correct_seq中包含的实体，但是没被NER识别出来的

        def _match_cnt(origin_entity_list, target_seq, is_exact_match = True):
            # 统计origin_entity_list中有多少个实体出现在target_seq中。
            # 两种模式，精确匹配模式和模糊匹配模式。
            #     模糊匹配模式，有一个word的实体类别正确则实体正确，没有的话便是错误的。
            #     精确匹配模式，完全匹配才正确。
            try:
                tp = 0
                for entity in origin_entity_list:
                    tags = entity["tags"]
                    cur = entity["start"]
                    cnt = 0
                    while cur <= entity["end"]:
                        if is_exact_match == True:
        #                     print(tags[cur-entity["start"]], target_seq[cur])
                            if tags[cur-entity["start"]] == target_seq[cur]:
                                cnt += 1
                        else:
                            if tags[cur-entity["start"]].split("-")[-1] == target_seq[cur].split("-")[-1]:
                                cnt += 1

                        cur+=1

                    if is_exact_match:
                        if cnt == (entity["end"] - entity["start"]+1):
                            tp += 1
                    else:
                        if cnt > 0:
                            tp += 1
            except:
                print("err in _match_cnt: ", origin_entity_list, target_seq)

            return tp

        assert len(predict_seq) == len(correct_seq)
        predict_entity_list = NERMatrics.my_get_entitys(predict_seq)
        true_entity_list = NERMatrics.my_get_entitys(correct_seq)

        tp1 = _match_cnt(true_entity_list, predict_seq, is_exact_match=is_exact_match)  # 正确的实体，被正确预测的个数
        tp2 = _match_cnt(predict_entity_list, correct_seq, is_exact_match=is_exact_match) # 预测出的实体，正确的个数
        total_entity = len(true_entity_list)
        total_predict = len(predict_entity_list)

        # In case you need the following code for calculating the p/r/f in a batch.
        # (When your batch is the complete german_coarse_doc)
        # precision = tp2 * 1.0 / total_predict * 100 if total_predict != 0 else 0
        # recall = tp1 * 1.0 / total_entity * 100 if total_entity != 0 else 0
        # fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0

        return tp1, tp2, total_entity, total_predict

    def evaluate_ner_seq_for_each_tag(self , predict_seq, correct_seq, k):
        # predict_seq, correct_seq 各是一个laebl序列，laebl序列里头的每个元素是一个标签。这两个laebl序列长度相同因为对应同一个样本
        # label必须是"B-","I-","E-","S-"开头
        # cal according to https://stackoverrun.com/cn/q/353940
        # 精确匹配：边界和类别都得正确。模糊匹配：类别正确，有交集就行
        # true positives（TP）：NER能正确识别实体
        # false positives（FP）：NER识别出的实体，但实体不正确
        # false negatives（FN）：correct_seq中包含的实体，但是没被NER识别出来的

        def _match_cnt(origin_entity_list, target_seq, is_exact_match = True):
            # 统计origin_entity_list中有多少个实体出现在target_seq中。
            # 两种模式，精确匹配模式和模糊匹配模式。
            #     模糊匹配模式，有一个word的实体类别正确则实体正确，没有的话便是错误的。
            #     精确匹配模式，完全匹配才正确。
            try:
                tp = 0
                tp_dict = {}
                for entity in origin_entity_list:
                    tags = entity["tags"]
                    cur = entity["start"]
                    cnt = 0
                    while cur <= entity["end"]:
                        if is_exact_match == True:
        #                     print(tags[cur-entity["start"]], target_seq[cur])
                            if tags[cur-entity["start"]] == target_seq[cur]:
                                cnt += 1
                        else:
                            if tags[cur-entity["start"]].split("-")[-1] == target_seq[cur].split("-")[-1]:
                                cnt += 1
                        cur+=1

                    if is_exact_match:
                        if cnt == (entity["end"] - entity["start"]+1):
                            tp += 1
                    else:
                        if cnt > 0:
                            tp += 1
            except:
                print("err in _match_cnt: ", origin_entity_list, target_seq)

            return tp

        assert len(predict_seq) == len(correct_seq)
        predict_entity_list = NERMatrics.my_get_entitys(predict_seq)
        correct_entity_list = NERMatrics.my_get_entitys(correct_seq)
        # 把这个换成confirm entity list
        confirm_list, revise_list, delete_list, add_list, add_list_only_preannotated, add_list_similarity_preannotated = \
            NERMatrics.category_entities(predict_entity_list, correct_entity_list, k, k4preannotated=float("inf"), need_fea_match=False)

        keys = set({entity["tags"][0].split("-")[-1] for entity in predict_entity_list}) | set({entity["tags"][0].split("-")[-1] for entity in correct_entity_list})
        confirm_cnt_dict = {key:0 for key in keys}
        predict_entity_cnt_dict = deepcopy(confirm_cnt_dict)
        true_entity_cnt_dict = deepcopy(confirm_cnt_dict)
        for entity in confirm_list:
            a_tag = entity["tags"][0].split("-")[-1]
            confirm_cnt_dict[a_tag] += 1

        for entity in predict_entity_list:
            a_tag = entity["tags"][0].split("-")[-1]
            predict_entity_cnt_dict[a_tag] += 1

        for entity in correct_entity_list:
            a_tag = entity["tags"][0].split("-")[-1]
            true_entity_cnt_dict[a_tag] += 1
        # In case you need the following code for calculating the p/r/f in a batch.
        # (When your batch is the complete german_coarse_doc)
        # precision = tp2 * 1.0 / total_predict * 100 if total_predict != 0 else 0
        # recall = tp1 * 1.0 / total_entity * 100 if total_entity != 0 else 0
        # fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
        return confirm_cnt_dict, predict_entity_cnt_dict, true_entity_cnt_dict

    def evaluate_batch_ner_seq(self, batch_predict_seq, batch_correct_seq, is_exact_match = True):
        # label必须是"B-","I-","E-","S-"开头
        batch_tp1, batch_tp2, batch_total_entity, batch_total_predict = 0,0,0,0
        for predict_seq, correct_seq in zip(batch_predict_seq, batch_correct_seq):
            # print(predict_seq, correct_seq)
            seq_tp1, seq_tp2, seq_total_entity, seq_total_predict = self.evaluate_ner_seq(predict_seq, correct_seq, is_exact_match = is_exact_match)
            batch_tp1 += seq_tp1
            batch_tp2 += seq_tp2
            batch_total_entity += seq_total_entity
            batch_total_predict += seq_total_predict

        # In case you need the following code for calculating the p/r/f in a batch.
        # (When your batch is the complete german_coarse_doc)
        # precision = batch_tp2 * 1.0 / batch_total_predict * 100 if total_predict != 0 else 0
        # recall = batch_tp1 * 1.0 / batch_total_entity * 100 if total_entity != 0 else 0
        # fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
        return batch_tp1, batch_tp2, batch_total_entity, batch_total_predict

    def evaluate_batch_ner_seq_for_each_tag(self, batch_predict_seq, batch_correct_seq, k):
        def _update_dict(target_dcit, source_dict):
            for key in source_dict:
                if key not in target_dcit:
                    target_dcit[key] = source_dict[key]
                else:
                    target_dcit[key] += source_dict[key]
            return target_dcit

        # label必须是"B-","I-","E-","S-"开头
        batch_confirm_cnt_dict, batch_predict_entity_cnt_dict, batch_true_entity_cnt_dict = {},{},{}
        for predict_seq, correct_seq in zip(batch_predict_seq, batch_correct_seq):
            # print(predict_seq, correct_seq)
            confirm_cnt_dict, predict_entity_cnt_dict, true_entity_cnt_dict = self.evaluate_ner_seq_for_each_tag(predict_seq, correct_seq, k)
            batch_confirm_cnt_dict = _update_dict(batch_confirm_cnt_dict, confirm_cnt_dict)
            batch_predict_entity_cnt_dict = _update_dict(batch_predict_entity_cnt_dict, predict_entity_cnt_dict)
            batch_true_entity_cnt_dict = _update_dict(batch_true_entity_cnt_dict, true_entity_cnt_dict)
        # In case you need the following code for calculating the p/r/f in a batch.
        # (When your batch is the complete german_coarse_doc)
        # precision = batch_tp2 * 1.0 / batch_total_predict * 100 if total_predict != 0 else 0
        # recall = batch_tp1 * 1.0 / batch_total_entity * 100 if total_entity != 0 else 0
        # fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
        return batch_confirm_cnt_dict, batch_predict_entity_cnt_dict, batch_true_entity_cnt_dict

    def to_prodigy_format(self, a_seq):
        entities = get_entities(a_seq)
        dict_list = []
        for entity in entities:
            adict = {"label":entity[0], "start": entity[1], "end": entity[2]}
            dict_list.append(adict)
        return dict_list

    def batch_ner_valuator(self, pred, true):
        true = [self.to_prodigy_format(a_seq) for a_seq in true]
        pred = [self.to_prodigy_format(a_seq) for a_seq in pred]
        evaluator = NEREvaluator(true, pred, self.tags_remove_prefix)
        results, results_per_tag = evaluator.evaluate()
        # results: 两层字典, 第一层是评估模式(ent_type, partial, strict, exact)，第二层是评估类型(f1, precision, recall e.g.)
        # results_per_tag: 三层字典, 第一层是tag, 第二层是评估模式('ent_type', 'partial', 'strict', 'exact'], dtype='object')，
        #    第三层是评估类型('actual', 'correct', 'f1', 'incorrect', 'missed', 'partial', 'possible', 'precision', 'recall', 'spurious')
        return results, results_per_tag

    def batch_ner_classification_report(self, pred, true):
        try:
            result_str = ner_classification_report(true, pred)
        except:
            first_level = ['MISC', 'PER', 'micro_avg', 'macro_avg']
            second_level = ['precision', 'recall', 'f1-score', 'support']
            result = {}
            for cn in first_level:
                result[cn] = {}
                for cn2 in second_level:
                    result[cn][cn2] = 0
            return result

        row_str = [[w for w in string.split("\n")] for string in result_str.split("\n")]
        row_str = [arow[0] for arow in row_str if arow[0]!= ""]
        table = [[w for w in string.split(" ") if w != ""] for string in [arow for arow in row_str]]
        header = table[0] 
        table = table[1:]
        indexes = ["_".join(arow[:len(arow) - len(header)]) for arow in table]
        table = [arow[len(arow) - len(header):] for arow in table]
        result = {}
        for i, acn in enumerate(indexes):
            result[acn] = {}
            for j, aidx in enumerate(header):
                result[acn][aidx] = float(table[i][j])
        # result is a two level dict, the first level are 'MISC', 'PER', 'micro_avg' and 'macro_avg'. 
        # the second level are 'precision', 'recall', 'f1-score' and 'support'
        return result

    def cal_time_cost(self, predict_seq, correct_seq, optional = 2, k = 1, k4preannotated=float("inf")):

        predict_entity_list = NERMatrics.my_get_entitys(predict_seq)
        correct_entity_list = NERMatrics.my_get_entitys(correct_seq)
        total_tokens_cnt = len(correct_seq)
        annotated_tokens_cnt = len([w for w in correct_seq if w != 'O'])
        confirm_list, revise_list, delete_list, add_list, add_list_only_preannotated, add_list_similarity_preannotated = \
            NERMatrics.category_entities(predict_entity_list, correct_entity_list, k, k4preannotated=k4preannotated, need_fea_match=False)

    #     info_dict["confirm_list_preannotated"] = confirm_list
    #     info_dict["revise_list_preannotated"] = revise_list
    #     info_dict["delete_list_preannotated"] = delete_list
    #     info_dict["add_list_preannotated"] = add_list
    #     info_dict["correct_entity_list"] = correct_entity_list
    #     info_dict["add_list_only_preannotated"] = add_list_only_preannotated
        info_dict = {}
        info_dict["confirm_cnt_preannotated"] = len(confirm_list)
        info_dict["revise_cnt_preannotated"] = len(revise_list)
        info_dict["delete_cnt_preannotated"] = len(delete_list)
        info_dict["add_cnt_preannotated"] = len(add_list)
        info_dict["total_tokens_cnt_plain"] = total_tokens_cnt
        info_dict["add_cnt_from_plain"] = len(correct_entity_list)
        info_dict["total_tokens_cnt_preannotated"] = annotated_tokens_cnt
        info_dict["add_only_preannotated_cnt"] = len(add_list_only_preannotated)
        info_dict["add_similarity_preannotated_cnt"] = len(add_list_similarity_preannotated)

        time_dict = self.cal_time_function(info_dict, optional=optional)

        info_dict.update(time_dict)

        return info_dict

    def cal_batch_time_cost(self, batch_predict_seq, batch_correct_seq, optional = 2, k = 1, k4preannotated=float("inf")):
        # label必须是"B-","I-","E-","S-"开头

        perfect_decision = []
        return_dict = defaultdict(lambda: 0)

        # if len(batch_predict_seq) == 0:
        #     return return_dict

        return_dict["total_confirm_cnt_preannotated"] = 0
        return_dict["total_revise_cnt_preannotated"] = 0
        return_dict["total_delete_cnt_preannotated"] = 0
        return_dict["total_add_cnt_preannotated"] = 0
        return_dict["total_total_tokens_cnt_plain"] = 0
        return_dict["total_add_cnt_from_plain"] = 0
        return_dict["total_total_tokens_cnt_preannotated"] = 0
        return_dict["total_add_only_preannotated_cnt"] = 0
        return_dict["total_add_similarity_preannotated_cnt"] = 0

        return_dict["total_confirm_time_cost_preannotated"] = 0
        return_dict["total_revise_time_cost_preannotated"] = 0
        return_dict["total_delete_time_cost_preannotated"] = 0
        return_dict["total_add_time_cost_preannotated"] = 0
        return_dict["total_read_time_cost_plain"] = 0
        return_dict["total_add_time_cost_plain"] = 0
        return_dict["total_read_time_cost_preannotated"] = 0
        return_dict["total_add_time_cost_only_preannotated"] = 0
        return_dict["total_add_time_cost_similarity_preannotated"] = 0

        return_dict["total_confirm_time_cost_preannotated_plus_add_time_cost_preannotated"] = 0
        return_dict["total_time_cost_from_preannotated_except_read_and_confirm"] = 0
        return_dict["total_time_cost_from_preannotated_except_read"] = 0
        return_dict["total_time_cost_from_preannotated"] = 0
        return_dict["total_time_cost_only_on_preannotated_without_add"] = 0
        return_dict["total_time_cost_only_on_preannotated"] = 0
        return_dict["total_time_cost_on_highlighted"] = 0
        return_dict["total_time_cost_from_plain_text"] = 0

        for ith, [predict_seq, correct_seq] in enumerate(list(zip(batch_predict_seq, batch_correct_seq))):
            info_dict = self.cal_time_cost(predict_seq, correct_seq, optional = optional, k = k, k4preannotated=k4preannotated)

            return_dict["total_confirm_cnt_preannotated"] += info_dict["confirm_cnt_preannotated"]
            return_dict["total_revise_cnt_preannotated"] += info_dict["revise_cnt_preannotated"]
            return_dict["total_delete_cnt_preannotated"] += info_dict["delete_cnt_preannotated"]
            return_dict["total_add_cnt_preannotated"] += info_dict["add_cnt_preannotated"]
            return_dict["total_total_tokens_cnt_plain"] += info_dict["total_tokens_cnt_plain"]
            return_dict["total_add_cnt_from_plain"] += info_dict["add_cnt_from_plain"]
            return_dict["total_total_tokens_cnt_preannotated"] += info_dict["total_tokens_cnt_preannotated"]
            return_dict["total_add_only_preannotated_cnt"] += info_dict["add_only_preannotated_cnt"]
            return_dict["total_add_similarity_preannotated_cnt"] += info_dict["add_similarity_preannotated_cnt"]

            return_dict["total_confirm_time_cost_preannotated"] += info_dict["confirm_time_cost_preannotated"]
            return_dict["total_revise_time_cost_preannotated"] += info_dict["revise_time_cost_preannotated"]
            return_dict["total_delete_time_cost_preannotated"] += info_dict["delete_time_cost_preannotated"]
            return_dict["total_add_time_cost_preannotated"] += info_dict["add_time_cost_preannotated"]
            return_dict["total_read_time_cost_plain"] += info_dict["read_time_cost_plain"]
            return_dict["total_add_time_cost_plain"] += info_dict["add_time_cost_plain"]
            return_dict["total_read_time_cost_preannotated"] += info_dict["read_time_cost_preannotated"]
            return_dict["total_add_time_cost_only_preannotated"] += info_dict["add_time_cost_only_preannotated"]
            return_dict["total_add_time_cost_similarity_preannotated"] += info_dict["add_time_cost_similarity_preannotated"]

            return_dict["total_confirm_time_cost_preannotated_plus_add_time_cost_preannotated"] += info_dict["confirm_time_cost_preannotated_plus_add_time_cost_preannotated"]
            return_dict["total_time_cost_from_preannotated_except_read_and_confirm"] += info_dict["time_cost_from_preannotated_except_read_and_confirm"]
            return_dict["total_time_cost_from_preannotated_except_read"] += info_dict["time_cost_from_preannotated_except_read"]
            return_dict["total_time_cost_from_preannotated"] += info_dict["time_cost_from_preannotated"]
            return_dict["total_time_cost_only_on_preannotated_without_add"] += info_dict["time_cost_only_on_preannotated_without_add"]
            return_dict["total_time_cost_only_on_preannotated"] += info_dict["time_cost_only_on_preannotated"]
            return_dict["total_time_cost_on_highlighted"] += info_dict["time_cost_on_highlighted"]
            return_dict["total_time_cost_from_plain_text"] += info_dict["time_cost_from_plain_text"]

            if self.annotation_decisions is not None:
                if self.annotation_decisions[ith] == 0:
                    return_dict["total_time_cost_by_decision"] += info_dict["time_cost_from_plain_text"]
                else:
                    return_dict["total_time_cost_by_decision"] += info_dict["time_cost_from_preannotated"]

            if self.annotation_decisions is not None:
                if self.annotation_decisions[ith] == 0:
                    return_dict["total_time_cost_without_read_by_decision"] += info_dict["add_time_cost_plain"]
                else:
                    return_dict["total_time_cost_without_read_by_decision"] += info_dict["time_cost_from_preannotated_except_read"]

            return_dict["total_time_cost_by_perfect_decision"] += min([info_dict["time_cost_from_plain_text"], info_dict["time_cost_from_preannotated"]])

            if info_dict["time_cost_from_plain_text"] > info_dict["time_cost_from_preannotated"]:
                perfect_decision.append(1)
            else:
                perfect_decision.append(0)

            return_dict["total_time_cost_without_read_by_perfect_decision"] += min(
                [info_dict["add_time_cost_plain"], info_dict["time_cost_from_preannotated_except_read"]])

        if self.annotation_decisions is not None:
            return_dict["machine_ratio_select_plain"] += 1 - sum(self.annotation_decisions) / len(self.annotation_decisions)

        if len(perfect_decision) > 0:
            return_dict["pefect_ratio_select_plain"] += 1 - sum(perfect_decision) / len(perfect_decision)

        # ZeroDivisionError: division by zero检查active learning utils中是否传入0样本的序列，可能是抽样不够样本了。
        return return_dict

    @staticmethod
    def category_entities(origin_entity_list, target_entity_list, k, k4preannotated=float("inf"),
                          need_fea_match=False):
        # 假设右边是正确的实体序列
        # 相似定义是，存在交集，非交集部分不超过k个单词，记录交集的单词数，非交集的单词数
        # （后面再改进，使用jaccard相似度，交集部分占的比例除以并集。jaccard需要大于k）
        # 相似的entity
        #     特征也一样，那么完全正确
        #     如果特征不一样，则需要修改
        # 不相似的entity
        #     出现在origin_entity_list，但不出现在target_entity_list，则为归类为标注错误的entity
        #     出现在target_entity_list，但不出现在origin_entity_list，则为归类为忘记标注的entity

        # test cases
        # target_entity_list = [{"start":1, "end":2, "tags":["A-A"]*2}, {"start":3, "end":10, "tags":["B-B"]*2},
        #                       {"start":11, "end":11, "tags":["A-A"]}, {"start":12, "end":15, "tags":["B-B"]*2},
        #                       {"start":16, "end":17, "tags":["A-A"]}, {"start":19, "end":20, "tags":["C-C"]},
        #                      {"start":99, "end":100, "tags":["C-C"]}]
        # origin_entity_list = [{"start":2, "end":3, "tags":["A-A"]*2}, {"start":11, "end":14, "tags":["A-A"]},
        #                       {"start":17, "end":18, "tags":["A-A"]}, {"start":20, "end":21, "tags":["B-B"]},
        #                      {"start":55, "end":66, "tags":["C-C"]}]
        # confirm_list, revise_list, delete_list, add_list, add_list_only_preannotated, add_list_similarity_preannotated = \
        #             category_entities(origin_entity_list, target_entity_list, 5, k4preannotated = float("inf"),
        #                       need_fea_match=False)
        # # confirm_list = [{'start': 17, 'end': 18, 'tags': ['A-A']}]
        # # revise_list = [{'start': 20, 'end': 21, 'tags': ['B-B']}]
        # # delete_list = [{'start': 2, 'end': 3, 'tags': ['A-A', 'A-A']},
        # #                   {'start': 11, 'end': 14, 'tags': ['A-A']},
        # #                   {'start': 55, 'end': 66, 'tags': ['C-C']}]
        # # add_list = [{'start': 1, 'end': 2, 'tags': ['A-A', 'A-A']},
        # #               {'start': 3, 'end': 10, 'tags': ['B-B', 'B-B']},
        # #               {'start': 11, 'end': 11, 'tags': ['A-A']},
        # #               {'start': 12, 'end': 15, 'tags': ['B-B', 'B-B']},
        # #               {'start': 99, 'end': 100, 'tags': ['C-C']}]

        # 定义四类标注类型
        confirm_list = []
        revise_list = []
        delete_list = []
        add_list = []

        orinin_delete_flag = [0] * len(origin_entity_list)
        target_add_flag = [0] * len(target_entity_list)

        #     origin_entity_list, target_entity_list, origin_found_list, target_found_list = k_similar_entities(origin_entity_list, target_entity_list, k)
        # 存在交集的就认为相似
        origin_ol_list, target_ol_list, _, _ = NERMatrics.k_similar_entities(origin_entity_list, target_entity_list,
                                                                             float("inf"),
                                                                             need_fea_match=need_fea_match)

        # 存在交集的且不相交部分小于k才相似
        origin_found_list, target_found_list, origin_entity_list, target_entity_list = NERMatrics.k_similar_entities(
            origin_entity_list, target_entity_list, k,
            need_fea_match=need_fea_match)

        #     print(origin_found_list)
        #     print(target_found_list)

        for ith, a_list in enumerate(target_ol_list):
            if len(a_list) > 1:  # 一个目标实体与多个预测实体有交集
                fea_list = set()
                # 不管是不是被拆分成多个首尾相连的实体。只要拆分成多个都认为不对。
                for jth, origin_entity in a_list:
                    orinin_delete_flag[jth] = 1

                target_add_flag[ith] = 1

        for ith, a_list in enumerate(origin_ol_list):
            if len(a_list) > 1:  # 一个预测实体与多个目标实体有交集
                #             print(a_list)
                for jth, target_entity in a_list:
                    target_add_flag[jth] = 1

                orinin_delete_flag[ith] = 1

        for ith, a_list in enumerate(target_found_list):
            if target_add_flag[ith] == 1:
                add_list.append(target_entity_list[ith])

            # 没被发现的目标实体
            if len(a_list) == 0:
                if not target_add_flag[ith]:
                    add_list.append(target_entity_list[ith])
                    target_add_flag[ith] = 1

        for ith, a_list in enumerate(origin_found_list):
            if orinin_delete_flag[ith] == 1:  #
                delete_list.append(origin_entity_list[ith])
                continue

            if len(a_list) == 0:  # 多标注的实体需要删掉
                delete_list.append(origin_entity_list[ith])
            elif len(a_list) == 1:  # 可能是标注对的实体，也可能是高亮对了，但是标签给错了
                #             print(a_list)
                if origin_entity_list[ith]['tags'][0].split('-')[1] == a_list[0][1]['tags'][0].split('-')[1]:
                    confirm_list.append(origin_entity_list[ith])
                else:
                    revise_list.append(origin_entity_list[ith])

            else:  # 一个预测实体对应多个真实实体
                delete_list.append(origin_entity_list[ith])

        add_found_list, _, sorted_add_list, _ = NERMatrics.k_similar_entities(add_list, origin_entity_list,
                                                                              k4preannotated,
                                                                              need_fea_match=need_fea_match)

        # add time cost preannotated 文本已经进行预标注的情况下，只在预标注的地方找出需要标注的实体。与要被删除的预测实体相似的真实实体。
        add_list_only_preannotated = []
        for ith, alist in enumerate(add_found_list):
            #     print(alist)
            if len(alist) > 0:
                add_list_only_preannotated.append(sorted_add_list[ith])

        # add similarity preannotated 从高亮文本中找出需要标注的实体
        add_list_similarity_preannotated = []
        for ith, alist in enumerate(target_found_list):
            #     print(alist)
            if len(alist) > 0:
                add_list_similarity_preannotated.append(target_entity_list[ith])

        return confirm_list, revise_list, delete_list, add_list, add_list_only_preannotated, add_list_similarity_preannotated

    @staticmethod
    def get_time_estimate_para():
        # unit : second
        time_estimate = [{'confirm': 1.68, 'revise': 4.38, 'delete': 0.31, 'add': 5.32, 'read': 0.49},
                         {'confirm': 1.71, 'revise': 9.54, 'delete': 0, 'add': 1.79, 'read': 0.49},
                         {'confirm': 2.06, 'revise': 5.99, 'delete': 1.61, 'add': 4.35, 'read': 0.49}, # 2 看录屏记录
                         {'confirm': 1, 'revise': 4, 'delete': 1.5, 'add': 10, 'read': 0.15},  # 3 根据经验调整后的参数
                         {'confirm': 1, 'revise': 4, 'delete': 1.5, 'add': 10, 'read': 0.05},  # 4 根据经验调整后的参数
                         {'confirm': 1, 'revise': 4, 'delete': 1.5, 'add': 10, 'read': 0.5},  # 5 根据经验调整后的参数
                         {'confirm': 1, 'revise': 4, 'delete': 1.5, 'add': 10, 'read': 1},  # 6 2*read 根据经验调整后的参数
                         {'confirm': 1, 'revise': 4, 'delete': 1.5, 'add': 20, 'read': 0.5},  # 7 2*add 根据经验调整后的参数
                         {'confirm': 1, 'revise': 4, 'delete': 3, 'add': 10, 'read': 0.5},  # 8 6 2*delete 根据经验调整后的参数
                         {'confirm': 1, 'revise': 8, 'delete': 1.5, 'add': 10, 'read': 0.5},  # 9 2*revise 根据经验调整后的参数
                         {'confirm': 2, 'revise': 4, 'delete': 1.5, 'add': 10, 'read': 0.5},  # 10 2*confirm 根据经验调整后的参数
                         ]
        return time_estimate

    def cal_time_function(self, info_dict, optional=0):
        return_dict = {}

        # unit : second
        time_estimate = NERMatrics.get_time_estimate_para()

        an_estimate = time_estimate[optional]
        return_dict["confirm_time_cost_preannotated"] = info_dict["confirm_cnt_preannotated"] * an_estimate['confirm']
        return_dict["revise_time_cost_preannotated"] = info_dict["revise_cnt_preannotated"] * an_estimate['revise']
        return_dict["delete_time_cost_preannotated"] = info_dict["delete_cnt_preannotated"] * an_estimate['delete']
        return_dict["add_time_cost_preannotated"] = info_dict["add_cnt_preannotated"] * an_estimate['add']
        return_dict["read_time_cost_plain"] = info_dict["total_tokens_cnt_plain"] * an_estimate['read']
        return_dict["add_time_cost_plain"] = info_dict["add_cnt_from_plain"] * an_estimate['add']
        return_dict["read_time_cost_preannotated"] = info_dict["total_tokens_cnt_preannotated"] * an_estimate['read']
        return_dict["add_time_cost_only_preannotated"] = info_dict["add_only_preannotated_cnt"] * an_estimate['add']
        return_dict["add_time_cost_similarity_preannotated"] = info_dict["add_similarity_preannotated_cnt"] * an_estimate['add']

        return_dict["confirm_time_cost_preannotated_plus_add_time_cost_preannotated"] = return_dict["confirm_time_cost_preannotated"] + return_dict["add_time_cost_preannotated"]

        return_dict["time_cost_from_preannotated_except_read_and_confirm"] = return_dict["add_time_cost_preannotated"] + return_dict["revise_time_cost_preannotated"] + \
                                                                             return_dict["delete_time_cost_preannotated"]

        return_dict["time_cost_from_preannotated_except_read"] = return_dict["confirm_time_cost_preannotated_plus_add_time_cost_preannotated"] + \
                                                                    return_dict["revise_time_cost_preannotated"] + return_dict["delete_time_cost_preannotated"]

        return_dict["time_cost_from_preannotated"] = return_dict["time_cost_from_preannotated_except_read"] + return_dict["read_time_cost_plain"]


        return_dict["time_cost_only_on_preannotated_without_add"] = return_dict["confirm_time_cost_preannotated"] + return_dict["revise_time_cost_preannotated"] + \
                                                                    return_dict["delete_time_cost_preannotated"] + return_dict["read_time_cost_preannotated"]

        return_dict["time_cost_only_on_preannotated"] = return_dict["time_cost_only_on_preannotated_without_add"] + return_dict["add_time_cost_only_preannotated"]

        return_dict["time_cost_on_highlighted"] = return_dict["add_time_cost_similarity_preannotated"] + return_dict["read_time_cost_preannotated"]


        return_dict["time_cost_from_plain_text"] = return_dict["add_time_cost_plain"] + return_dict["read_time_cost_plain"]

        return return_dict
