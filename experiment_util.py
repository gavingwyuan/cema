from model.CostEstimator import CostPerformaceEval
from model.TransformerNER import TransformerNER
from model.TransformerActionPredictor import TransformerActionPredictor, TFAPDataset
from active_learning.acquisition import get_rnd_topk, get_min_len_topk, get_least_topk
from active_learning.acquisition import CEMA_Doc_Selctor
import numpy as np
from copy import deepcopy
import os
from shutil import copyfile
from active_learning.acquisition import CEMAPartialStrategy
from data_processed.split_data import to_segmented

filter_part = 1000

def get_action_predictor(train_sents, tagger, logger, args, save_model=False, required_preds = True, ap_batch_size = 2):

    seg_train_sents, _, _, _ = to_segmented(train_sents, end_symbol=args.ner_end_symbol, initial_num=0)

    action_predictor = TransformerActionPredictor(list(args.unique_label_list), learning_rate=args.ap_lr,
                                                  cls_model_name=args.ap_cls_model_name,
                                                  lstm_out_features=None,
                                                  transformers_nhid=args.ap_transformer_nhids,
                                                  epochs=args.ap_epochs, batch_size=ap_batch_size,
                                                  device=args.ap_device, output_folder=args.at_query_strategy + "_aggressive_action_predictor",
                                                  base_model_name=args.ap_base_model_name,
                                                  max_length=args.ner_max_length, end_symbol=args.ap_end_symbol)

    if len(seg_train_sents) > 0:
        labels_set = set(tagger.label2idx) - {"O"}
        labels_set = {label.split("-")[-1] for label in labels_set} - {"O"}

        seg_train_preds = tagger.predict(seg_train_sents)
        train_sents_4ap, train_sents_4ap_true_labels, seg_train_sents, seg_train_preds = TFAPDataset.get_ap_input(
            seg_train_sents, seg_train_preds,
            labels_set, args.ner_end_symbol,
            sim_threshold=args.at_sim_threshold,
            sim_method=args.at_sim_method)
        if len(train_sents_4ap) > 0:
            action_predictor.train(train_sents_4ap, logger=logger, save_model=save_model)

    return action_predictor

def gen_new_labels(sents, sents_preds, action_predictor, cls_labels, args):
    sents_sents_4ap = TFAPDataset.get_tf_ap_input(sents, sents_preds,
                                                    sim_threshold=args.at_sim_threshold,
                                                    sim_method=args.at_sim_method)
    pred_action_labels = action_predictor.predict(sents_sents_4ap)

    merged_labels = []
    for i in range(len(cls_labels)):
        if cls_labels[i] > 0 or len(set(pred_action_labels[i]) - {"O", "o"}) > 0:
            merged_labels.append(1)
        else:
            merged_labels.append(0)

    action_driven_labels = []
    for i in range(len(cls_labels)):
        if len(set(pred_action_labels[i]) - {"O", "o"}) > 0:
            action_driven_labels.append(1)
        else:
            action_driven_labels.append(0)

    return merged_labels, action_driven_labels

def my_copyfiles(source_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    # fetch all files
    for file_name in os.listdir(source_folder):
        # construct full file path
        source = os.path.join(source_folder, file_name)
        destination = os.path.join(destination_folder, file_name)
        # copy only files
        if os.path.isfile(source):
            copyfile(source, destination)
            print('copied', file_name)
        else:
            my_copyfiles(source, destination)

def get_action_labels(action_predictor, sents, pred_labels, args, logger, type="confirmation_reading"):
    if action_predictor is not None:
        if not hasattr(args, "cema_type") or "SelfReviewNER" not in args.cema_type:
            # 正常
            sent_4ap = TFAPDataset.get_tf_ap_input(sents, pred_labels,
                                                       sim_threshold=args.at_sim_threshold,
                                                       sim_method=args.at_sim_method)
            logger.info('start to get actions')
            action_labels = action_predictor.predict(sent_4ap)
            logger.info('end of getting actions')
        else:
            _, action_labels = action_predictor.predict_start_from_tokens(sents, iterations=0,
                                                                                input_pred_labels=pred_labels,
                                                                                adjust_action=True)
    else:
        # without action predictor
        if type == "only_reading": # 没用到ground true
            # cost = only reading cost
            action_labels = [["O"] * len(row) for row in pred_labels]
        elif type == "confirmation_reading": # 没用到ground true
            # cost = confirmation cost + reading cost while assuming all predicted labels are correct
            cpe = CostPerformaceEval(true_label_list=pred_labels, pred_labels_list=pred_labels,
                                     sim_threshold=args.at_sim_threshold, sim_method=args.at_sim_method)
            action_labels = [aa.action_labels for aa in cpe.annotaion_actions]
        elif type == "ground_true" or type == "ground_true_for_cost" or type == "ground_true_for_add_paragraph": # 用到ground true
            true_label_list = []
            for sent in sents:
                true_label_list.append([label for token, label in sent])
            cpe = CostPerformaceEval(true_label_list=true_label_list, pred_labels_list=pred_labels,
                                     sim_threshold=args.at_sim_threshold, sim_method=args.at_sim_method)
            action_labels = [aa.action_labels for aa in cpe.annotaion_actions]
        else:
            raise NotImplementedError

    return action_labels

def get_refer_action_labels(all_sents, pred_labels, sent_trn, tagger, logger, args, save_model=False, required_preds=True):
    # refer_action_labels = deepcopy(action_labels[len(sent_trn):])
    action_predictor_for_add = get_action_predictor(sent_trn, tagger, logger, args, save_model=save_model, required_preds=required_preds,
                                                    ap_batch_size = args.ap_batch_size)
    refer_action_labels = get_action_labels(action_predictor_for_add, all_sents,
                                            pred_labels, args, logger, type="use action predictor")
    # del action_predictor_for_add
    return refer_action_labels, action_predictor_for_add

def get_next_k_instance(sent_trn, sent_pool, model, step, args, logger, test_sents,
                        tagger = None, action_predictor = None):
    entity_embedding_step_size = 30
    sim_mat_mul_batch_size = 256
    results = []
    logger.info(' * Start learning phase *')
    for t in range(0, args.at_steps_in_iteration):
        if len(sent_pool) == 0:
            raise NotImplementedError("waring => sent_pool is empty **************** ")

        step_results = {"step":None, "selected_sents": [], "selected_pred_labels": [], "selected_preds_actions": [],
                        "test_sent": "only keep it in step 0", "test_sent_pred_labels": "only keep in some steps",
                        "orignial_sent":[],
                        "selected_refer_preds_actions":[]}
        doc_num = args.doc_num_a_step
        at_query_strategy = args.at_query_strategy

        if at_query_strategy == "random":
            logger.info(" * step-{} Random selections".format(step))
            top_k_indices = get_rnd_topk(sent_pool, doc_num)
        elif "cema_partial" in at_query_strategy:
            logger.info(" * step-{} CEMA (partial segmented) selections".format(step))

            all_sents = sent_trn + sent_pool
            train_index = list(range(len(sent_trn))) # all_sents的topk个是被标注的样本
            pred_labels = model.predict(all_sents)
            # merge entity
            pred_labels = [CostPerformaceEval.formatBIOES(seq, is_merge=args.at_is_merge) for seq in pred_labels]
            refer_pred_labels = deepcopy(pred_labels)

            if args.cema_type == "ground_true":
                action_labels_for_cost = get_action_labels(action_predictor, all_sents, pred_labels, args, logger,
                                                           type=args.cema_type)
                refer_action_labels = deepcopy(action_labels_for_cost)
            elif args.cema_type == "ground_true_for_cost":
                action_labels_for_cost = get_action_labels(action_predictor, all_sents, pred_labels, args, logger,
                                                           type=args.cema_type)
                refer_action_labels = [["O" for x in row] for row in refer_pred_labels]
            elif args.cema_type == "ground_true_for_add_paragraph":
                action_labels_for_cost = get_action_labels(None, all_sents, pred_labels, args, logger, type="confirmation_reading")
                refer_action_labels = get_action_labels(action_predictor, all_sents, pred_labels, args, logger,
                                                           type=args.cema_type)
            elif args.cema_type == "use action predictor only for cost":
                action_labels_for_cost = get_action_labels(action_predictor, all_sents, pred_labels, args, logger,
                                                           type=args.cema_type)
                refer_action_labels = [["O" for x in row] for row in action_labels_for_cost]
            else:
                # args.cema_type == "confirmation_reading"
                # args.cema_type == "only_reading"
                # args.cema_type == "use action predictor"
                # args.cema_type == "argumented_AP"
                # args.cema_type == "shared_layer_NER_AP"
                # args.cema_type == "shared_layer_NER_argumented_AP"
                action_labels_for_cost = get_action_labels(action_predictor, all_sents, pred_labels, args, logger,
                                                           type=args.cema_type)
                refer_action_labels = deepcopy(action_labels_for_cost)

            # replace by correct labels and actions
            if len(sent_trn) > 0:
                sent_trn_true_label_list = []
                for sent in sent_trn:
                    sent_trn_true_label_list.append([label for token, label in sent])
                sent_trn_true_action_labels = get_action_labels(None, sent_trn, pred_labels[:len(sent_trn)], args, logger, type="ground_true")
                pred_labels[:len(sent_trn)] = sent_trn_true_label_list
                action_labels_for_cost[:len(sent_trn)] = sent_trn_true_action_labels

            assert refer_pred_labels is not None and refer_action_labels is not None

            # 将sent_pool中的数据只留下包含特征的样本， idx_mapping是将一个样本在sents_partial_no_empty中的标签映射为在同一样本在sent_pool中的标签。
            # idx_mapping: partial => original
            idx_mapping, sents_partial_no_empty, pred_labels_partial_no_empty, action_labels_partial_no_empty, refer_action_labels_partial_no_empty = CEMAPartialStrategy.partial_with_ety(
                                                                                all_sents[len(sent_trn):],
                                                                                pred_labels[len(sent_trn):],
                                                                                action_labels_for_cost[len(sent_trn):],
                                                                                args.ner_end_symbol,
                                                                                refer_pred_labels=refer_pred_labels[len(sent_trn):],
                                                                                refer_action_labels=refer_action_labels[len(sent_trn):])

            all_sents_partial_no_empty = all_sents[:len(sent_trn)] + sents_partial_no_empty
            all_pred_labels_partial_no_empty = pred_labels[:len(sent_trn)] + pred_labels_partial_no_empty
            all_action_labels_partial_no_empty = action_labels_for_cost[:len(sent_trn)] + action_labels_partial_no_empty
            all_refer_action_labels_partial_no_empty = refer_action_labels[:len(sent_trn)] + refer_action_labels_partial_no_empty

            entities = CEMA_Doc_Selctor.get_entities(all_pred_labels_partial_no_empty, actions = all_refer_action_labels_partial_no_empty, length = 5)
            logger.info('start to get entity_ebd_dict')
            entity_ebd_dict = tagger.get_entity_embedding(all_sents_partial_no_empty, entities, step_size=entity_embedding_step_size)
            print("No. of entities is {}.".format(len(entity_ebd_dict)))
            logger.info('end of getting entity_ebd_dict')

            if len(entity_ebd_dict)==0:
                print("In CEMA: all predicted feature labels are 'O'")

            cema_selector = CEMA_Doc_Selctor(train_index, all_action_labels_partial_no_empty, args.at_time_para, entity_ebd_dict,
                                                                entity_sim_threshold = args.at_entity_sim_threshold,
                                                                influence_weight = args.at_influence_weight,
                                                                batch_size = sim_mat_mul_batch_size,
                                                                log_preprocessed=False)

            cema_doc_num = sum(np.random.rand(doc_num) > args.rnd_ratio)
            top_k_indices = []
            if cema_doc_num > 0:
                top_k_indices = cema_selector.cema_sampling(cema_doc_num, stop_at_zero=args.at_stop_at_zero, logger=logger,
                                                            influence_can_zero=args.cema_influ_can_zero)

            n_train = len(sent_trn)
            n_cema = len(top_k_indices)
            # for consistency,
            # (1) idx-n_train align all index in all_action_labels_partial_no_empty with that in sents_partial_no_empty.
            # (2) idx_mapping[idx-n_train] align all index in sents_partial_no_empty with that in sent_pool
            top_k_indices = [idx_mapping[idx-n_train] for idx in top_k_indices]

            if n_cema < doc_num:
                logger.info(" * step-{} cema could not find enough instances".format(step))
                if args.at_unseen_entity == "random":
                    logger.info(" * step-{} random is used to find unseen entities".format(step))
                    supplement_indices = get_rnd_topk(sent_pool, doc_num)
                elif args.at_unseen_entity == "mnlp":
                    logger.info(" * step-{} mnlp is used to find unseen entities".format(step))
                    prob_dict = model.get_probability(sent_pool, obtain=["mnlp"])
                    norm_log_prob = prob_dict["mnlp"]
                    supplement_indices = get_least_topk(norm_log_prob, doc_num)
                else:
                    logger.info(" * step-{} random is used to find unseen entities".format(step))
                    supplement_indices = get_rnd_topk(sent_pool, doc_num)

                supplement_indices = [idx for idx in supplement_indices if idx not in top_k_indices]
                top_k_indices = top_k_indices + supplement_indices[:doc_num-len(top_k_indices)]
        else:
            print(at_query_strategy)
            raise  NotImplementedError()

        if "cema_partial" in at_query_strategy:
            for ith in range(n_cema):
                # inverse_idx_mapping map the index in sent_pool with that in sents_partial_no_empty.
                inverse_idx_mapping = {idx_mapping[key]:key for key in idx_mapping}
                # align the top_k_indices[ith] with that in sents_partial_no_empty.
                theindex = inverse_idx_mapping[top_k_indices[ith]]
                sent_trn.append(sents_partial_no_empty[theindex])
                step_results["selected_sents"].append(sents_partial_no_empty[theindex])
                step_results["orignial_sent"].append(sent_pool[top_k_indices[ith]])
            for theindex in top_k_indices[n_cema:]:
                sent_trn.append(sent_pool[theindex])
                step_results["selected_sents"].append(sent_pool[theindex])
                step_results["orignial_sent"].append(sent_pool[theindex])
            train_pred_labels = model.predict(sent_trn)
        else: # cema and random
            for theindex in top_k_indices:
                sent_trn.append(sent_pool[theindex])
                step_results["selected_sents"].append(sent_pool[theindex])
                step_results["orignial_sent"].append(sent_pool[theindex])
            train_pred_labels = model.predict(sent_trn)


        step_results["selected_pred_labels"] = deepcopy(train_pred_labels[-len(top_k_indices):])
        if action_predictor is not None:
            if args.cema_type in ["SelfReviewNER"]:
                _, top_k_preds_actions = action_predictor.predict_start_from_tokens(step_results["selected_sents"], iterations=0,
                                                                              input_pred_labels=step_results["selected_pred_labels"],
                                                                              adjust_action=True)
            else:
                top_k_sents_4ap = TFAPDataset.get_tf_ap_input(step_results["selected_sents"], step_results["selected_pred_labels"],
                                                              sim_threshold=args.at_sim_threshold, sim_method=args.at_sim_method)
                top_k_preds_actions = action_predictor.predict(top_k_sents_4ap)

            step_results["selected_preds_actions"] = top_k_preds_actions

        if args.cema_type in ["use action predictor"]:
            model.train(sent_trn, logger = logger, save_model=True)
            if action_predictor is not None:
                action_predictor = update_action_predictor(sent_trn, tagger, action_predictor, args.ner_end_symbol, args.at_sim_threshold,
                                        args.at_sim_method, logger, save_model=True, required_bioes=True)

        # delete the selected data point from the pool
        new_sent_pool = []
        for idx in range(len(sent_pool)):
            if idx not in top_k_indices:
                new_sent_pool.append(sent_pool[idx])
        sent_pool = new_sent_pool

        step += 1

        if step >= args.at_annotation_budget or len(new_sent_pool) < args.doc_num_a_step:
            break

        if (step) % args.interval == 0 or step <= 15:
            test_sent_pred_labels = model.predict(test_sents)
            step_results["test_sent_pred_labels"] = test_sent_pred_labels
            test_sent_pred_labels = [CostPerformaceEval.formatBIOES(seq, is_merge=args.at_is_merge) for seq in test_sent_pred_labels]
            test_preds_actions = None
            if action_predictor is not None:
                if args.cema_type in ["SelfReviewNER"]:
                    _, test_preds_actions = action_predictor.predict_start_from_tokens(test_sents,
                                                                                        iterations=0,
                                                                                        input_pred_labels=test_sent_pred_labels,
                                                                                        adjust_action=True)
                else:
                    test_preds_actions = get_action_labels(action_predictor, test_sents, test_sent_pred_labels, args, logger, type=args.cema_type)
            step_results["test_preds_actions"] = test_preds_actions

            f1_score = model.test(test_sents)
            logger.info('[Sampling phase] the no. of samples selected so far: {}'.format(str(step)))
            logger.info(' * Labeled data size: {}'.format(str(len(sent_trn))))
            logger.info(' * Unlabeled data size: {}'.format(str(len(sent_pool))))
            logger.info(" [Step {}] F1 score : {}".format(str(step), str(f1_score)))

        step_results["step"] = step
        results.append(step_results)
    return model, step, sent_trn, sent_pool, results


def get_initial_tagger(train_la, tagger_output, label2idx, test_sents, args, logger, save_model=True):
    logger.info("load initial tagger")
    if not os.path.exists(os.path.join(args.ner_init_path, "tokenizer.pkl")):
        tagger = TransformerNER(label2idx, learning_rate=args.ner_lr, epochs=args.ner_init_epochs,
                                            batch_size=args.ner_batch_size, device=args.ner_device,
                                            output_folder=args.ner_init_path, base_model_name=args.ner_base_model_name,
                                            end_symbol=args.ner_end_symbol,
                                            max_length=args.ner_max_length)
        tagger.train(train_la, logger=logger, save_model=save_model)
        tagger.save_model()
        test(tagger, None, test_sents, args, logger)

    my_copyfiles(args.ner_init_path, tagger_output)
    tagger = TransformerNER(label2idx, learning_rate=args.ner_lr, epochs=args.ner_epochs,
                                        batch_size=args.ner_batch_size, device=args.ner_device,
                                        output_folder=tagger_output, base_model_name=args.ner_base_model_name,
                                        end_symbol=args.ner_end_symbol, max_length=args.ner_max_length)
    test(tagger, None, test_sents, args, logger)
    return tagger

def get_initial_action_predictor(train_sents, tagger, action_predictor_output, args, logger, label2idx = None, save_model = True):

    logger.info("load initial action predictor")
    if not os.path.exists(os.path.join(args.ap_init_path, "tokenizer.pkl")):
        action_predictor = TransformerActionPredictor(list(args.unique_label_list), learning_rate = args.ap_lr,
                                                      epochs=args.ap_init_epochs, batch_size=args.ap_batch_size,
                                                      device=args.ap_device, output_folder=args.ap_init_path,
                                                      base_model_name=args.ap_base_model_name,
                                                      end_symbol=args.ap_end_symbol, max_length=args.ap_max_length)

        action_predictor = update_action_predictor(train_sents, tagger, action_predictor,
                                args.ner_end_symbol, args.at_sim_threshold, args.at_sim_method,
                                logger, save_model=save_model, required_bioes=True)

        action_predictor.save_model()

    my_copyfiles(args.ap_init_path, action_predictor_output)
    action_predictor = TransformerActionPredictor(list(args.unique_label_list), learning_rate=args.ap_lr,
                                                  epochs=args.ap_epochs, batch_size=args.ap_batch_size,
                                                  device=args.ap_device, output_folder=action_predictor_output,
                                                  base_model_name=args.ap_base_model_name,
                                                  end_symbol=args.ap_end_symbol, max_length=args.ap_max_length)

    return action_predictor

def test(tagger, action_predictor, test_sents, args, logger):
    tagger_f1_score = tagger.test(test_sents)
    logger.info("tagger - token level F1 : {}".format(str(tagger_f1_score)))
    action_predictor_f1_score = None
    if action_predictor is not None:
        test_pred = tagger.predict(test_sents)
        if not hasattr(args, "cema_type") or "SelfReviewNER" not in args.cema_type:
            test_sents_4ap = TFAPDataset.get_tf_ap_input(test_sents, test_pred,
                                                          sim_threshold=args.at_sim_threshold,
                                                          sim_method=args.at_sim_method)
            action_predictor_f1_score = action_predictor.test(test_sents_4ap)
        else:
            _, action_predictor_f1_score = action_predictor.test(test_sents, input_pred_labels=test_pred, iterations=0,
                           sim_threshold=args.at_sim_threshold, sim_method=args.at_sim_method, logger=logger)

        logger.info("action predictor - token level F1 : {}".format(str(action_predictor_f1_score)))
    return tagger_f1_score, action_predictor_f1_score

def get_sents_for_action_predictor(train_sents, tagger, ner_end_symbol, at_sim_threshold, at_sim_method, required_bioes=True):
    seg_train_sents, _, _, _ = to_segmented(train_sents, end_symbol=ner_end_symbol, initial_num=0, required_bioes=required_bioes)

    labels_set = set(tagger.label2idx.keys()) - {"O"}
    labels_set = {label.split("-")[-1] for label in labels_set} - {"O"}

    seg_train_preds = tagger.predict(seg_train_sents)
    train_sents_4ap, train_sents_4ap_true_labels, seg_train_sents, seg_train_preds = TFAPDataset.get_ap_input(
        seg_train_sents, seg_train_preds,
        labels_set, ner_end_symbol,
        sim_threshold=at_sim_threshold,
        sim_method=at_sim_method)

    return train_sents_4ap

def update_action_predictor(train_sents, tagger, action_predictor, ner_end_symbol, at_sim_threshold, at_sim_method, logger, save_model=True, required_bioes=True):
    original_epoches = action_predictor.epochs
    action_predictor.epochs = 1
    for _ in range(original_epoches):
        train_sents_4ap = get_sents_for_action_predictor(train_sents, tagger, ner_end_symbol, at_sim_threshold, at_sim_method, required_bioes = required_bioes)
        action_predictor.train(train_sents_4ap, logger=logger, save_model=save_model)
    action_predictor.epochs = original_epoches

    return action_predictor
