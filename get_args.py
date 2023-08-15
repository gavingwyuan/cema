import argparse

def my_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--at_repetition', type=list, default=[1],
                        help='the no. of times to repeat the experiment')
    parser.add_argument('--result_root', default='results',
                        help='a folder for results')
    parser.add_argument('--at_dataset_name', default='german_doc',
                        help='it is used to rename output folder')
    parser.add_argument('--labels_map_file', default='dataset/german_doc/label2idx_dict',
                        help='')
    parser.add_argument('--train_file', default='dataset/german_doc/train.txt',
                        help='')
    parser.add_argument('--val_file', default='dataset/german_doc/val.txt',
                        help='')
    parser.add_argument('--test_file', default='dataset/german_doc/test.txt',
                        help='')

    parser.add_argument('--at_query_strategy', default='cema_partial_normal',
                        help='data point query strategy: Random, Uncertain, Diversity')
    parser.add_argument('--at_annotation_budget', type=int, default=30,
                        help='the no. of document selection steps will be run')
    parser.add_argument('--at_initial_training_size', type=int, default=2,
                        help='Number of data point to initialize underlying model')
    parser.add_argument('--interval', type=int, default=3,
                        help='Each interval step, model is evaluated.')
    parser.add_argument('--doc_num_a_step', type=int, default=2,
                        help='number of instances are selected in each step')
    parser.add_argument('--at_steps_in_iteration', type=int, default=5,
                        help='number of training steps in an iteration')

    parser.add_argument('--at_sim_threshold', default=5,
                        help='threshold for similarity relationship of spans. 0 means exact match if sim_method is not_intersection.')
    parser.add_argument('--at_sim_method', default="not_intersection",
                        help='the similarity of span is judged the number of tokens are not intersected')
    parser.add_argument('--at_influence_weight', type=float, default=0.5,
                        help='score in cema = (1-at_influence_weight) * normalized cost + at_influence_weight * normalized influence')
    parser.add_argument('--at_entity_sim_threshold', type=float, default=0.8,
                        help='threshold for entity similarity. The similarity is used in influence estimation.')

    # some fixed values
    parser.add_argument('--at_unseen_entity', default='mnlp',
                        help='strategy for unseen entity')
    parser.add_argument('--rnd_ratio', default=0,
                        help='In each iteration, the ratio of instances will be picked randomly.')
    parser.add_argument('--cema_influ_can_zero', default=True,
                        help='fixed value, cema_influ_can_zero = True, the instance with 0 influence will not be filtered.')
    parser.add_argument('--at_stop_at_zero', default=False,
                        help='at_stop_at_zero = False: if all instances have 0 influence, they will be pick randomly.')
    parser.add_argument('--cema_type', default="use action predictor",
                        help='fixed value, cema_type = use action predictor')
    parser.add_argument('--at_is_merge', default=True,
                        help='True: merge consecutive identical features into an annotation. Otherwise, nothing to do with the predicted labels')
    parser.add_argument('--at_vocab_size', type=int, default=20000,
                        help='max vocab size used to build vocabulary and word to idx in diversity selection')
    parser.add_argument('--at_bioes_scheme', default=True,
                        help='fixed value, at_bioes_scheme=True, the format of annotations')


    parser.add_argument('--ner_max_length', type=int, default=198,
                        help='max sequence length')
    parser.add_argument('--ner_batch_size', type=int, default=8,
                        help='the number of instances in a batch')
    parser.add_argument('--ner_lr', type=float, default=2 / (10 ** 5),
                        help='learning rate')
    parser.add_argument('--ner_epochs', type=int, default=5,
                        help='the no. of epochs for training')
    parser.add_argument('--ner_init_epochs', type=int, default=50,
                        help='the no. of epochs for initial training')
    parser.add_argument('--ner_device', type=str, default="cuda:0",
                        help='the device to run model')
    parser.add_argument('--ner_end_symbol', type=list, default=["\n"],
                        help='end_symbol to split document')
    parser.add_argument('--ner_base_model_name', type=str, default="dbmdz/bert-base-german-cased",
                        help='ner_base_model_name: bert or dbmdz/bert-base-german-cased')
    parser.add_argument('--ner_init_path', default=None,
                        help='the folder to load init model')

    parser.add_argument('--ap_max_length', type=int, default=198,
                        help='max sequence length')
    parser.add_argument('--ap_batch_size', type=int, default=3,
                        help='the number of instances in a batch')
    parser.add_argument('--ap_lr', type=float, default=2 / (10 ** 5),
                        help='learning rate')
    parser.add_argument('--ap_epochs', type=int, default=5,
                        help='the no. of epochs for training')
    parser.add_argument('--ap_init_epochs', type=int, default=50,
                        help='the no. of epochs for initial training')
    parser.add_argument('--ap_device', type=str, default="cuda:0",
                        help='the device to run model')
    parser.add_argument('--ap_end_symbol', type=list, default=["\n"],
                        help='end_symbol to split document')
    parser.add_argument('--ap_base_model_name', type=str, default="dbmdz/bert-base-german-cased",
                        help='ap_base_model_name: bert or dbmdz/bert-base-german-cased')
    parser.add_argument('--ap_init_path', default=None,
                        help='the folder to load init model')
    parser.add_argument('--ap_cls_model_name', type=str, default="linear",
                        help='BERT + [classifier]. The classifier can be liner and a transformer.')
    parser.add_argument('--ap_transformer_nhids', default=None,
                        help='no used. dimension of hidden state in transformer')

    return parser.parse_args()


def adjust_args(args):

    args.at_time_para = {'confirm': 1, 'revise': 4, 'delete': 1.5, 'add': 10, 'read': 0.3}

    args.result_root = "results/{}/maa_dn{}_itv_{}_init_ep50_5_bs8".format(args.at_dataset_name, args.doc_num_a_step,
                                                                        args.interval)

    if "cema" in args.at_query_strategy:
        args.at_output = "{}/{}_w{}_tr_{}_budget_{}_sim_{}".format(
                                                                                           args.result_root,
                                                                                           args.at_query_strategy,
                                                                                           args.at_influence_weight,
                                                                                           args.at_time_para["read"],
                                                                                           args.at_annotation_budget,
                                                                                           args.at_entity_sim_threshold)
    else:
        args.at_output = "{}/{}_{}".format(args.result_root, args.at_query_strategy, args.at_annotation_budget)

    args.ner_init_path = "{}/ner_init_model/{}_{}_tagger_initn{}_initep{}".format(args.result_root,
                                                                                  args.at_dataset_name,
                                                                                  args.ner_base_model_name,
                                                                                  args.at_initial_training_size,
                                                                                  args.ner_init_epochs)

    args.ap_init_path = "{}/ap_init_model/{}_{}_action_predictor_initn{}_initep{}".format(args.result_root,
                                                                                          args.at_dataset_name,
                                                                                          args.ap_base_model_name,
                                                                                          args.at_initial_training_size,
                                                                                          args.ap_init_epochs)

    return args

