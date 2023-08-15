import time
import pickle as pkl
from utils.utils import init_logger
from get_args import *
from experiment_util import *
from data_processed.split_data import *

args = my_args()
args = adjust_args(args)

# args.ner_init_epochs = 2
# args.ap_init_epochs = 2
# args.ner_epochs = 2
# args.ap_epochs = 2

if not os.path.exists(args.at_output):
    os.makedirs(args.at_output)

if not os.path.exists(args.ner_init_path):
    os.makedirs(args.ner_init_path)

if not os.path.exists(args.ap_init_path):
    os.makedirs(args.ap_init_path)

log_path = "{}/{}_AL_{}_dn_{}_nr_{}_log.txt".format(args.at_output,
                                                    args.at_dataset_name,
                                                    args.at_query_strategy,
                                                    args.at_annotation_budget,
                                                    args.at_steps_in_iteration)
logger = init_logger(log_path)

logger.info(" * Input data: {}".format(args.at_dataset_name))
logger.info(" * Log OUTPUT file: {}".format(log_path))

if args.at_bioes_scheme:
    logger.info("Processing data - BIOES scheme")
    dataset, label2idx, vocab_processor = utilities.load_data2labels_BIOES(args, args.at_vocab_size)
    train_x, train_y, train_lens = dataset[0]
    dev_x, dev_y, dev_lens = dataset[1]
    test_x, test_y, test_lens = dataset[2]
    args.unique_label_list = list(label2idx.keys())
else:
    raise NotImplementedError()

logger.info(args)

# train_x, test_x = train_x[:10], test_x[:10]
# train_y, test_y = train_y[:10], test_y[:10]
# train_lens, test_lens = train_lens[:10], test_lens[:10]

k_folds_datas = k_folds_split4test((train_x + test_x), (train_y + test_y), (train_lens + test_lens),
                                  args.at_initial_training_size, n_splits=5)

start_time = time.time()
for r in args.at_repetition:

    train_x, train_y, train_lens, test_x, test_y, test_lens, train_sents, test_sents = k_folds_datas[r-1]

    all_results = []
    logger.info('Repetition:' + str(r))

    tagger_output = "{}/{}_tagger_r_{}_qs_{}".format(args.at_output, args.ner_base_model_name, r, args.at_query_strategy)
    action_predictor_output = "{}/{}_action_predictor_r_{}_qs_{}".format(args.at_output, args.ap_base_model_name, r, args.at_query_strategy)
    intermediate_results = tagger_output + ".intermediate_results.pkl"

    logger.info("[Repetition {}] Load tagger from {}".format(str(r), args.ner_init_path))
    logger.info(" * TAGGER OUTPUT file: {}".format(tagger_output))
    logger.info(" * INTERMEDIATE RESULTS OUTPUT file: {}".format(intermediate_results))

    step_results = {"step": None, "selected_sents": [], "selected_pred_labels": [], "selected_preds_actions": [],
                         "test_sents": None, "test_sent_pred_labels": None, "orignial_sent": []}

    logger.info("results format: {}".format(step_results))

    if ("cema" in args.at_query_strategy):
        logger.info("Load action predictor from {}".format(str(r), args.ap_init_path))
        logger.info(" * Action predictor OUTPUT path: {}".format(action_predictor_output))

    logger.info("Training size: {}".format(len(train_sents)))

    train_la = []
    train_pool = []

    for i in range(0, len(train_sents)):
        if i < args.at_initial_training_size:
            train_la.append(train_sents[i])
        else:
            train_pool.append(train_sents[i])

    indices = np.random.RandomState(r).permutation(len(train_pool))
    train_pool = [train_pool[idx] for idx in indices]

    logger.info(' * Start Machine assisted annotations.')
    step = 0

    # get initial Tagger
    tagger = get_initial_tagger(train_la, tagger_output, label2idx, test_sents, args, logger)

    step_results["step"] = 0
    train_la_pred = tagger.predict(train_la)
    step_results["selected_sents"] = deepcopy(train_la)
    step_results["orignial_sent"] = deepcopy(train_la)
    step_results["selected_pred_labels"] = deepcopy(train_la_pred)
    step_results["selected_preds_actions"] = None

    # train initial Action predictor
    action_predictor = None
    if ("cema" in args.at_query_strategy):
        action_predictor = get_initial_action_predictor(train_la, tagger, action_predictor_output, args, logger)

    test_y_pred = tagger.predict(test_sents)
    step_results["test_sents"] = deepcopy(test_sents)
    step_results["test_sent_pred_labels"] = deepcopy(test_y_pred)
    _, _ = test(tagger, action_predictor, test_sents, args, logger)

    all_results.append(step_results)
    while step < args.at_annotation_budget:
        logger.info(' * Begin sampling..')
        tagger, step, train_la, train_pool, results = get_next_k_instance(train_la, train_pool, tagger,
                                                                        step, args, logger, test_sents,
                                                                        tagger=tagger,
                                                                        action_predictor=action_predictor)
        all_results.extend(results)
        logger.info("One iteration time cost:--- {} seconds ---".format(time.time() - start_time))

        # write results
        with open(intermediate_results, "wb") as file:
            pkl.dump(all_results, file)
    logger.info(" * INTERMEDIATE RESULTS OUTPUT file: {}".format(intermediate_results))

logger.info(log_path)



