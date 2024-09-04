import random
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from time import time
from prettytable import PrettyTable
import logging
from utils.parser import parse_args
from utils.data_loader import load_data
from modules.RAFRCE import Recommender
from utils.evaluate import test
# from utils.helper import early_stopping

import logging
n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0

logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)
def get_feed_dict(train_entity_pairs, start, end):
    train_entity_pairs = torch.LongTensor(np.array([[cf[0], cf[1], cf[2]] for cf in train_entity_pairs], np.int32))
    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['items'] = entity_pairs[:, 1]
    feed_dict['labels'] = entity_pairs[:, 2]
    return feed_dict

def get_feed_dict_topk(train_entity_pairs, start, end):
    train_entity_pairs = torch.LongTensor(np.array([[cf[0], cf[1], cf[2]] for cf in train_entity_pairs], np.int32))
    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['items'] = entity_pairs[:, 1]
    feed_dict['labels'] = entity_pairs[:, 2]
    return feed_dict
    # def negative_sampling(user_item, train_user_set):
    #     neg_items = []
    #     for user, _ in user_item.cpu().numpy():
    #         user = int(user)
    #         while True:
    #             neg_item = np.random.randint(low=0, high=n_items, size=1)[0]
    #             if neg_item not in train_user_set[user]:
    #                 break
    #         neg_items.append(neg_item)
    #     return neg_items
    #
    # feed_dict = {}
    # entity_pairs = train_entity_pairs[start:end].to(device)
    # feed_dict['users'] = entity_pairs[:, 0]
    # feed_dict['pos_items'] = entity_pairs[:, 1]
    # feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs,
    #                                                             train_user_set)).to(device)

def _show_recall_info(recall_zip):
    res = ""
    for i, j in recall_zip:
        res += "K@%d:%.4f  "%(i,j)
    logging.info(res)

def _get_topk_feed_data(user, items):
    res = list()
    for item in items:
        res.append([user, item, 0])
    return np.array(res)

def _get_user_record(data, is_train):
    user_history_dict = dict()
    for rating in data:
        user = rating[0]
        item = rating[1]
        label = rating[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict

def ctr_eval(model, data):
    auc_list = []
    f1_list = []
    model.eval()
    start = 0
    while start < data.shape[0]:

        batch = get_feed_dict(data, start, start + args.batch_size)
        labels = data[start:start + args.batch_size, 2]
        _, scores, _, _ = model(batch)
        scores = scores.detach().cpu().numpy()
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        f1 = f1_score(y_true=labels, y_pred=predictions)
        auc_list.append(auc)
        f1_list.append(f1)
        start += args.batch_size
    model.train()
    auc = float(np.mean(auc_list))
    f1 = float(np.mean(f1_list))
    return auc, f1

def topk_eval(model, train_data, data):
    # logging.info('calculating recall ...')
    k_list = [5, 10, 20, 50, 100]#k_list = [5, 10, 20, 50, 100]
    recall_list = {k: [] for k in k_list}
    item_set = set(train_data[:, 1].tolist() + data[:, 1].tolist())
    train_record = _get_user_record(train_data, True)
    test_record = _get_user_record(data, False)
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    user_num = 100
    if len(user_list) > user_num:
        np.random.seed()
        user_list = np.random.choice(user_list, size=user_num, replace=False)

    model.eval()
    for user in user_list:
        test_item_list = list(item_set-set(train_record[user]))
        item_score_map = dict()
        start = 0
        while start + args.batch_size <= len(test_item_list):
            items = test_item_list[start:start + args.batch_size]
            input_data = _get_topk_feed_data(user, items)
            batch = get_feed_dict_topk(input_data, start, start + args.batch_size)
            _, scores, _, _ = model(batch)
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += args.batch_size
        if start < len(test_item_list):
            res_items = test_item_list[start:] + [test_item_list[-1]] * (args.batch_size - len(test_item_list) + start)
            input_data = _get_topk_feed_data(user, res_items)
            batch = get_feed_dict_topk(input_data, start, start + args.batch_size)
            _, scores, _, _ = model(batch)
            for item, score in zip(res_items, scores):
                item_score_map[item] = score
        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]
        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & set(test_record[user]))
            recall_list[k].append(hit_num / len(set(test_record[user])))
    model.train()

    recall = [np.mean(recall_list[k]) for k in k_list]
    return recall
if __name__ == '__main__':
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    global args, device
    args = parse_args()
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

    train_cf, test_cf, user_dict, n_params, graph, mat_list = load_data(args)
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']


    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1], cf[2]] for cf in test_cf], np.int32))

    model = Recommender(n_params, args, graph, mean_mat_list[0]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_result_auc = 0
    best_result_f1 = 0
    stopping_step = 0
    should_stop = False
    early_stop_cnt = 10
    print("start training ...")
    for epoch in range(args.epoch):

        index = np.arange(len(train_cf))
        np.random.shuffle(index)
        train_cf = train_cf[index]

        loss, s, cor_loss = 0, 0, 0

        while s + args.batch_size <= len(train_cf):
            batch = get_feed_dict(train_cf, s, s + args.batch_size)
            batch_loss, _, _, _ = model(batch)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss

            s += args.batch_size

        if 1:
            """testing"""
            # test_s_t = time()
            # ret = test(model, user_dict, n_params)
            test_auc, test_f1 = ctr_eval(model, test_cf_pairs)
            # recall = topk_eval(model, train_cf, test_cf)
            # # print(recall)
            # topk_info = 'epoch %.2d  loss: %f   recall@5: %.4f  recall@10: %.4f  recall@20: %.4f  recall@50: %.4f  recall@100: %.4f'
            # logging.info(topk_info, epoch, loss, recall[0], recall[1], recall[2], recall[3], recall[4])
            # test_e_t = time()
            ctr_info = 'epoch %.2d  loss: %f   test auc: %.4f f1: %.4f'
            logging.info(ctr_info, epoch, loss, test_auc, test_f1)
            if test_f1 > best_result_f1:
                best_result_f1 = test_f1
            if test_auc > best_result_auc:
                stopping_step = 0
                best_result_auc = test_auc
                logging.info('find a better model')
                # torch.save(Recmodel.state_dict(), weight_file)
            else:
                stopping_step += 1
                if stopping_step >= early_stop_cnt:
                    print(f"early stop triggerd at epoch {epoch}")
                    break

    print('early stopping at %d, test_auc_best:%.4f, test_f1_best:%.4f' % (epoch, best_result_auc, best_result_f1))
    logging.info('best_result_auc:%.4f  best_result_f1:%.4f', best_result_auc, best_result_f1)




