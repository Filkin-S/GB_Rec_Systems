'''Metrics for recomendation system'''

import numpy as np


def hit_rate_at_k(recommended_list, bought_list, k=5):
    '''
    Hit rate@k = (был ли хотя бы 1 релевантный товар
    среди топ-k рекомендованных)
    '''

    flags = np.isin(bought_list, recommended_list[:k])

    hit_rate = (flags.sum() > 0) * 1

    return hit_rate


def precision_at_k(recommended_list, bought_list, k=5):
    '''
    Precision - доля релевантных товаров среди рекомендованных =
    Какой % рекомендованных товаров юзер купил

    Precision@k = (# of recommended items @k that are relevant) /
                  (# of recommended items @k)
    '''

    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    bought_list = bought_list
    recommended_list = recommended_list[:k]

    flags = np.isin(bought_list, recommended_list)

    precision = flags.sum() / len(recommended_list)

    return precision


def money_precision_at_k(recommended_list, bought_list,
                         prices_recommended, k=5):
    '''
    Precision - доля релевантных товаров среди рекомендованных =
    Какой % рекомендованных товаров юзер купил

    Money Precision@k = (revenue of recommended items @k that are relevant) /
                        (revenue of recommended items @k)
    '''

    flags = np.isin(recommended_list[:k], bought_list)

    precision = np.sum(
                np.multiply(flags, prices_recommended[:k])
                ) / np.sum(prices_recommended[:k])

    return precision


def recall_at_k(recommended_list, bought_list, k=5):
    '''
    Recall - доля рекомендованных товаров среди релевантных =
    Какой % купленных товаров был среди рекомендованных

    Recall@k = (# of recommended items @k that are relevant) /
                (# of relevant items)
    '''
    flags = np.isin(bought_list, recommended_list[:k])

    recall = flags.sum() / len(bought_list)

    return recall


def money_recall_at_k(recommended_list, bought_list,
                      prices_recommended, prices_bought, k=5):
    '''
    Recall - доля рекомендованных товаров среди релевантных =
    Какой % купленных товаров был среди рекомендованных

    Money Recall@k = (revenue of recommended items @k that are relevant) /
                    (revenue of relevant items)
    '''
    flags = np.isin(recommended_list[:k], bought_list)

    recall = np.sum(
                np.multiply(flags, prices_recommended[:k])
                ) / np.sum(prices_bought)

    return recall
