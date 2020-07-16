'''Recommendations from ALS recommender system'''

import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender
from implicit.nearest_neighbours import bm25_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data, weighting=True):

        # Топ покупок каждого юзера
        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != 999999]

        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != 999999]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()

        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid,
        self.itemid_to_id, self.userid_to_id = prepare_dicts(self.user_item_matrix)

        # Словарь {item_id: 0/1}. 0/1 - факт принадлежности товара к СТМ
        self.item_id_to_ctm = dict(zip(item_features.item_id.values,
                                       (item_features.brand == 'Private').astype(int).values))

        # Own recommender обучается до взвешивания матрицы
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.model = self.fit(self.user_item_matrix)

    @staticmethod
    def prepare_matrix(data):

        user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values='quantity',
                                          aggfunc='count',
                                          fill_value=0
                                          )

        user_item_matrix = user_item_matrix.astype(float)

        return user_item_matrix

    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(csr_matrix(self.user_item_matrix).T.tocsr())

        return model

    def get_similar_items_recommendation(self, user, filter_ctm=True, N=5):

        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        self.top_user_purchases = self.top_purchases[self.top_purchases['user_id'] == user]

        if filter_ctm:
            self.top_user_purchases['ctm'] = self.top_user_purchases['item_id'].map(item_id_to_ctm)
            self.top_user_purchases = self.top_user_purchases[self.top_user_purchases['ctm'] == 0]

        self.top_user_purchases = self.top_user_purchases.head(N)

        def get_rec(model, x):
            recs = model.similar_items(itemid_to_id[x], N=2)
            top_rec = recs[1][0]
            return id_to_itemid[top_rec]

        def get_rec_ctm(model, x):
            recs = model.similar_items(itemid_to_id[x], N=20)
            recs = [id_to_itemid[x[0]] for x in recs]
            recs_ctm = [x for x in recs if item_id_to_ctm[x]]
            return recs_ctm[0]

        if filter_ctm:
            self.top_purchases_similar['similar_recommendation'] = self.top_purchases_similar['item_id'].\
                                                                   apply(lambda x: get_rec_ctm(self.model, x))
        else:
            self.top_purchases_similar['similar_recommendation'] = self.top_purchases_similar['item_id'].\
                                                                   apply(lambda x: get_rec(self.model, x))

        res = self.top_user_purchases['similar_recommendation'].tolist()

        return res

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        similar_users = self.model.similar_users(userid_to_id[user], 6)
        similar_users = [id_to_userid[x[0]] for x in similar_users][1:]

        recs_own = self.own_recommender.recommend(userid=userid_to_id[user],
                        user_items=csr_matrix(self.user_item_matrix).tocsr(),
                        N=N,
                        filter_already_liked_items=False,
                        filter_items=None,
                        recalculate_user=False)
        recs_own = [id_to_itemid[x[0]] for x in recs_own]

        top_purchases_similar_users = self.top_purchases[(self.top_purchases['user_id'].isin(similar_users))
                                                         & (~self.top_purchases['item_id'].isin(recs_own))]
        res = top_purchases_similar_users['item_id'].head(N).tolist()

        return res
