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
        self.id_to_itemid, self.id_to_userid,\
            self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)

        # Словарь {item_id: 0/1}. 0/1 - факт принадлежности товара к СТМ
        # self.item_id_is_ctm = dict(zip(data.item_id.values,
        #                                (data.brand == 'Private').astype(int).values))

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

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())

        return model

    def _update_dict(self, user_id):
        """Если появился новыю user / item, то нужно обновить словари"""

        if user_id not in self.userid_to_id.keys():

            max_id = max(list(self.userid_to_id.values()))
            max_id += 1

            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})

    def _extend_with_top_popular(self, recommendations, N=5):
        """Если кол-во рекоммендаций < N, то дополняем их топ-популярными"""

        if len(recommendations) < N:
            recommendations.extend(self.overall_top_purchases[:N])
            recommendations = recommendations[:N]

        return recommendations

    def _get_recommendations(self, user, model, N=5):
        """Рекомендации через стардартные библиотеки implicit"""

        self._update_dict(user_id=user)
        res = [self.id_to_itemid[rec[0]] for rec in model.recommend(userid=self.userid_to_id[user],
                                        user_items=csr_matrix(self.user_item_matrix).tocsr(),
                                        N=N,
                                        filter_already_liked_items=False,
                                        # filter_items=[self.itemid_to_id[999999]],
                                        recalculate_user=True)]

        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_als_recommendations(self, user, N=5):
        """Рекомендации через стардартные библиотеки implicit"""

        return self._get_recommendations(user, model=self.model, N=N)

    def get_own_recommendations(self, user, N=5):
        """Рекомендуем товары среди тех, которые юзер уже купил"""

        return self._get_recommendations(user, model=self.own_recommender, N=N)

    def get_similar_items_recommendation(self, user,
                                         filter_ctm=False, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        self.top_user_purchases = self.top_purchases[self.top_purchases['user_id'] == user]

        if filter_ctm:
            self.top_user_purchases['ctm'] = self.top_user_purchases['item_id'].map(self.item_is_to_ctm)
            self.top_user_purchases = self.top_user_purchases[self.top_user_purchases['ctm'] == 0]

        self.top_user_purchases = self.top_user_purchases.head(N)

        if filter_ctm:
            res = self.top_user_purchases['item_id'].\
                       apply(lambda x: self._get_rec_ctm(self.model, x)).tolist()
        else:
            res = self.top_user_purchases['item_id'].\
                       apply(lambda x: self._get_rec(self.model, x)).tolist()

        res = self._extend_with_top_popular(res, N=N)

        return res

    def _get_rec(self, model, x):
        recs = model.similar_items(self.itemid_to_id[x], N=2)
        top_rec = recs[1][0]
        return self.id_to_itemid[top_rec]

    def _get_rec_ctm(self, model, x):
        recs = model.similar_items(self.itemid_to_id[x], N=20)
        recs = [self.id_to_itemid[x[0]] for x in recs]
        recs_ctm = [x for x in recs if self.item_id_is_ctm[x]]
        return recs_ctm[0]

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        similar_users = self.model.similar_users(self.userid_to_id[user], 6)
        similar_users = [self.id_to_userid[x[0]] for x in similar_users][1:]

        recs_own = self.own_recommender.recommend(userid=self.userid_to_id[user],
                        user_items=csr_matrix(self.user_item_matrix).tocsr(),
                        N=N,
                        filter_already_liked_items=False,
                        filter_items=None,
                        recalculate_user=False)
        recs_own = [self.id_to_itemid[x[0]] for x in recs_own]

        top_purchases_similar_users = self.top_purchases[(self.top_purchases['user_id'].isin(similar_users))
                                                         & (~self.top_purchases['item_id'].isin(recs_own))]
        res = top_purchases_similar_users['item_id'].head(N).tolist()

        return res
