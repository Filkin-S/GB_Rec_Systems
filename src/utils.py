'''Functions for data processing'''

import numpy as np
import pandas as pd


def prefilter_items(data, take_n_popular=5000, item_features=None):
    '''Предфильтрация товаров'''
    # Уберем самые популярные товары (их и так купят)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

    top_popular = popularity[popularity['share_unique_users'] > 0.6].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]

    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.01].item_id.tolist()
    data = data[~data['item_id'].isin(top_notpopular)]

    # Уберем не интересные для рекоммендаций категории (department)
    if item_features is not None:
        department_size = pd.DataFrame(item_features.\
                                       groupby('department')['item_id'].nunique().\
                                       sort_values(ascending=False)).reset_index()

        department_size.columns = ['department', 'n_items']
        rare_departments = department_size[department_size['n_items'] < 150].department.tolist()
        items_in_rare_departments = item_features[item_features['department'].isin(rare_departments)].item_id.unique().tolist()

        data = data[~data['item_id'].isin(items_in_rare_departments)]

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    data = data[data['price'] > 1]

    # Уберем слишком дорогие товары
    data = data[data['price'] < 45]

    # Удаление товаров, которые не продавались последние 4 месяца (16 нед.)
    item_last_sale = data.groupby('item_id')['week_no'].max().reset_index()
    freq_sold_items = item_last_sale[(data['week_no'].max()
                                     - item_last_sale['week_no']) < 16
                                     ].item_id.tolist()

    data = data[data['item_id'].isin(freq_sold_items)]

    # Возьмем топ по популярности
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()

    # Заведем фиктивный item_id (если юзер покупал товары не из топ-5000, то он "купил" такой товар)
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999
    
    return data


def postfilter_items(user_id, recs, item_features, own_purchases,
                     top_costly_items, top_purchases, items_price):
    """Пост-фильтрация товаров
        Input
    -----
    user_id:
        id юзера которому фильтруем товары
    recs: list
        Ранжированный список item_id для рекомендаций
    item_features: pd.DataFrame
        Датафрейм с информацией о товарах
    own_purchases: pd.DataFrame
        Датафрейм с топом покупок каждого пользователя
    top_costly_items: list
        Список топ дорогих товаров
    top_purchases: list
        Список топ всех покупок всего датасета
    items_price: dict
        Словарь с ценами товаров
    """
    # Уникальность
    # recs = list(set(recs)) - неверно! так теряется порядок
    unique_recs = []
    [unique_recs.append(item) for item in recs if item not in unique_recs]

    # Проверка на разные категории
    categories_used = []
    CATEGORY_NAME = 'sub_commodity_desc'

    final_recs = []

    # Добавляем дорогой товар из рекомендаций юзера, если такого нет то из топа дорогих товаров
    unique_costly_items = [item for item in unique_recs if items_price[item] > 7]

    for item in unique_costly_items:
        category = item_features.loc[item_features['item_id'] == item, CATEGORY_NAME].values[0]
        final_recs.append(item)
        categories_used.append(category)

        if len(final_recs) == 1:
            break

    if not len(final_recs):
        for item in top_costly_items:
            category = item_features.loc[item_features['item_id'] == item, CATEGORY_NAME].values[0]
            final_recs.append(item)
            categories_used.append(category)

            if len(final_recs) == 1:
                break

    # Добавляем два товара из личного топа юзера
    user_top_list = own_purchases[own_purchases['user_id'] == user_id]['item_id'].tolist()

    for item in user_top_list:
        category = item_features.loc[item_features['item_id'] == item, CATEGORY_NAME].values[0]

        if (category not in categories_used
                and item not in final_recs):
            final_recs.append(item)
            categories_used.append(category)

        if len(final_recs) == 3:
            break

    # Добавляем два товара - снижает метрику
    # for item in unique_recs:
    #     category = item_features.loc[item_features['item_id'] == item, CATEGORY_NAME].values[0]

    #     if (category not in categories_used
    #             and item not in final_recs):
    #         final_recs.append(item)
    #         categories_used.append(category)

    #     if len(final_recs) == 3:
    #         break

    # Добавляем два новых для юзера товара
    for item in unique_recs:
        category = item_features.loc[item_features['item_id'] == item, CATEGORY_NAME].values[0]

        if (category not in categories_used
                and item not in user_top_list
                and item not in final_recs):
            final_recs.append(item)
            categories_used.append(category)

        if len(final_recs) == 5:
            break

    if len(final_recs) < 5:
        for item in top_purchases:
            category = item_features.loc[item_features['item_id'] == item, CATEGORY_NAME].values[0]

            if (category not in categories_used
                    and item not in final_recs):
                final_recs.append(item)
                categories_used.append(category)

            if len(final_recs) == 5:
                break

    return final_recs


def get_recommendations(data, old_users_list,
                        old_user_model, new_user_model, N):
    '''Рекомендации для новых и старых пользователей'''
    data_new_users = data[~data['user_id'].isin(old_users_list)].copy()
    data_new_users['pre_rec'] = data_new_users['user_id'].apply(lambda x: new_user_model[:N])

    data_old_users = data[data['user_id'].isin(old_users_list)].copy()
    data_old_users['pre_rec'] = data_old_users['user_id'].apply(lambda x: old_user_model(x, N=N))

    result = pd.concat([data_new_users, data_old_users])

    return result
