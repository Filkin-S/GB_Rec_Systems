'''Functions for data processing'''

import numpy as np


def prefilter_items(data, take_n_popular=5000):
    '''Предфильтрация товаров

    1. Удаление товаров со средней ценой <= 1$
    2. Удаление товаров со средней ценой > 30$
    '''
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    mean_prices = data.groupby('item_id')['price'].mean().reset_index()
    normal_price_items = mean_prices.loc[(mean_prices['price'] > 1)
                                         & (mean_prices['price'] <= 30)
                                         ].item_id.tolist()

    data = data[data['item_id'].isin(normal_price_items)]

    # 3.Удаление товаров, которые не продавались последние 6 месяцев (24 нед.)
    item_last_sale = data.groupby('item_id')['week_no'].max().reset_index()
    freq_sold_items = item_last_sale[data['week_no'].max()
                                     - item_last_sale['week_no'] < 24
                                     ].item_id.tolist()

    data = data[data['item_id'].isin(freq_sold_items)]

    # 4. Выбор топ-N самых популярных товаров (N = take_n_popular)
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
    popularity.sort_values('n_sold', ascending=False)
    top_N = popularity.head(take_n_popular).item_id.tolist()

    data = data[data['item_id'].isin(top_N)]

    return data
