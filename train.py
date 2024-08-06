import pickle
from _decimal import Decimal

import tqdm
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from algo import naive_algorithm
from forward import query_and_score
from pools import generate_assets_and_pools
from train_constants import TRAIN_COLUMNS


def prepare_data(data_size):
    """
    Preparing training data
    :param data_size: size of training data. Recommended: 1000-3000
    :return: pd.DataFrame
    """
    data = {column: list() for column in TRAIN_COLUMNS}

    for _ in tqdm.tqdm(range(data_size)):
        assets_and_pools = generate_assets_and_pools()
        convert_assets_and_pools = convert_pool(assets_and_pools, e=1)
        pools = convert_assets_and_pools['pools']
        pool = pools['0']
        for k, v in pool.items():
            if k in data:
                data[k].append(v)

        total_asset = convert_assets_and_pools['total_assets']
        # simple_allocations = {str(k): total_asset / len(pools) for k, v in pools.items()}
        simple_allocations = naive_algorithm(assets_and_pools)

        apys, max_apy = query_and_score([simple_allocations], assets_and_pools)
        data['apy'].append(max_apy)
    return pd.DataFrame(data)


def train(df):
    """
    Train random forest model.
    :param df:
    :return:
    """
    # Preparing data. Ground truth y is apy
    y = df['apy'].astype(float)
    X = df.drop(columns=['apy'])
    X = X.astype(float)

    # Splitting data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creating and training a random forest regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train.values, y_train)

    importances = model.feature_importances_

    # Create a DataFrame with the feature names and their importances
    feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

    print(feature_importances)

    # save model to file
    with open('model.pkl.bk', 'wb') as f:
        pickle.dump(model, f)


def convert_allocation(allocation):
    final_allocation = {str(k): float(Decimal(str(float(v) * 1e18))) for k, v in allocation.items()}
    return final_allocation


def convert_pool(asset_and_pools, e=1e18):
    new_pools = {'0': {}, '1': {}, '2': {}, '3': {}, '4': {}, '5': {}, '6': {}, '7': {}, '8': {}, '9': {}}
    new_asset_and_pools = {'total_assets': asset_and_pools['total_assets'] / e, 'pools': new_pools}
    for x, pools in asset_and_pools['pools'].items():
        new_asset_and_pools['pools'][x]['base_slope'] = pools.base_slope / e
        new_asset_and_pools['pools'][x]['kink_slope'] = pools.kink_slope / e
        new_asset_and_pools['pools'][x]['optimal_util_rate'] = pools.optimal_util_rate / e
        new_asset_and_pools['pools'][x]['borrow_amount'] = pools.borrow_amount / e
        new_asset_and_pools['pools'][x]['reserve_size'] = pools.reserve_size / e
        new_asset_and_pools['pools'][x]['base_rate'] = pools.base_rate / e
    return new_asset_and_pools


if __name__ == '__main__':
    df = prepare_data(data_size=10_000)
    train(df)
