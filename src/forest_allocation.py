import pickle
from decimal import Decimal

import numpy as np

from train_constants import TRAIN_COLUMNS


class RandomForestAllocation:
    def __init__(self):
        self._columns = list(TRAIN_COLUMNS)
        self._columns.remove('apy')
        with open('old_model.pkl', 'rb') as f:
            self._old_model = pickle.load(f)

        with open('model.pkl', 'rb') as f:
            self._model = pickle.load(f)

        with open('scaler.pkl', 'rb') as f:
            self._scaler = pickle.load(f)

        self._old_model.n_jobs = 1
        self._model.n_jobs = 1

    def predict_allocation(self, assets_and_pools, model='new', index=100000000):
        total_assets = Decimal(assets_and_pools['total_assets'])

        batch = []
        for pool in assets_and_pools['pools'].values():
            data = [pool[column] for column in self._columns]
            batch.append(data)

        batch = np.array(batch)

        batch = self._scaler.transform(batch)
        if model == 'new':
            y = self._model.predict(batch)
        else:
            y = self._old_model.predict(batch)

        y = [Decimal(alc) for alc in y.tolist()]
        sum_y = Decimal(sum(y))
        y = [self.round_down(alc / sum_y, index) * total_assets for alc in y]
        predicted_allocated = {str(i): float(v) for i, v in enumerate(y)}
        return predicted_allocated

    def round_down(self, value, index=10000000000000000):
        return ((Decimal(str(index)) * value) // Decimal('1')) / Decimal(str(index))
