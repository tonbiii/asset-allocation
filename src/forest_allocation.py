import pickle
from decimal import Decimal

import numpy as np

from train_constants import TRAIN_COLUMNS



class RandomForestAllocation:
    def __init__(self):
        self._columns = list(TRAIN_COLUMNS)
        self._columns.remove('apy')

        # Attempt to load old_model.pkl with error handling
        try:
            with open('old_model.pkl', 'rb') as f:
                self._old_model = pickle.load(f, encoding='latin1')
        except Exception as e:
            print(f"Error loading old_model.pkl: {e}")
            self._old_model = None  # Handle the case where the model can't be loaded

        # Attempt to load model.pkl with error handling
        try:
            with open('model.pkl', 'rb') as f:
                self._model = pickle.load(f, encoding='latin1')
        except Exception as e:
            print(f"Error loading model.pkl: {e}")
            self._model = None  # Handle the case where the model can't be loaded

        # Attempt to load scaler.pkl with error handling
        try:
            with open('scaler.pkl', 'rb') as f:
                self._scaler = pickle.load(f, encoding='latin1')
        except Exception as e:
            print(f"Error loading scaler.pkl: {e}")
            self._scaler = None  # Handle the case where the scaler can't be loaded

        # Set n_jobs to 1 if models are loaded successfully
        if self._old_model:
            self._old_model.n_jobs = 1
        if self._model:
            self._model.n_jobs = 1

    def predict_allocation(self, assets_and_pools, model='new', index=100000000):
        total_assets = Decimal(assets_and_pools['total_assets'])

        batch = []
        for pool in assets_and_pools['pools'].values():
            data = [pool[column] for column in self._columns]
            batch.append(data)

        batch = np.array(batch)

        if self._scaler:
            batch = self._scaler.transform(batch)
        else:
            print("Scaler is not loaded. Cannot transform the batch.")
            return None  # Return None if the scaler is not loaded

        if model == 'new' and self._model:
            y = self._model.predict(batch)
        elif self._old_model:
            y = self._old_model.predict(batch)
        else:
            print("Model is not loaded. Cannot make predictions.")
            return None  # Return None if the model is not loaded

        y = [Decimal(alc) for alc in y.tolist()]
        sum_y = Decimal(sum(y))
        y = [self.round_down(alc / sum_y, index) * total_assets for alc in y]
        predicted_allocated = {str(i): float(v) for i, v in enumerate(y)}
        return predicted_allocated

    def round_down(self, value, index=10000000000000000):
        return ((Decimal(str(index)) * value) // Decimal('1')) / Decimal(str(index))
