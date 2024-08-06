import copy
from decimal import Decimal

import torch
import numpy as np
import torch
from torch import optim

from src.module import Model, projection_simplex_sort


class SGDAllocation:
    def __init__(self, epoch=15, lr=1e-3, num_cpu=3, device='cpu'):
        self.epoch = epoch
        self.lr = lr
        self._device = device
        torch.set_num_threads(num_cpu)
        torch.set_num_interop_threads(num_cpu)

    def convert_pool_to_tensor(self, assets_and_pools):
        columns = ['base_rate', 'base_slope', 'borrow_amount', 'kink_slope', 'optimal_util_rate', 'reserve_size']
        data = []
        for id, pool in assets_and_pools['pools'].items():
            data.append([pool[column] for column in columns])
        pools = torch.tensor(data, device=self._device, dtype=torch.float32)
        return pools

    def _maximize_apy_allocations(self, assets_and_pools, init_allocations):
        init_allocations = np.array(list(init_allocations.values()), dtype=np.float32)

        total_assets = torch.tensor(assets_and_pools['total_assets'], device=self._device)
        pools = self.convert_pool_to_tensor(assets_and_pools)

        model = Model(init_allocations)
        model = model.to(self._device)
        optimizer = optim.SGD(params=model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

        with torch.no_grad():
            model.allocations.copy_(projection_simplex_sort(model.allocations))

        for epoch in range(self.epoch):
            optimizer.zero_grad()

            apy = model(pools, total_assets)
            apy = -apy
            apy.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                model.allocations.copy_(projection_simplex_sort(model.allocations))

        return model.allocations

    def predict_allocation(self, assets_and_pools, initial_allocations=None):
        allocations = self._maximize_apy_allocations(copy.deepcopy(assets_and_pools), initial_allocations)

        total_assets = Decimal(assets_and_pools['total_assets'])

        allocations = [Decimal(float(allocation)) for allocation in allocations.data.cpu().numpy()]
        sum_allocs = Decimal(sum(allocations))
        allocations = [self.round_down(allocation / sum_allocs) * total_assets for allocation in allocations]
        normalized_allocations = {str(k): float(v) for k, v in enumerate(allocations)}
        return normalized_allocations

    def round_down(self, value, index=100000000):
        return ((Decimal(str(index)) * value) // Decimal('1')) / Decimal(str(index))
