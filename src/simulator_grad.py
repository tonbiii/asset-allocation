import torch
from torch import Tensor


@torch.jit.script
def borrow_rate(util_rate, pools: torch.Tensor) -> Tensor:
    interest_rate = torch.where(
        util_rate < pools[:, 4],
        pools[:, 0] + (util_rate / pools[:, 4]) * pools[:, 1],
        pools[:, 0] + pools[:, 1] + ((util_rate - pools[:, 4]) / (1 - pools[:, 4])) * pools[:, 3])
    return interest_rate


@torch.jit.script
def supply_rate(util_rate, pool: torch.Tensor) -> torch.Tensor:
    return util_rate * borrow_rate(util_rate, pool)


@torch.jit.script
def init_data(init_pools: torch.Tensor) -> torch.Tensor:
    pool_history = torch.stack(tensors=[
        init_pools[:, 2],
        init_pools[:, 5],
        borrow_rate(init_pools[:, 2] / init_pools[:, 5], init_pools)
    ], dim=1)
    return pool_history


@torch.jit.script
def update_reserves_with_allocs(allocations, pools, pool_history):
    reserve_size = pool_history[:, 1] + allocations
    pool_history = torch.stack(tensors=[
        pool_history[:, 0],
        reserve_size,
        borrow_rate(pool_history[:, 0] / reserve_size, pools)
    ], dim=1)
    return pool_history


@torch.jit.script
def calc_apy(allocations, pools, pool_history):
    pools_yield = allocations * supply_rate(pool_history[:, 0] / pool_history[:, 1], pools)
    pool_yield = torch.sum(pools_yield)
    return pool_yield


@torch.jit.script
def generate_new_pool_data(pools, pool_history):
    curr_borrow_amounts = pool_history[:, 0]
    curr_reserve_sizes = pool_history[:, 1]
    curr_borrow_rates = pool_history[:, 2]

    median_rate = torch.median(curr_borrow_rates)  # Calculate the median borrow rate
    rate_changes = -0.1 * (curr_borrow_rates - median_rate)  # Mean reversion principle
    new_borrow_amounts = curr_borrow_amounts + rate_changes * curr_borrow_amounts  # Update the borrow amounts
    amounts = torch.clip(new_borrow_amounts, min=torch.zeros_like(curr_reserve_sizes),
                         max=curr_reserve_sizes)  # Ensure borrow amounts do not exceed reserves
    pool_history = torch.stack([
        amounts,
        curr_reserve_sizes,
        borrow_rate(amounts / curr_reserve_sizes, pools)
    ], dim=1)
    return pool_history


@torch.jit.script
def run(allocations, pools, initial_balance):
    timesteps = 1
    pct_yield = 0

    pool_history = init_data(pools)
    pool_history = update_reserves_with_allocs(allocations, pools, pool_history)
    pct_yield += calc_apy(allocations, pools, pool_history)
    for ts in range(1, timesteps):
        generate_new_pool_data(pools, pool_history)
        pct_yield += calc_apy(allocations, pools, pool_history)
    pct_yield /= initial_balance
    aggregate_apy = (pct_yield / timesteps) * 365
    return aggregate_apy
