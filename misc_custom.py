
import copy
import logging
import sys
import traceback
from decimal import Decimal
from typing import Dict, Any, Union

import numpy as np

from constants import CHUNK_RATIO, GREEDY_SIG_FIGS
from simulator import Simulator


# TODO: cleanup functions - lay them out better across files?


# rand range but float
def randrange_float(
        start,
        stop,
        step,
        sig: int = GREEDY_SIG_FIGS,
        max_prec: int = GREEDY_SIG_FIGS,
        rng_gen=np.random,
):
    num_steps = int((stop - start) / step)
    random_step = rng_gen.randint(0, num_steps + 1)
    return format_num_prec(start + random_step * step, sig=sig, max_prec=max_prec)



def format_num_prec(
        num: float, sig: int = GREEDY_SIG_FIGS, max_prec: int = GREEDY_SIG_FIGS
) -> float:
    return float(f"{{0:.{max_prec}f}}".format(float(format(num, f".{sig}f"))))


def borrow_rate(util_rate: float, pool: Dict) -> float:
    interest_rate = (
        pool["base_rate"] + (util_rate / pool["optimal_util_rate"]) * pool["base_slope"]
        if util_rate < pool["optimal_util_rate"]
        else pool["base_rate"]
             + pool["base_slope"]
             + ((util_rate - pool["optimal_util_rate"]) / (1 - pool["optimal_util_rate"]))
             * pool["kink_slope"]
    )

    return interest_rate


def supply_rate(util_rate: float, pool: Dict) -> float:
    return util_rate * borrow_rate(util_rate, pool)


def check_allocations(
        assets_and_pools: Dict[str, Union[Dict[str, float], float]],
        allocations: Dict[str, float],
) -> bool:
    """
    Checks allocations from miner.

    Args:
    - assets_and_pools (Dict[str, Union[Dict[str, float], float]]): The assets and pools which the allocations are for.
    - allocations (Dict[str, float]): The allocations to validate.

    Returns:
    - bool: Represents if allocations are valid.
    """

    # Ensure the allocations are provided and valid
    if not allocations or not isinstance(allocations, Dict):
        return False

    # Ensure the 'total_assets' key exists in assets_and_pools and is a valid number
    to_allocate = assets_and_pools.get("total_assets")
    if to_allocate is None or not isinstance(to_allocate, (int, float)):
        return False

    to_allocate = Decimal(str(to_allocate))
    total_allocated = Decimal(0)

    # Check allocations
    for _, allocation in allocations.items():
        try:
            allocation_value = Decimal(str(allocation))
        except (ValueError, TypeError):
            return False

        if allocation_value < 0:
            return False

        total_allocated += allocation_value

        if total_allocated > to_allocate:
            return False

    # Ensure total allocated does not exceed the total assets
    if total_allocated > to_allocate:
        return False

    return True


def f_value(m, n, x):
    return Decimal('-1') * m * x * x + (m + n) * x


def find_max_of_func(m, n, min_x, max_x):
    max_dict = {}
    mid = (m + n) / (Decimal('2') * m)
    if min_x <= mid <= max_x:
        val_mid_x = f_value(m, n, mid)
        max_dict[str(mid)] = val_mid_x

    val_min_x = f_value(m, n, min_x)
    max_dict[str(min_x)] = val_min_x

    val_max_x = f_value(m, n, max_x)
    max_dict[str(max_x)] = val_max_x

    sorted_items = sorted(max_dict.items(), key=lambda item: item[1], reverse=True)
    highest_value_item = sorted_items[0]
    return float(highest_value_item[0]), highest_value_item[1]


def find_max_of_2(x1, val1, x2, val2):
    if val1 >= val2:
        return x1 / (1 - x1), val1
    else:
        return x2 / (1 - x2), val2


def maximum_some_pool_allocation_algorithm(assets_and_pools, take_items):
    try:
        logging.info("start maximum_one_pool_allocation_algorithm")
        pools = assets_and_pools["pools"]
        total_assets = Decimal(str(assets_and_pools["total_assets"]))
        result_dict = {}
        current_allocations = {k: 0.0 for k, _ in pools.items()}
        for key, value in pools.items():
            a = Decimal(str(value['base_rate']))
            b = Decimal(str(value['base_slope']))
            c = Decimal(str(value['kink_slope']))
            d = Decimal(str(value['optimal_util_rate']))
            e = Decimal(str(value['borrow_amount']))
            # f = Decimal(str(value['reserve_size']))

            if e >= d:
                m1 = (b * e * e) / d
                n1 = a * e
                min_x_1 = (e - d) / e
                max_x_1 = Decimal(str(2 / 3))
                # logging.info(
                #     f"y1 = {-1 * m1}x^2 + {m1 + n1}x    {max_x_1} >= x >= {min_x_1}  (m+n)/2m = {(m1 + n1) / (2 * m1)}")
                max_1, max_val_1 = find_max_of_func(m=m1, n=n1, min_x=min_x_1, max_x=max_x_1)

                m2 = (c * e * e) / (1 - d)
                n2 = (a + b) * e - ((c * d * e) / (1 - d))
                min_x_2 = Decimal('0')
                max_x_2 = (e - d) / e
                # logging.info(
                #     f"y2 = {-1 * m2}x^2 + {m2 + n2}x     {min_x_2} <= x <= {max_x_2}  (m+n)/2m = {(m2 + n2) / (2 * m2)}")
                max_2, max_val_2 = find_max_of_func(m=m2, n=n2, min_x=min_x_2, max_x=max_x_2)

                result_x, result_val = find_max_of_2(x1=max_1, val1=max_val_1, x2=max_2, val2=max_val_2)

                # logging.info(f"result_x = {result_x}, value = {result_val}")

                result_dict[key] = [result_x, result_val]
            else:
                m = (b * e * e) / d
                n = a * e
                min_x = Decimal('0')
                max_x = Decimal(str(2 / 3))
                # logging.info(
                #     f"y3 = {-1 * m}x^2 + {m + n}x   {max_x} >= x >= {min_x}   (m+n)/2m = {(m + n) / (2 * m)}")
                temp, result_val = find_max_of_func(m=m, n=n, min_x=min_x, max_x=max_x)
                result_x = Decimal(str(temp)) / (Decimal('1') - Decimal(str(temp)))
                # logging.info(f"result_x = {result_x}, value = {result_val}")

                result_dict[key] = [result_x, result_val]
        sorted_items = sorted(result_dict.items(), key=lambda item: item[1][1], reverse=True)

        result_allocations = []
        for take_item in take_items:
            temp_allocations = copy.deepcopy(current_allocations)
            sub_sorted_items = sorted_items[:take_item]
            for item in sub_sorted_items:
                temp_allocations[item[0]] = total_assets / Decimal(str(take_item))
            result_allocations.append(temp_allocations)

        return result_allocations
    except Exception as e:
        logging.error(e)
        traceback.print_exc()


def simple_allocation_algorithm(assets_and_pools) -> Dict:
    logging.info("start simple_allocation_algorithm")
    pools = assets_and_pools["pools"]
    current_allocations = {k: v["borrow_amount"] for k, v in pools.items()}
    return current_allocations


def call_allocation_algorithm(assets_and_pools) -> Dict:
    allocations = []
    greedy_alloc = greedy_allocation_algorithm(assets_and_pools)
    max_some_pool_alloc = maximum_some_pool_allocation_algorithm(assets_and_pools, [5, 6, 7, 8])
    allocations.append(greedy_alloc)
    allocations.extend(convert_list_to_float(max_some_pool_alloc))
    logging.info(f"all allocations {allocations}")
    apys, max_index = get_apys(allocation_list=allocations,
                               assets_and_pools=assets_and_pools)
    logging.info(f"pre-calculate apys {apys}, max_index: {max_index}")
    return allocations[max_index]


def convert_list_to_float(list_allocation):
    result = []
    for allocation in list_allocation:
        result.append(convert_to_float(allocation))
    return result


def convert_to_float(allocation):
    result = {}
    for key, value in allocation.items():
        result[key] = float(value)
    return result


def greedy_allocation_algorithm(assets_and_pools) -> Dict:
    logging.info("start greedy_allocation_algorithm")
    max_balance = assets_and_pools["total_assets"]
    balance = max_balance
    pools = assets_and_pools["pools"]

    # how much of our assets we have allocated
    current_allocations = {k: 0.0 for k, _ in pools.items()}

    assert balance >= 0

    # run greedy algorithm to allocate assets to pools
    while balance > 0:
        # TODO: use np.float32 instead of format()??
        current_supply_rates = {
            k: format_num_prec(
                supply_rate(
                    util_rate=v["borrow_amount"]
                              / (current_allocations[k] + pools[k]["reserve_size"]),
                    pool=v,
                )
            )
            for k, v in pools.items()
        }

        default_chunk_size = format_num_prec(CHUNK_RATIO * max_balance)
        to_allocate = 0

        if balance < default_chunk_size:
            to_allocate = balance
        else:
            to_allocate = default_chunk_size

        balance = format_num_prec(balance - to_allocate)
        assert balance >= 0
        max_apy = max(current_supply_rates.values())
        min_apy = min(current_supply_rates.values())
        apy_range = format_num_prec(max_apy - min_apy)

        alloc_it = current_allocations.items()
        for pool_id, _ in alloc_it:
            delta = format_num_prec(
                to_allocate * ((current_supply_rates[pool_id] - min_apy) / (apy_range)),
            )
            current_allocations[pool_id] = format_num_prec(
                current_allocations[pool_id] + delta
            )
            to_allocate = format_num_prec(to_allocate - delta)

        assert to_allocate == 0  # should allocate everything from current chunk

    return current_allocations


def calculate_aggregate_apy(
        allocations: Dict[str, float],
        assets_and_pools: Dict[str, Union[Dict[str, float], float]],
        timesteps: int,
        pool_history: Dict[str, Dict[str, Any]],
):
    """
    Calculates aggregate yields given intial assets and pools, pool history, and number of timesteps
    """

    # calculate aggregate yield
    initial_balance = assets_and_pools["total_assets"]
    pct_yield = 0
    for pools in pool_history:
        curr_yield = 0
        for uid, allocs in allocations.items():
            pool_data = pools[uid]
            util_rate = pool_data["borrow_amount"] / pool_data["reserve_size"]
            pool_yield = allocs * supply_rate(util_rate, assets_and_pools["pools"][uid])
            curr_yield += pool_yield
        pct_yield += curr_yield

    pct_yield /= initial_balance
    aggregate_apy = (
                            pct_yield / timesteps
                    ) * 365  # for simplicity each timestep is a day in the simulator

    return aggregate_apy


def get_apys(allocation_list, assets_and_pools):
    simulator = Simulator()
    simulator.initialize()
    simulator.init_data(init_assets_and_pools=assets_and_pools)

    max_apy = sys.float_info.min
    max_apy_idx = -1
    apys = []

    init_assets_and_pools = copy.deepcopy(simulator.assets_and_pools)

    for alloc_index, allocations in enumerate(allocation_list):
        # reset simulator for next run
        simulator.reset()

        # validator miner allocations before running simulation
        # is the miner cheating w.r.t allocations?
        cheating = True
        try:
            cheating = not check_allocations(init_assets_and_pools, allocations)
        except Exception as e:
            logging.error(e)

        # score response very low if miner is cheating somehow or returns allocations with incorrect format
        if cheating:
            logging.warning(
                f"CHEATER DETECTED  - ALLOCATION WITH index {alloc_index} - PUNISHING 👊😠"
            )
            apys.append(sys.float_info.min)
            continue

        # miner does not appear to be cheating - so we init simulator data
        simulator.init_data(copy.deepcopy(init_assets_and_pools), allocations)

        # update reserves given allocations
        try:
            simulator.update_reserves_with_allocs()
        except Exception as e:
            logging.error(e)
            logging.error(
                "Failed to update reserves with miner allocations - PENALIZING ALLOCATION"
            )
            apys.append(sys.float_info.min)
            continue

        simulator.run()

        aggregate_apy = calculate_aggregate_apy(
            allocations,
            init_assets_and_pools,
            simulator.timesteps,
            simulator.pool_history,
        )

        if aggregate_apy > max_apy:
            max_apy = aggregate_apy
            max_apy_idx = len(apys)

        apys.append(aggregate_apy)

    return apys, max_apy_idx


