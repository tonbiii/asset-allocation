# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Syeam Bin Abdullah
import sys

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import bittensor as bt
import numpy as np
import torch
from typing import List, Dict, Tuple, Any, Union
import copy

import web3

from constants import QUERY_TIMEOUT, SIMILARITY_THRESHOLD
from pools import POOL_TYPES, BasePoolModel, ChainBasedPoolModel
from ethmath import wei_div, wei_mul
from misc import check_allocations
from protocol import REQUEST_TYPES, AllocInfo


def get_response_times(uids: List[int], responses, timeout: float) -> Dict[str, int]:
    """
    Returns a list of axons based on their response times.

    This function pairs each uid with its corresponding axon's response time.
    Lower response times are considered better.

    Args:
        uids (List[int]): List of unique identifiers for each axon.
        responses (List[Response]): List of Response objects corresponding to each axon.

    Returns:
        List[Tuple[int, float]]: A sorted list of tuples, where each tuple contains an axon's uid and its response time.

    Example:
        >>> get_sorted_response_times([1, 2, 3], [response1, response2, response3])
        [(2, 0.1), (1, 0.2), (3, 0.3)]
    """
    axon_times = {
        str(uids[idx]): (
            response.dendrite.process_time
            if response.dendrite.process_time is not None
            else timeout
        )
        for idx, response in enumerate(responses)
    }
    # Sorting in ascending order since lower process time is better
    return axon_times


def format_allocations(
        allocations: Dict[str, int],
        assets_and_pools: Dict[
            str, Union[Dict[str, Union[ChainBasedPoolModel, BasePoolModel]], int]
        ],
):
    # TODO: better way to do this?
    if allocations is None:
        allocations = {}
    allocs = allocations.copy()
    pools = assets_and_pools["pools"]

    # pad the allocations
    for pool_id in pools.keys():
        if pool_id not in allocs:
            allocs[pool_id] = 0

    # sort the allocations by pool id
    formatted_allocs = {pool_id: allocs[pool_id] for pool_id in sorted(allocs.keys())}

    return formatted_allocs


def reward_miner_apy(
        query: int,
        max_apy: int,
        miner_apy: int,
) -> int:
    # Calculate the adjusted APY reward
    if max_apy <= 0:
        return 0

    return (miner_apy) / (max_apy)


def calculate_penalties(
        similarity_matrix: Dict[str, Dict[str, int]],
        axon_times: Dict[str, float],
        similarity_threshold: float = SIMILARITY_THRESHOLD,
):
    penalties = {miner: 0 for miner in similarity_matrix}

    for miner_a, similarities in similarity_matrix.items():
        for miner_b, similarity in similarities.items():
            if similarity <= similarity_threshold:
                if axon_times[miner_a] <= axon_times[miner_b]:
                    penalties[miner_b] += 1

    return penalties


def calculate_rewards_with_adjusted_penalties(miners, rewards_apy, penalties):
    rewards = torch.zeros(len(miners))
    max_penalty = max(penalties.values())
    if max_penalty == 0:
        return rewards_apy

    for idx, miner_id in enumerate(miners):
        # Calculate penalty adjustment
        penalty_factor = (max_penalty - penalties[miner_id]) / max_penalty

        # Calculate the final reward
        reward = rewards_apy[idx] * penalty_factor
        rewards[idx] = reward

    return rewards


def get_similarity_matrix(
        apys_and_allocations: Dict[str, Dict[str, Union[Dict[str, int], int]]],
        assets_and_pools: Dict[str, Union[Dict[str, int], int]],
):
    """
    Calculates the similarity matrix for the allocation strategies of miners using normalized Euclidean distance.

    This function computes a similarity matrix based on the Euclidean distance between the allocation vectors of miners,
    normalized by the maximum possible distance in the given asset space. Each miner's allocation is compared with every
    other miner's allocation, resulting in a matrix where each element (i, j) represents the normalized Euclidean distance
    between the allocations of miner_i and miner_j.

    The similarity metric is scaled between 0 and 1, where 0 indicates identical allocations and 1 indicates the maximum
    possible distance between the allocation 'vectors'.

    Args:
        apys_and_allocations (Dict[str, Dict[str, Union[Dict[str, int], int]]]):
            A dictionary containing the APY and allocation strategies for each miner. The keys are miner identifiers,
            and the values are dictionaries with their respective allocations and APYs.
        assets_and_pools (Dict[str, Union[Dict[str, int], int]]):
            A dictionary representing the assets available to the miner as well as the pools they can allocate to

    Returns:
        Dict[str, Dict[str, float]]:
            A nested dictionary where each key is a miner identifier, and the value is another dictionary containing the
            normalized Euclidean distances to every other miner. The distances are scaled between 0 and 1.
    """

    similarity_matrix = {}
    total_assets = assets_and_pools["total_assets"]
    for miner_a, info_a in apys_and_allocations.items():
        _alloc_a = info_a["allocations"]
        alloc_a = np.array(
            list(format_allocations(_alloc_a, assets_and_pools).values()),
            dtype=np.float32,
        )
        similarity_matrix[miner_a] = {}
        for miner_b, info_b in apys_and_allocations.items():
            if miner_a != miner_b:
                _alloc_b = info_b["allocations"]
                if _alloc_a is None or _alloc_b is None:
                    similarity_matrix[miner_a][miner_b] = float("inf")
                    continue
                alloc_b = np.array(
                    list(format_allocations(_alloc_b, assets_and_pools).values()),
                    dtype=np.float32,
                )
                similarity_matrix[miner_a][miner_b] = np.linalg.norm(
                    alloc_a - alloc_b
                ) / np.sqrt(float(2 * total_assets ** 2))

    return similarity_matrix


def adjust_rewards_for_plagiarism(
        rewards_apy: torch.FloatTensor,
        apys_and_allocations: Dict[str, Dict[str, Union[Dict[str, int], int]]],
        assets_and_pools: Dict[
            str, Union[Dict[str, Union[ChainBasedPoolModel, BasePoolModel]], int]
        ],
        uids: List,
        axon_times: Dict[str, float],
        similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> torch.FloatTensor:
    """
    Adjusts the annual percentage yield (APY) rewards for miners based on the similarity of their allocations
    to others and their arrival times, penalizing plagiarized or overly similar strategies.

    This function calculates the similarity between each pair of miners' allocation strategies and applies a penalty
    to those whose allocations are too similar to others, considering the order in which they arrived. Miners who
    arrived earlier with unique strategies are given preference, and those with similar strategies arriving later
    are penalized. The final APY rewards are adjusted accordingly.

    Args:
        rewards_apy (torch.FloatTensor): The initial APY rewards for the miners, before adjustments.
        apys_and_allocations (Dict[str, Dict[str, Union[Dict[str, int], int]]]):
            A dictionary containing APY values and allocation strategies for each miner. The keys are miner identifiers,
            and the values are dictionaries that include their allocations and APYs.
        assets_and_pools (Dict[str, Union[Dict[str, int], int]]):
            A dictionary representing the available assets and their corresponding pools.
        uids (List): A list of unique identifiers for the miners.
        axon_times (Dict[str, float]): A dictionary that tracks the arrival times of each miner, with the keys being
            miner identifiers and the values being their arrival times. Earlier times are lower values.

    Returns:
        torch.FloatTensor: The adjusted APY rewards for the miners, accounting for penalties due to similarity with
        other miners' strategies and their arrival times.
    Notes:
        - This function relies on the helper functions `calculate_penalties` and `calculate_rewards_with_adjusted_penalties`
          which are defined separately.
        - The `format_allocations` function used in the similarity calculation converts the allocation dictionaries
          to a consistent format suitable for comparison.
    """
    # Step 1: Calculate pairwise similarity (e.g., using Euclidean distance)
    similarity_matrix = get_similarity_matrix(apys_and_allocations, assets_and_pools)

    # Step 2: Apply penalties considering axon times
    penalties = calculate_penalties(similarity_matrix, axon_times, similarity_threshold)

    # Step 3: Calculate final rewards with adjusted penalties
    rewards = calculate_rewards_with_adjusted_penalties(uids, rewards_apy, penalties)

    return rewards


def _get_rewards(
        self,
        query: int,
        max_apy: float,
        apys_and_allocations: Dict[str, Dict[str, Union[Dict[str, int], int]]],
        assets_and_pools: Dict[
            str, Union[Dict[str, Union[ChainBasedPoolModel, BasePoolModel]], int]
        ],
        uids: List[int],
        axon_times: List[float],
) -> float:
    """
    Rewards miner responses to request. This method returns a reward
    value for the miner, which is used to update the miner's score.

    Returns:
    - adjusted_rewards: The reward values for the miners.
    """

    rewards_apy = torch.FloatTensor(
        [
            reward_miner_apy(
                query,
                max_apy=max_apy,
                miner_apy=apys_and_allocations[uid]["apy"],
            )
            for uid in uids
        ]
    ).to(self.device)

    adjusted_rewards = adjust_rewards_for_plagiarism(
        rewards_apy, apys_and_allocations, assets_and_pools, uids, axon_times
    )

    return adjusted_rewards


def calculate_apy(
        self,
        allocations: Dict[str, int],
        assets_and_pools: Dict[
            str, Union[Dict[str, Union[ChainBasedPoolModel, BasePoolModel]], int]
        ],
):
    """
    Calculates immediate projected yields given intial assets and pools, pool history, and number of timesteps
    """

    # calculate projected yield
    initial_balance = assets_and_pools["total_assets"]
    pools = assets_and_pools["pools"]
    pct_yield = 0
    for uid, pool in pools.items():
        allocation = allocations[uid]
        match pool.pool_type:
            case POOL_TYPES.STURDY_SILO:
                pool_yield = wei_mul(
                    allocation, pool.supply_rate(amount=allocation)
                )
            case _:
                pool_yield = wei_mul(
                    allocation, pool.supply_rate(user_addr=pool.user_address, amount=allocation)
                )
        pct_yield += pool_yield
    pct_yield = wei_div(pct_yield, initial_balance)

    return pct_yield


def calculate_aggregate_apy(
        allocations: Dict[str, int],
        assets_and_pools: Dict[
            str, Union[Dict[str, Union[ChainBasedPoolModel, BasePoolModel]], int]
        ],
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
            pool_yield = wei_mul(allocs, pool_data.supply_rate)
            curr_yield += pool_yield
        pct_yield += curr_yield

    pct_yield = wei_div(pct_yield, initial_balance)
    aggregate_apy = int(
        pct_yield // timesteps
    )  # for simplicity each timestep is a day in the simulator

    return aggregate_apy


def get_rewards(simulator, allocations_list: List, assets_and_pools):
    # maximum yield to scale all rewards by
    # total apys of allocations per miner
    max_apy = sys.float_info.min
    apys = []

    init_assets_and_pools = copy.deepcopy(assets_and_pools)

    bt.logging.debug(
        f"Running simulator for {simulator.timesteps} timesteps for each allocation..."
    )

    for alloc_ind, allocations in enumerate(allocations_list):
        # reset simulator for next run
        simulator.reset()

        # validator miner allocations before running simulation
        # is the miner cheating w.r.t allocations?
        cheating = True
        try:
            cheating = not check_allocations(init_assets_and_pools, allocations)
        except Exception as e:
            bt.logging.error(e)

        # score response very low if miner is cheating somehow or returns allocations with incorrect format
        if cheating:
            bt.logging.warning(
                f"CHEATER DETECTED - PUNISHING 👊😠"
            )
            apys.append(sys.float_info.min)
            continue

        try:
            # miner does not appear to be cheating - so we init simulator data
            simulator.init_data(
                init_assets_and_pools=copy.deepcopy(init_assets_and_pools),
                init_allocations=allocations,
            )
            simulator.update_reserves_with_allocs()

            simulator.run()

            resulting_apy = calculate_aggregate_apy(
                allocations,
                init_assets_and_pools,
                simulator.timesteps,
                simulator.pool_history,
            )
            apys.append(resulting_apy)
        except Exception as e:
            bt.logging.error(e)
            bt.logging.error("Failed to calculate apy - PENALIZING MINER")
            apys.append(sys.float_info.min)
            import traceback
            traceback.print_exc()
            continue

        if resulting_apy > max_apy:
            max_apy = resulting_apy

    return apys, max_apy
