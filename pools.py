# The MIT License (MIT)
# Copyright © 2023 Syeam Bin Abdullah

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

import json
import math
from enum import Enum
from pathlib import Path
from typing import Dict, Literal, Union

import bittensor as bt
import numpy as np
import web3.constants
from pydantic import BaseModel, Field, PrivateAttr, model_validator, validator
from web3 import Web3 
from web3.contract import Contract
from web3.types import ChecksumAddress

from constants import *
from ethmath import wei_div, wei_mul
from misc import (
    randrange_float,
    format_num_prec,
    retry_with_backoff,
    rayMul,
    getReserveFactor,
    ttl_cache,
)

class POOL_TYPES(str, Enum):
    STURDY_SILO = "STURDY_SILO"
    AAVE = "AAVE"
    SYNTHETIC = "SYNTHETIC"


class BasePoolModel(BaseModel):
    pool_model_disc: Literal['SYNTHETIC'] = Field(default='SYNTHETIC', description="pool model discriminator")
    pool_id: str = Field(..., description="pool id")
    pool_type: POOL_TYPES = Field(..., description="type of pool")
    base_rate: float = Field(..., description="base rate")
    base_slope: float = Field(..., description="base slope")
    kink_slope: float = Field(..., description="kink slope")
    optimal_util_rate: float = Field(..., description="optimal utilization rate")
    borrow_amount: int = Field(..., description="borrow amount in wei")
    reserve_size: Union[int, str] = Field(..., description="pool reserve size in wei")

    @validator('reserve_size', pre=True, always=True)
    def parse_reserve_size(cls, value):
        if isinstance(value, str):
            try:
                value = int(value)
            except ValueError:
                raise ValueError(f"reserve_size must be an integer or a string representing an integer, but got {value}")
        elif not isinstance(value, int):
            raise ValueError(f"reserve_size must be an integer, but got {value}")
        return value


    @model_validator(mode='before')
    @classmethod
    def check_pool_id(cls, values):
        pool_id = values.get('pool_id')
        if pool_id is None or len(pool_id) <= 0:
            raise ValueError("pool id is empty")
        return values

    @model_validator(mode='before')
    @classmethod
    def check_non_negative(cls, values):
        fields_to_check = ['base_rate', 'base_slope', 'kink_slope', 'optimal_util_rate', 'borrow_amount', 'reserve_size']
        for field in fields_to_check:
            if values.get(field, 0) < 0:
                raise ValueError(f"{field.replace('_', ' ')} is negative")
        return values


class BasePool(BasePoolModel):
    """This class defines the base pool type

    Args:
        pool_id: (str),
        base_rate: (float),
        base_slope: (float),
        kink_slope: (float),
        optimal_util_rate: (float),
        borrow_amount: (float),
        reserve_size: (float),
    """

    @property
    def util_rate(self) -> int:
        return wei_div(self.borrow_amount, self.reserve_size)

    @property
    def borrow_rate(self) -> int:
        util_rate = self.util_rate
        interest_rate = (
            self.base_rate
            + wei_mul(wei_div(util_rate, self.optimal_util_rate), self.base_slope)
            if util_rate < self.optimal_util_rate
            else self.base_rate
                 + self.base_slope
                 + wei_mul(
                wei_div(
                    (util_rate - self.optimal_util_rate),
                    (1e18 - self.optimal_util_rate),
                ),
                self.kink_slope,
            )
        )

        return interest_rate

    @property
    def supply_rate(self):
        return wei_mul(self.util_rate, self.borrow_rate)


class ChainBasedPoolModel(BaseModel):
    """This serves as the base model of pools which need to pull data from on-chain

    Args:
        pool_id: (str),
        contract_address: (str),
    """

    pool_model_disc: Literal['CHAIN'] = Field(default='CHAIN', description="pool type discriminator")
    pool_id: str = Field(..., description="uid of pool")
    pool_type: str = Field(..., description="type of pool")
    user_address: str = Field(
        default=Web3.to_checksum_address("0x0000000000000000000000000000000000000000"),
        description="address of the 'user' - used for various on-chain calls",
    )
    contract_address: str = Field(
        default=Web3.to_checksum_address("0x0000000000000000000000000000000000000000"),
        description="address of contract to call"
    )

    _initted: bool = PrivateAttr(False)
    _user_address_checksum: ChecksumAddress = PrivateAttr(default=Web3.to_checksum_address("0x0000000000000000000000000000000000000000"))
    _contract_address_checksum: ChecksumAddress = PrivateAttr(default=Web3.to_checksum_address("0x0000000000000000000000000000000000000000"))

    @validator("user_address", "contract_address", pre=True, always=True)
    def validate_checksum_address(cls, value):
        if not Web3.isAddress(value):
            raise ValueError(f"Address {value} is not valid!")
        return Web3.to_checksum_address(value)

    @model_validator(mode='before')
    @classmethod
    def check_params(cls, values):
        if len(values.get("pool_id", "")) <= 0:
            raise ValueError("pool id is empty")
        if not Web3.isAddress(values.get("contract_address", "")):
            raise ValueError("pool address is invalid!")
        if not Web3.isAddress(values.get("user_address", "")):
            raise ValueError("user address is invalid!")
        return values

    def pool_init(self, **args):
        raise NotImplementedError("pool_init() has not been implemented!")

    def sync(self, **args):
        raise NotImplementedError("sync() has not been implemented!")

    def supply_rate(self, **args):
        raise NotImplementedError("supply_rate() has not been implemented!")


class PoolFactory:
    @staticmethod
    def create_pool(
            pool_type: POOL_TYPES, **kwargs
    ) -> Union[ChainBasedPoolModel, BasePoolModel]:
        match pool_type:
            case POOL_TYPES.SYNTHETIC:
                return BasePool(**kwargs)
            case POOL_TYPES.AAVE:
                return AaveV3DefaultInterestRatePool(**kwargs)
            case POOL_TYPES.STURDY_SILO:
                return VariableInterestSturdySiloStrategy(**kwargs)
            case _:
                raise ValueError(f"Unknown pool type: {pool_type}")


class AaveV3DefaultInterestRatePool(ChainBasedPoolModel):
    """This class defines the default pool type for Aave"""

    pool_type: Literal['AAVE'] = Field(
        default='AAVE', description="type of pool"
    )

    _atoken_contract: Contract = PrivateAttr()
    _pool_contract: Contract = PrivateAttr()
    _underlying_asset_contract: Contract = PrivateAttr()
    _underlying_asset_address: str = PrivateAttr()
    _reserve_data = PrivateAttr()
    _strategy_contract = PrivateAttr()
    _nextTotalStableDebt = PrivateAttr()
    _nextAvgStableBorrowRate = PrivateAttr()
    _variable_debt_token_contract = PrivateAttr()
    _totalVariableDebt = PrivateAttr()
    _reserveFactor = PrivateAttr()
    _decimals: int = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    def __hash__(self):
        return hash((self._atoken_contract.address, self._underlying_asset_address))

    def __eq__(self, other):
        if not isinstance(other, AaveV3DefaultInterestRatePool):
            return NotImplemented
        # Compare the attributes for equality
        return (self._atoken_contract.address, self._underlying_asset_address) == (
            other._atoken_contract.address,
            other._underlying_asset_address,
        )

    def pool_init(self, web3_provider: Web3):
        try:
            assert web3_provider.is_connected()
        except Exception as err:
            bt.logging.error("Failed to connect to Web3 instance!")
            bt.logging.error(err)

        try:
            atoken_abi_file_path = Path(__file__).parent / "../abi/AToken.json"
            atoken_abi_file = atoken_abi_file_path.open()
            atoken_abi = json.load(atoken_abi_file)
            atoken_abi_file.close()
            atoken_contract = web3_provider.eth.contract(
                abi=atoken_abi, decode_tuples=True
            )
            self._atoken_contract = retry_with_backoff(
                atoken_contract,
                address=self.contract_address,
            )

            pool_abi_file_path = Path(__file__).parent / "../abi/Pool.json"
            pool_abi_file = pool_abi_file_path.open()
            pool_abi = json.load(pool_abi_file)
            pool_abi_file.close()

            atoken_contract = self._atoken_contract
            pool_address = retry_with_backoff(atoken_contract.functions.POOL().call)

            pool_contract = web3_provider.eth.contract(abi=pool_abi, decode_tuples=True)
            self._pool_contract = retry_with_backoff(
                pool_contract, address=pool_address
            )

            self._underlying_asset_address = retry_with_backoff(
                self._atoken_contract.functions.UNDERLYING_ASSET_ADDRESS().call
            )

            erc20_abi_file_path = Path(__file__).parent / "../abi/IERC20.json"
            erc20_abi_file = erc20_abi_file_path.open()
            erc20_abi = json.load(erc20_abi_file)
            erc20_abi_file.close()

            underlying_asset_contract = web3_provider.eth.contract(
                abi=erc20_abi, decode_tuples=True
            )
            self._underlying_asset_contract = retry_with_backoff(
                underlying_asset_contract, address=self._underlying_asset_address
            )

            self._initted = True

        except Exception as err:
            bt.logging.error("Failed to load contract!")
            bt.logging.error(err)

        return web3_provider

    def sync(self, web3_provider: Web3):
        """Syncs with chain"""
        if not self._initted:
            self.pool_init(web3_provider)
        try:
            pool_abi_file_path = Path(__file__).parent / "../abi/Pool.json"
            pool_abi_file = pool_abi_file_path.open()
            pool_abi = json.load(pool_abi_file)
            pool_abi_file.close()

            atoken_contract_onchain = self._atoken_contract
            pool_address = retry_with_backoff(
                atoken_contract_onchain.functions.POOL().call
            )

            pool_contract = web3_provider.eth.contract(abi=pool_abi, decode_tuples=True)
            self._pool_contract = retry_with_backoff(
                pool_contract, address=pool_address
            )

            self._underlying_asset_address = retry_with_backoff(
                self._atoken_contract.functions.UNDERLYING_ASSET_ADDRESS().call
            )

            self._reserve_data = retry_with_backoff(
                self._pool_contract.functions.getReserveData(
                    self._underlying_asset_address
                ).call
            )

            reserve_strat_abi_file_path = (
                    Path(__file__).parent / "../abi/IReserveInterestRateStrategy.json"
            )
            reserve_strat_abi_file = reserve_strat_abi_file_path.open()
            reserve_strat_abi = json.load(reserve_strat_abi_file)
            reserve_strat_abi_file.close()

            strategy_contract = web3_provider.eth.contract(abi=reserve_strat_abi)
            self._strategy_contract = retry_with_backoff(
                strategy_contract,
                address=self._reserve_data.interestRateStrategyAddress,
            )

            stable_debt_token_abi_file_path = (
                    Path(__file__).parent / "../abi/IStableDebtToken.json"
            )
            stable_debt_token_abi_file = stable_debt_token_abi_file_path.open()
            stable_debt_token_abi = json.load(stable_debt_token_abi_file)
            stable_debt_token_abi_file.close()

            stable_debt_token_contract = web3_provider.eth.contract(
                abi=stable_debt_token_abi
            )
            stable_debt_token_contract = retry_with_backoff(
                stable_debt_token_contract,
                address=self._reserve_data.stableDebtTokenAddress,
            )

            (
                _,
                self._nextTotalStableDebt,
                self._nextAvgStableBorrowRate,
                _,
            ) = retry_with_backoff(
                stable_debt_token_contract.functions.getSupplyData().call
            )

            variable_debt_token_abi_file_path = (
                    Path(__file__).parent / "../abi/IVariableDebtToken.json"
            )
            variable_debt_token_abi_file = variable_debt_token_abi_file_path.open()
            variable_debt_token_abi = json.load(variable_debt_token_abi_file)
            variable_debt_token_abi_file.close()

            variable_debt_token_contract = web3_provider.eth.contract(
                abi=variable_debt_token_abi
            )
            self._variable_debt_token_contract = retry_with_backoff(
                variable_debt_token_contract,
                address=self._reserve_data.variableDebtTokenAddress,
            )

            nextVariableBorrowIndex = self._reserve_data.variableBorrowIndex

            nextScaledVariableDebt = retry_with_backoff(
                self._variable_debt_token_contract.functions.scaledTotalSupply().call
            )
            self._totalVariableDebt = rayMul(
                nextScaledVariableDebt, nextVariableBorrowIndex
            )

            reserveConfiguration = self._reserve_data.configuration
            self._reserveFactor = getReserveFactor(reserveConfiguration)
            self._decimals = retry_with_backoff(
                self._underlying_asset_contract.functions.decimals().call
            )

        except Exception as err:
            bt.logging.error("Failed to sync to chain!")
            bt.logging.error(err)

    # last 256 unique calls to this will be cached for the next 60 seconds
    @ttl_cache(maxsize=256, ttl=60)
    def supply_rate(self, user_addr: str, amount: int) -> int:
        """Returns supply rate given new deposit amount"""
        try:
            already_deposited = int(
                retry_with_backoff(
                    self._atoken_contract.functions.balanceOf(
                        Web3.to_checksum_address(user_addr)
                    ).call
                )
                * 10**self._decimals
                // 1e18
            )

            delta = amount - already_deposited
            to_deposit = delta if delta > 0 else 0
            to_remove = abs(delta) if delta < 0 else 0

            (nextLiquidityRate, _, _) = retry_with_backoff(
                self._strategy_contract.functions.calculateInterestRates(
                    (
                        self._reserve_data.unbacked,
                        int(to_deposit),
                        int(to_remove),
                        self._nextTotalStableDebt,
                        self._totalVariableDebt,
                        self._nextAvgStableBorrowRate,
                        self._reserveFactor,
                        self._underlying_asset_address,
                        self._atoken_contract.address,
                    )
                ).call
            )

            # return liquidity_rate / 1e27
            return Web3.to_wei(nextLiquidityRate / 1e27, "ether")

        except Exception as e:
            bt.logging.error("Failed to retrieve supply apy!")
            bt.logging.error(e)

        return 0


class VariableInterestSturdySiloStrategy(ChainBasedPoolModel):
    """This class defines the default pool type for Sturdy Silo"""

    pool_type: Literal['STURDY_SILO'] = Field(
        default='STURDY_SILO', description="type of pool"
    )

    _silo_strategy_contract: Contract = PrivateAttr()
    _pair_contract: Contract = PrivateAttr()
    _rate_model_contract: Contract = PrivateAttr()
    _curr_deposit_amount: int = PrivateAttr()
    _util_prec: int = PrivateAttr()
    _fee_prec: int = PrivateAttr()
    _totalAsset: int = PrivateAttr()
    _totalBorrow: int = PrivateAttr()
    _current_rate_info = PrivateAttr()
    _rate_prec: int = PrivateAttr()
    _block: web3.types.BlockData = PrivateAttr()

    def __hash__(self):
        return hash((self._silo_strategy_contract.address, self._pair_contract))

    def __eq__(self, other):
        if not isinstance(other, VariableInterestSturdySiloStrategy):
            return NotImplemented
        # Compare the attributes for equality
        return (self._silo_strategy_contract.address, self._pair_contract) == (
            other._silo_strategy_contract.address,
            other._pair_contract.address,
        )

    def pool_init(self, user_addr: str, web3_provider: Web3):
        try:
            assert web3_provider.is_connected()
        except Exception as err:
            bt.logging.error("Failed to connect to Web3 instance!")
            bt.logging.error(err)

        try:
            silo_strategy_abi_file_path = (
                    Path(__file__).parent / "../abi/SturdySiloStrategy.json"
            )
            silo_strategy_abi_file = silo_strategy_abi_file_path.open()
            silo_strategy_abi = json.load(silo_strategy_abi_file)
            silo_strategy_abi_file.close()

            silo_strategy_contract = web3_provider.eth.contract(
                abi=silo_strategy_abi, decode_tuples=True
            )
            self._silo_strategy_contract = retry_with_backoff(
                silo_strategy_contract, address=self.contract_address
            )

            pair_abi_file_path = Path(__file__).parent / "../abi/SturdyPair.json"
            pair_abi_file = pair_abi_file_path.open()
            pair_abi = json.load(pair_abi_file)
            pair_abi_file.close()

            pair_contract_address = retry_with_backoff(
                self._silo_strategy_contract.functions.pair().call
            )
            pair_contract = web3_provider.eth.contract(abi=pair_abi, decode_tuples=True)
            self._pair_contract = retry_with_backoff(
                pair_contract, address=pair_contract_address
            )

            rate_model_abi_file_path = (
                    Path(__file__).parent / "../abi/VariableInterestRate.json"
            )
            rate_model_abi_file = rate_model_abi_file_path.open()
            rate_model_abi = json.load(rate_model_abi_file)
            rate_model_abi_file.close()

            rate_model_contract_address = retry_with_backoff(
                self._pair_contract.functions.rateContract().call
            )
            rate_model_contract = web3_provider.eth.contract(
                abi=rate_model_abi, decode_tuples=True
            )
            self._rate_model_contract = retry_with_backoff(
                rate_model_contract, address=rate_model_contract_address
            )

            self._initted = True

        except Exception as e:
            bt.logging.error(e)

    def sync(self, user_addr: str, web3_provider: Web3):
        """Syncs with chain"""
        if not self._initted:
            self.pool_init(user_addr, web3_provider)

        user_shares = retry_with_backoff(
            self._pair_contract.functions.balanceOf(user_addr).call
        )
        self._curr_deposit_amount = retry_with_backoff(
            self._pair_contract.functions.convertToAssets(user_shares).call
        )

        constants = retry_with_backoff(
            self._pair_contract.functions.getConstants().call
        )
        self._util_prec = constants[2]
        self._fee_prec = constants[3]
        self._totalAsset = retry_with_backoff(
            self._pair_contract.functions.totalAsset().call
        )
        self._totalBorrow = retry_with_backoff(
            self._pair_contract.functions.totalBorrow().call
        )

        self._block = web3_provider.eth.get_block("latest")

        self._current_rate_info = retry_with_backoff(
            self._pair_contract.functions.currentRateInfo().call
        )

        self._rate_prec = retry_with_backoff(
            self._rate_model_contract.functions.RATE_PREC().call
        )

    # last 256 unique calls to this will be cached for the next 60 seconds
    @ttl_cache(maxsize=256, ttl=60)
    def supply_rate(self, amount: int) -> int:
        delta = amount - self._curr_deposit_amount
        """Returns supply rate given new deposit amount"""
        util_rate = (self._util_prec * self._totalBorrow.amount) // (
                self._totalAsset.amount + delta
        )

        last_update_timestamp = self._current_rate_info.lastTimestamp
        current_timestamp = self._block["timestamp"]
        delta_time = current_timestamp - last_update_timestamp

        protocol_fee = self._current_rate_info.feeToProtocolRate
        (new_rate_per_sec, _) = retry_with_backoff(
            self._rate_model_contract.functions.getNewRate(
                delta_time, util_rate, self._current_rate_info.fullUtilizationRate
            ).call
        )

        supply_apy = int(
            new_rate_per_sec
            * 31536000
            * 1e18
            * util_rate
            // self._rate_prec
            // self._util_prec
            * (1 - (protocol_fee / self._fee_prec))
        )  # (rate_per_sec_pct * seconds_in_year * util_rate_pct) * 1e18

        return supply_apy


def generate_assets_and_pools(rng_gen=np.random) -> Dict:  # generate pools
    assets_and_pools = {}

    pools = [
        BasePool(
            pool_id=str(x),
            pool_type=POOL_TYPES.SYNTHETIC,
            base_rate=randrange_float(
                MIN_BASE_RATE, MAX_BASE_RATE, BASE_RATE_STEP, rng_gen=rng_gen
            ),
            base_slope=randrange_float(
                MIN_SLOPE, MAX_SLOPE, SLOPE_STEP, rng_gen=rng_gen
            ),
            kink_slope=randrange_float(
                MIN_KINK_SLOPE, MAX_KINK_SLOPE, SLOPE_STEP, rng_gen=rng_gen
            ),  # kink rate - kicks in after pool hits optimal util rate
            optimal_util_rate=randrange_float(
                MIN_OPTIMAL_RATE,
                MAX_OPTIMAL_RATE,
                OPTIMAL_UTIL_STEP,
                rng_gen=rng_gen,
            ),  # optimal util rate - after which the kink slope kicks in
            borrow_amount=int(
                format_num_prec(
                    wei_mul(
                        POOL_RESERVE_SIZE,
                        randrange_float(
                            MIN_UTIL_RATE,
                            MAX_UTIL_RATE,
                            UTIL_RATE_STEP,
                            rng_gen=rng_gen,
                        ),
                    )
                )
            ),  # initial borrowed amount from pool
            reserve_size=POOL_RESERVE_SIZE,
        )
        for x in range(NUM_POOLS)
    ]

    pools = {str(pool.pool_id): pool for pool in pools}

    assets_and_pools["total_assets"] = math.floor(
        randrange_float(
            MIN_TOTAL_ASSETS, MAX_TOTAL_ASSETS, TOTAL_ASSETS_STEP, rng_gen=rng_gen
        )
    )
    assets_and_pools["pools"] = pools

    return assets_and_pools


# generate intial allocations for pools
def generate_initial_allocations_for_pools(
        assets_and_pools: Dict, rng_gen=np.random
) -> Dict:
    pools = assets_and_pools["pools"]
    alloc = assets_and_pools["total_assets"] / len(pools)
    allocations = {str(uid): alloc for uid in pools}

    return allocations
