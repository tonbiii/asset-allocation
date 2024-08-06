import json
import time
from argparse import ArgumentParser
from threading import Thread

import requests as re


def parse():
    parser = ArgumentParser()
    parser.add_argument('instances', type=int, default=10)
    return parser.parse_args()


def data_gen(sample_filepath: str, total_assets=2):
    with open(sample_filepath, 'r') as f:
        lines = f.readlines()

    for i in range(len(lines) // 4):
        pools = json.loads(lines[i * 4 + 0].replace('\'', '"'))
        yield dict(
            total_assets=total_assets,
            pools=pools
        )


def post(assets_and_pools, ip):
    response = re.post(f'http://{ip}/predict', json={'assets_and_pools': assets_and_pools})
    return response.json()


def run():
    for assets_and_pools in data_gen('training_data.txt'):
        time_0 = time.time_ns()
        response = post(assets_and_pools, ip='127.0.0.1:8080')
        time_1 = time.time_ns()
        print(f"time processing: {(time_1 - time_0)//1000_000}")


if __name__ == '__main__':
    args = parse()

    threads = []
    for i in range(args.instances):
        thread = Thread(target=run)
        threads.append(thread)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
