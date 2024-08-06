import multiprocessing
import os
import time
from argparse import ArgumentParser

from flask import Flask, jsonify, request

from src.forest_allocation import RandomForestAllocation
from src.sgd_allocation import SGDAllocation


def parse():
    parser = ArgumentParser()
    parser.add_argument('instances', type=int, default=10)
    return parser.parse_args()


args = parse()


class AllocationProcess:
    def __init__(self):
        self._model = RandomForestAllocation()
        self._sgd = SGDAllocation()
        print("PID:", os.getpid())

    def process(self, assets_and_pools):
        model_allocation = self._model.predict_allocation(assets_and_pools)
        sgd_allocation = self._sgd.predict_allocation(assets_and_pools, initial_allocations=model_allocation)
        return sgd_allocation


def task(data):
    return worker_instance.process(data)


def init_worker():
    global worker_instance
    worker_instance = AllocationProcess()


app = Flask(__name__)

pool = multiprocessing.Pool(processes=args.instances, initializer=init_worker, initargs=())


@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        data = request.get_json()
        assets_and_pools = data['assets_and_pools']

        t1 = time.time()
        allocations = pool.apply(task, (assets_and_pools,))
        t2 = time.time()
        print(f"Time: {(t2 - t1) * 1000:.2f} ms")

        response = {
            "message": "predict successfully",
            "result": allocations
        }
        return jsonify(response), 200
    else:
        return jsonify({"error": "Request must be JSON"}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=8080)
