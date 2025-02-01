from mat.utils import HyperbandConfig, run_sweep

import mat.scripts.mamujoco as train_script

hyperband = HyperbandConfig(first_eval=5, num_brackets=4, eta=2)
sweep_params = {
    # Using the exact same parameter names as in ExperimentArgumentHandler
    "lr": {
        "distribution": "log_uniform_values",
        "min": 1e-6,  # Paper uses 5e-5
        "max": 1e-3,
    },
    "clip-param": {
        "distribution": "uniform",
        "min": 0.05,  # Paper uses 0.05
        "max": 0.4,
    },
    "entropy-coef": {
        "distribution": "log_uniform_values",
        "min": 0.0001,
        "max": 0.01,
    },
    "value-loss-coef": {
        "distribution": "uniform",
        "min": 0.5,
        "max": 2.0,  # Paper uses 1.0
    },
    "bs": {
        "values": [64, 100, 200, 400],  # Paper uses 100
    },
    "ppo-epochs": {
        "values": [5, 10, 15],
    },
    "embed-dim": {
        "values": [32, 64, 128, 256],  # Paper uses 64
    },
    "encoder-depth": {
        "values": [1, 2, 3],  # Paper uses 1
    },
    "decoder-depth": {
        "values": [1, 2, 3],  # Paper uses 1
    },
    "n-heads": {
        "values": [1, 2, 4, 8],  # Paper uses 1
    },
    "envs": {
        "values": [20, 40, 80],  # Paper uses 40
    },
    "buffer-len": {
        "values": [100, 200, 400],  # Paper uses 100
    },
    "gamma": {
        "distribution": "uniform",
        "min": 0.95,
        "max": 0.999,  # Paper uses 0.99
    },
    "gae-lambda": {
        "distribution": "uniform",
        "min": 0.9,
        "max": 0.99,  # Paper uses 0.95
    },
    "max-grad-norm": {
        "distribution": "uniform",
        "min": 0.5,  # Paper uses 0.5
        "max": 5.0,
    },
}


if __name__ == "__main__":
    run_sweep(project="MaMuJoCo_HalfCheetah", sweep_params=sweep_params, hyperband=hyperband, train_fn=train_script.main)
