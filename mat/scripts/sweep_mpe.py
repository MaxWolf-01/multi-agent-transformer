import mat.scripts.mpe as train_script
from mat.utils import HyperbandConfig, run_sweep

hyperband = HyperbandConfig(first_eval=3, num_brackets=3, eta=3)
sweep_params = {
    # Using the exact same parameter names as in ExperimentArgumentHandler
    "lr": {
        "distribution": "log_uniform_values",
        "min": 1e-5,
        "max": 1e-3,
    },
    "clip-param": {
        "distribution": "uniform",
        "min": 0.05,
        "max": 0.5,
    },
    "entropy-coef": {
        "distribution": "log_uniform_values",
        "min": 0.0001,
        "max": 0.05,
    },
    "value-loss-coef": {
        "distribution": "uniform",
        "min": 0.5,
        "max": 2.0,
    },
    "bs": {
        "values": [1600, 3200, 6400],
    },
    "ppo-epochs": {
        "values": [5, 10, 15],
    },
    "embed-dim": {
        "values": [32, 64, 128],
    },
    "encoder-depth": {
        "values": [1, 2, 3],
    },
    "decoder-depth": {
        "values": [1, 2, 3],
    },
    "n-heads": {
        "values": [1, 2, 4],
    },
    "envs": {
        "values": [64, 128, 256],
    },
    "buffer-len": {
        "values": [25, 50, 100],
    },
    "gamma": {
        "distribution": "uniform",
        "min": 0.95,
        "max": 0.999,
    },
    "gae-lambda": {
        "distribution": "uniform",
        "min": 0.9,
        "max": 0.99,
    },
    "max-grad-norm": {
        "distribution": "uniform",
        "min": 0.5,
        "max": 5.0,
    },
}

if __name__ == "__main__":
    run_sweep(project="MPE_simple_spread_v3", sweep_params=sweep_params, hyperband=hyperband, train_fn=train_script.main)
