import argparse
import sys

import wandb

# Hyperband configuration
FIRST_EVAL = 2  # iterations
NUM_BRACKETS = 3
ETA = 3
brackets = [FIRST_EVAL * (ETA**i) for i in range(NUM_BRACKETS)]
MAX_ITERATIONS = brackets[-1]

SWEEP_CFG = {
    "method": "bayes",
    "metric": {
        "name": "mean_reward",
        "goal": "maximize",
    },
    "parameters": {
        # Using the exact same parameter names as in ExperimentArgumentHandler
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-3,
        },
        "clip-param": {
            "distribution": "uniform",
            "min": 0.1,
            "max": 0.4,
        },
        "entropy-coef": {
            "distribution": "log_uniform_values",
            "min": 0.001,
            "max": 0.05,
        },
        "value-loss-coef": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 2.0,
        },
        "ppo-epochs": {
            "values": [5, 10, 15],
        },
        "bs": {
            "values": [1600, 3200, 6400],
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
            "values": [25, 50],
        },
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": FIRST_EVAL,
        "eta": ETA,
    },
}


def run_sweep() -> None:
    run = wandb.init(project="MPE_simple_spread_v3")
    sweep_config = run.config
    args = []
    for key, value in sweep_config.items():
        if isinstance(value, bool) and value:
            args.append(f"--{key}")
        else:
            args.append(f"--{key}")
            args.append(str(value))
    args.extend(["--wandb", "--steps", str(MAX_ITERATIONS * sweep_config.get("envs", 128) * sweep_config.get("buffer-len", 25))])
    sys.argv[1:] = args

    import mat.scripts.mpe as train_script

    train_script.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run or continue a W&B sweep")
    parser.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="Existing sweep ID to continue / add an agent to a run. If not provided, a new sweep will be created. Format: username/project/sweep_id",
    )
    print(f"Training will use these evaluation checkpoints (iterations): {brackets}")
    print(f"Final training length will be {MAX_ITERATIONS} iterations")

    args = parser.parse_args()
    sweep_id = wandb.sweep(SWEEP_CFG, project="MPE_simple_spread_v3") if args.sweep_id is None else args.sweep_id
    wandb.agent(sweep_id, function=run_sweep)
