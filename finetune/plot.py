import json, os
from typing import Literal
from math import exp, log as ln

import matplotlib.pyplot as plt

# Path to your JSON log file

directory = 'models'
log_files = []
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.log'):
            log_files.append(os.path.join(root, file))


steps = []
train_losses = []
eval_losses = []
test_losses = []

def load_logs(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_losses(train_logs: dict) -> dict:
    """
    single batch: {
      "loss": 5.2304,
      "grad_norm": 49.25,
      "learning_rate": 7.750000000000001e-07,
      "epoch": 1.0,
      "step": 32
    }

    whole epoch: {
      "eval_loss": 5.052532196044922,
      "eval_model_preparation_time": 0.0076,
      "eval_runtime": 8.3437,
      "eval_samples_per_second": 59.926,
      "eval_steps_per_second": 14.981,
      "epoch": 1.0,
      "step": 32
    }
    """
    train_steps, train_epochs, train_losses = [], [], []
    eval_steps, eval_epochs, eval_losses = [], [], []
    for entry in train_logs:
        if "loss" in entry:
            train_steps.append(entry["step"])
            train_epochs.append(entry.get("epoch"))
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry["step"])
            eval_epochs.append(entry.get("epoch"))
            eval_losses.append(entry["eval_loss"])
    return {
        "train": {"steps": train_steps, "epochs": train_epochs, "losses": train_losses},
        "eval": {"steps": eval_steps, "epochs": eval_epochs, "losses": eval_losses},
    }



def plot_losses(data, name, custom_x_train=None, custom_x_eval=None, mode:Literal["default", "perplexity"] = "default"):
    mode = mode.lower()

    train_series = data["train"]
    eval_series = data["eval"]

    if mode == 'perplexity':
        train_series['perplexity'] = list(map(exp, train_series['losses']))
        eval_series['perplexity']  = list(map(exp, eval_series['losses']))


    x_train = custom_x_train
    x_eval = custom_x_eval

    plt.figure(figsize=(10,6))
    if train_series["losses"]:
        if mode == "default":
            plt.plot(x_train, train_series["losses"], label="Train Loss", color="blue", linewidth=1)
        elif mode == "perplexity":
            plt.plot(x_train, train_series["perplexity"], label="Train Perplexity", color="blueviolet", linewidth=1)
        else:
            raise ValueError(f"Unknown mode {mode}")

    if eval_series["losses"]:
        if mode == "default":
            plt.plot(x_eval, eval_series["losses"], label="Eval Loss", color="orange", linewidth=1.2, marker="o", markersize=3)
        elif mode == 'perplexity':
            plt.plot(x_eval, eval_series["perplexity"], label="Eval Perplexity", color="red", linewidth=1.2, marker="o", markersize=3)
        else:
            raise ValueError(f"Unknown mode {mode}")

    ylabel = "Loss" if mode == 'default' else "Perplexity"
    title = f'{ylabel} Curves'
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.title(f"{title} Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plot/{os.path.basename(log_file)}_{ylabel}.png', dpi=150)
    plt.show()

def main(log_file: str):
    stats = load_logs(log_file)
    stats["train_logs"].insert(0, {
        'epoch': 0.0,
        'step': 0.0,
        'eval_loss': ln(stats['raw_perplexity']),
    })
    train_logs = stats["train_logs"]
    losses = extract_losses(train_logs)

    train_steps = losses["train"]["steps"]
    if train_steps:
        train_epochs = losses["train"]["epochs"]
        eval_epochs = losses["eval"]["epochs"]
        plot_losses(losses, log_file, custom_x_train=train_epochs, custom_x_eval=eval_epochs, mode="default")
        plot_losses(losses, log_file, custom_x_train=train_epochs, custom_x_eval=eval_epochs, mode='perplexity')
    else:
        plot_losses(losses, log_file)

if __name__ == "__main__":
    for log_file in log_files:
        main(log_file)
