import argparse
import yaml

from training.train import train
from training.evaluate import evaluate

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train","eval"], help="train or eval")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.mode == "train":
        train(argparse.Namespace(**cfg["train"]))
    else:
        evaluate(argparse.Namespace(**cfg["eval"]))

if __name__ == "__main__":
    main()

