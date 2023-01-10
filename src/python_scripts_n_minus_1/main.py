import argparse
from experiment import run_experiment

def main(parsed_args):

    args = {
        "grid": parsed_args.GRID,
        "primal": eval(parsed_args.PRIMAL),
        "architecture_size": parsed_args.ARCHITECTURE_SIZE,
        "batch_size": parsed_args.BATCH_SIZE,
        "kernel_wt": parsed_args.KERNEL_WT,
        "train_size": parsed_args.TRAIN_SIZE,
        "test_size": parsed_args.TEST_SIZE,
        "kernel_size": parsed_args.KERNEL_SIZE,
        "learning_rate": parsed_args.LEARNING_RATE,
        "patience": parsed_args.PATIENCE,
        "min_epochs": parsed_args.MIN_EPOCHS,
        "max_epochs": parsed_args.MAX_EPOCHS,
        "results": parsed_args.RESULTS,
    }
    args.update({"samples": parsed_args.DATA + "uniform_processed_samples_10000_" + parsed_args.GRID + ".json"})
    args.update({"samples_test": parsed_args.DATA_TEST + "Uniform_samples_1000_" + parsed_args.GRID + ".json"})

    for seed in parsed_args.SEEDS.split():
        args.update({"seed": int(seed)})
        for architecture in parsed_args.ARCHITECTURES.split():
            args.update({"architecture": architecture})
            if architecture not in ["fcnn", "fcnn1", "cnn"]:
                args.update({"local": eval(parsed_args.LOCAL)})
            else:
                args.update({"local": False})
            if not eval(parsed_args.PRIMAL):
                args.update({"local": False})
            run_experiment(args)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--DIRECTORY", dest="DIRECTORY", type=str)
    parser.add_argument("--DATA", dest="DATA", type=str)
    parser.add_argument("--DATA_TEST", dest="DATA_TEST", type=str)
    parser.add_argument("--RESULTS", dest="RESULTS", type=str)
    parser.add_argument("--GRID", dest="GRID", type=str)
    parser.add_argument("--PRIMAL", dest="PRIMAL", type=str)
    parser.add_argument("--LOCAL", dest="LOCAL", type=str)
    parser.add_argument("--ARCHITECTURES", dest="ARCHITECTURES", type=str)
    parser.add_argument("--ARCHITECTURE_SIZE", dest="ARCHITECTURE_SIZE", type=str, default="big")
    parser.add_argument("--KERNEL_SIZE", dest="KERNEL_SIZE", type=int, default=3)
    parser.add_argument("--PATIENCE", dest="PATIENCE", type=int, default=5)
    parser.add_argument("--MIN_EPOCHS", dest="MIN_EPOCHS", type=int, default=5)
    parser.add_argument("--MAX_EPOCHS", dest="MAX_EPOCHS", type=int, default=10)
    parser.add_argument("--BATCH_SIZE", dest="BATCH_SIZE", type=int, default=100)
    parser.add_argument("--TRAIN_SIZE", dest="TRAIN_SIZE", type=float, default=0.8)
    parser.add_argument("--TEST_SIZE", dest="TEST_SIZE", type=float, default=0.1)
    parser.add_argument("--KERNEL_WT", dest="KERNEL_WT", type=float, default=0)
    parser.add_argument("--LEARNING_RATE", dest="LEARNING_RATE", type=float, default=0)
    parser.add_argument("--SEEDS", dest="SEEDS", type=str, default="123")
    main(parser.parse_args())
