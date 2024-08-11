import os
import subprocess
import shutil

# Define the global paths and URLs
unli = r"/sise/home/orisim/projects/UNLI/"
snli_qrels = unli+"snli_qrels"
usnli_qrels = unli+"usnli_qrels"
aggregator = "mean"


def surrogate_scores(snli, usnli, aggregator, out):
    command = [
        "python", f"{unli}/scripts/compute_surrogates.py",
        "--snli", snli,
        "--usnli", f"{usnli}/train.qrels",
        "--aggregator", aggregator
    ]
    with open(out, "w") as output_file:
        subprocess.run(command, stdout=output_file)


def snli_with_surrogates_dataset(snli, usnli, surrogate, out):
    os.makedirs(out, exist_ok=True)
    command = [
        "python", f"{unli}/scripts/gen_snli_with_surrogates_dataset.py",
        "--input", f"{snli}/train.qrels",
        "--surrogate", surrogate
    ]
    train_qrels = os.path.join(out, "train.qrels")
    with open(train_qrels, "w") as output_file:
        subprocess.run(command, stdout=output_file)

    shutil.copy(f"{snli}/train.l", os.path.join(out, "train.l"))
    shutil.copy(f"{snli}/train.r", os.path.join(out, "train.r"))
    shutil.copy(f"{snli}/dev.l", os.path.join(out, "dev.l"))
    shutil.copy(f"{snli}/dev.r", os.path.join(out, "dev.r"))
    shutil.copy(f"{snli}/test.l", os.path.join(out, "test.l"))
    shutil.copy(f"{snli}/test.r", os.path.join(out, "test.r"))
    shutil.copy(f"{usnli}/dev.qrels", os.path.join(out, "dev.qrels"))
    shutil.copy(f"{usnli}/test.qrels", os.path.join(out, "test.qrels"))


def snli_combined_with_usnli_dataset(snli, usnli, surrogate, out):
    os.makedirs(out, exist_ok=True)
    command = [
        "python", f"{unli}/scripts/gen_combined_fallback_surrogate_dataset.py",
        "--snli", f"{snli}/train.qrels",
        "--usnli", f"{usnli}/train.qrels",
        "--surrogate", surrogate
    ]
    train_qrels = os.path.join(out, "train.qrels")
    with open(train_qrels, "w") as output_file:
        subprocess.run(command, stdout=output_file)

    shutil.copy(f"{snli}/train.l", os.path.join(out, "train.l"))
    shutil.copy(f"{snli}/train.r", os.path.join(out, "train.r"))
    shutil.copy(f"{snli}/dev.l", os.path.join(out, "dev.l"))
    shutil.copy(f"{snli}/dev.r", os.path.join(out, "dev.r"))
    shutil.copy(f"{snli}/test.l", os.path.join(out, "test.l"))
    shutil.copy(f"{snli}/test.r", os.path.join(out, "test.r"))
    shutil.copy(f"{usnli}/dev.qrels", os.path.join(out, "dev.qrels"))
    shutil.copy(f"{usnli}/test.qrels", os.path.join(out, "test.qrels"))


def usnli_dataset(snli, usnli, out):
    os.makedirs(out, exist_ok=True)

    shutil.copy(f"{snli}/train.l", os.path.join(out, "train.l"))
    shutil.copy(f"{snli}/train.r", os.path.join(out, "train.r"))
    shutil.copy(f"{snli}/dev.l", os.path.join(out, "dev.l"))
    shutil.copy(f"{snli}/dev.r", os.path.join(out, "dev.r"))
    shutil.copy(f"{snli}/test.l", os.path.join(out, "test.l"))
    shutil.copy(f"{snli}/test.r", os.path.join(out, "test.r"))
    shutil.copy(f"{usnli}/train.qrels", os.path.join(out, "train.qrels"))
    shutil.copy(f"{usnli}/dev.qrels", os.path.join(out, "dev.qrels"))
    shutil.copy(f"{usnli}/test.qrels", os.path.join(out, "test.qrels"))


# Execute the tasks

# surrogate_scores(snli_qrels, usnli_qrels, aggregator, "surrogate.scores")

# snli_with_surrogates_dataset(snli_qrels, usnli_qrels, "surrogate.scores", unli+"/surrogate_dataset")
snli_combined_with_usnli_dataset(snli_qrels, usnli_qrels, "surrogate.scores", unli+"/combined_dataset")
# usnli_dataset(snli_qrels, usnli_qrels, os.path.join(unli, "usnli_dataset")) #  "path/to/output/uSnliDataset"