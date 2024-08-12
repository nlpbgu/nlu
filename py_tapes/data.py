# 1. Downloading and Extracting SNLI Dataset (task snli)

import os
import shutil
import subprocess
import os
import shutil
import subprocess
import urllib.request
import zipfile
import argparse
from py_tapes.assemble import *


# Define global variables
snli_url = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
usnli_url = "http://nlp.jhu.edu/unli/u-snli.zip"

# root_dir = r"C:\Users\User\Desktop\MSc\NLU\unli"
parser = argparse.ArgumentParser(description="")
parser.add_argument("--rootdir", type=str, default="", help="Path to root dir of your repository")
ARGS = parser.parse_args()
rootdir = ARGS.rootdir
os.environ['PYTHONPATH'] = rootdir


# Task functions
def download_and_extract(url, output_dir):
    # Download the zip file
    zip_filename = os.path.basename(url)
    zip_path = os.path.join(rootdir, zip_filename)
    urllib.request.urlretrieve(url, zip_path)

    # Extract the contents
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        print(zip_path)
        print(output_dir)

        zip_ref.extractall(output_dir)

    # Clean up: remove the zip file
    os.remove(zip_path)


def move_all_files(source_dir, dest_dir):

    os.makedirs(dest_dir, exist_ok=True)

    for filename in os.listdir(source_dir):
        source_file = os.path.join(source_dir, filename)
        dest_file = os.path.join(dest_dir, filename)

        if os.path.isfile(source_file):
            shutil.move(source_file, dest_file)


def snli_task(out_dir):
    snli_out_dir = os.path.join(rootdir, out_dir)
    os.makedirs(snli_out_dir, exist_ok=True)
    download_and_extract(snli_url, snli_out_dir)
    # snli_extracted_dir = os.path.join(snli_out_dir, "snli_1.0")
    # shutil.move(os.path.join(snli_extracted_dir, "*.*"), snli_out_dir)
    # move_all_files(snli_extracted_dir,snli_out_dir)

    # os.rmdir(snli_extracted_dir)

def usnli_csv_task(out_dir):
    usnli_out_dir = os.path.join(rootdir, out_dir)
    os.makedirs(usnli_out_dir, exist_ok=True)
    download_and_extract(usnli_url, usnli_out_dir)
    # shutil.move(os.path.join(usnli_out_dir, "*.csv"), usnli_out_dir)
    # move_all_files(usnli_out_dir,usnli_out_dir)


def snli_qrels_task( out_dir):
    snli_dir = os.path.join(rootdir, "snli_dataset/snli_1.0")
    snli_qrels_out_dir = os.path.join(rootdir, out_dir)
    os.makedirs(snli_qrels_out_dir, exist_ok=True)

    subprocess.run(["python", os.path.join(rootdir, "scripts/snli_to_qrels.py"),
                    "--snli", os.path.join(snli_dir, "snli_1.0"),
                    "--out", snli_qrels_out_dir])


def usnli_qrels_task(u_snli_csv_dir, out_dir):
    # unli_dir = os.path.join(root_dir, unli_dir)

    usnli_csv_dir = os.path.join(rootdir, u_snli_csv_dir)
    usnli_qrels_out_dir = os.path.join(rootdir, out_dir)

    os.makedirs(usnli_qrels_out_dir, exist_ok=True)
    subprocess.run(["python", os.path.join(rootdir, "scripts/usnli_to_qrels_align.py"),
                    "--snli", os.path.join(rootdir, "snli_qrels"),
                    "--usnli_train", os.path.join(usnli_csv_dir, "train.csv"),
                    "--usnli_dev", os.path.join(usnli_csv_dir, "dev.csv"),
                    "--usnli_test", os.path.join(usnli_csv_dir, "test.csv"),
                    "--out", usnli_qrels_out_dir])

# Execute tasks
snli_task("snli_dataset")
usnli_csv_task("usnli_dataset")
snli_qrels_task("snli_qrels")
usnli_qrels_task("usnli_dataset", "usnli_qrels")


snli_qrels = os.path.join(rootdir,"snli_qrels")
usnli_qrels =os.path.join(rootdir,"usnli_qrels")
aggregator = "mean"

surrogate_scores(rootdir,snli_qrels, usnli_qrels, aggregator, "surrogate.scores")
snli_with_surrogates_dataset(rootdir,snli_qrels, usnli_qrels, "surrogate.scores", rootdir+"/surrogate_dataset")
snli_combined_with_usnli_dataset(rootdir,snli_qrels, usnli_qrels, "surrogate.scores", rootdir+"/combined_dataset")
usnli_dataset(snli_qrels, usnli_qrels, os.path.join(rootdir, "usnli_dataset"))

