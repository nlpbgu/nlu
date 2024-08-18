import os
import shutil
import subprocess
import argparse
from unli.commands.regression import regression
from datetime import datetime


def plan_regression():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--rootdir", type=str, default="", help="root dir of your repository")
    parser.add_argument("--augmentation", type=str, help="Enable augmentation")  # action='store_true'
    parser.add_argument("--training_augmentation", action='store_true', help="trainig the augmentation you created")
    parser.add_argument("--threshold", type=str , help="threshold to be reference" , default="0.8")
    parser.add_argument("--nli", type=str, help="type of nli. can be [ENT,CON,NEU]")
    parser.add_argument("--nli1", type=str, help="")
    parser.add_argument("--nli2", type=str, help="")
    parser.add_argument("--dir_augmentation", type=str, help="this parameter to identity the directory to training our augmentation")

    parser.add_argument("--outdir", type=str, default="", help="Output path to store the weights")
    parser.add_argument("--pretrained", type=str, default="", help="Pretrained model")
    parser.add_argument("--margin", type=float, default=0.3, help="")
    parser.add_argument("--num_samples", type=int, default=1, help="")
    parser.add_argument("--seed", type=int, default=0xCAFEBABE, help="")
    parser.add_argument("--batch_size", type=int, default=64, help="")
    parser.add_argument("--gpuid", type=int, default=0)


    ARGS = parser.parse_args()
    print("PYTHONPATH" , os.getenv('PYTHONPATH'))
    rootdir =  ARGS.rootdir
    outdir = ARGS.outdir
    nli1 = ARGS.nli1
    nli2 = ARGS.nli2
    nli = ARGS.nli

    threshold = ARGS.threshold
    augmentation  , training = ARGS.augmentation , ARGS.training_augmentation
    gpu_id = None

    dataset = {
        'combined': rootdir + "/combined_dataset",
        # 'surrogate': unli_dir+"/surrogate_dataset",
        # 'usnli': unli_dir+"/usnli_dataset",
        # 'hyp-only-surrogate': '/path/to/hyp_only_surrogate_dataset',
        # 'hyp-only-combined': '/path/to/hyp_only_combined_dataset',
        # 'hyp-only-usnli': '/path/to/hyp_only_usnli_dataset'
    }


    for scenario, dataset_path in dataset.items():

        if augmentation:

            dir_augmentation = f"{scenario}_aug_{augmentation}_threshold_{threshold}_nli1_{nli1}_nli2_{nli2}_nli_{nli}"
            scenario_out_dir = os.path.join(outdir, "comet", f"{dir_augmentation}")

        if training:
            dir_augmentation = ARGS.dir_augmentation
            scenario_out_dir = os.path.join(outdir, "comet", f"{dir_augmentation}_trainingaugmentation")



        regression(rootdir,dataset_path,ARGS.seed,ARGS.pretrained,scenario_out_dir,ARGS.margin,ARGS.num_samples,ARGS.batch_size,ARGS.gpuid,augmentation,training,threshold,dir_augmentation)



plan_regression()