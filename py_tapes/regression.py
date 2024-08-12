import os
import shutil
import subprocess
import argparse
from unli.commands.regression import regression

def regression_task(out_dir,root_dir,dataset,gpu_id,augmentation,training,threshold):

    os.makedirs(out_dir, exist_ok=True)
    
    # os.environ['PYTHONPATH'] = unli_dir
    
    command = [
        '/home/orisim/.virtualenvs/UNLI/bin/python', os.path.join(root_dir, 'unli', 'commands', 'regression.py'),
        '--root_dir', root_dir,
        '--data', dataset,
        '--out', out_dir
    ]

    if gpu_id :
        command.extend(['--gpu', str(gpu_id)])

    if augmentation :
        command.extend(['--augmentation', augmentation])
        if threshold :
            command.extend(['--threshold', threshold])

    if training :
        print("training augmentation")
        command.extend(['--training_augmentation'])


    subprocess.run(command, check=True)

    # Command to run regression.py script
    # command = f"/home/orisim/.virtualenvs/UNLI/bin/python {os.path.join(unli_dir, 'unli', 'commands', 'regression.py')} --data {dataset} --out {out_dir} --batch_size 64" # training_augmentation --augmentation --pretrained /sise/home/orisim/projects/UNLI/comet-atomic_2020_BART_aaai/pytorch_model.bin
    # command = f"/home/orisim/.virtualenvs/UNLI/bin/python {os.path.join(unli_dir, 'unli', 'commands', 'regression.py')} --data {dataset} --out {out_dir} --batch_size 64 --augmentation bart" # training_augmentation --augmentation --pretrained /sise/home/orisim/projects/UNLI/comet-atomic_2020_BART_aaai/pytorch_model.bin"
    # command = f"/home/orisim/.virtualenvs/UNLI/bin/python {os.path.join(unli_dir, 'unli', 'commands', 'regression.py')} --data {dataset} --out {out_dir} --batch_size 64 --training_augmentation" # training_augmentation --augmentation --pretrained /sise/home/orisim/projects/UNLI/comet-atomic_2020_BART_aaai/pytorch_model.bin"

    # if gpu_id is not None:
    #     command += f" --gpu {gpu_id}"
    #
    # os.system(command)

def plan_regression():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--rootdir", type=str, default="", help="root dir of your repository")
    parser.add_argument("--augmentation", type=str, help="Enable augmentation")  # action='store_true'
    parser.add_argument("--training_augmentation", action='store_true', help="trainig the augmentation you created")
    parser.add_argument("--threshold", type=str , help="threshold to be reference" , default="0.8")

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
        scenario_out_dir = os.path.join(outdir  ,"comet" , scenario)
        # regression_task(scenario_out_dir, rootdir, dataset_path , gpu_id, augmentation , training , threshold )
        regression(rootdir,dataset_path,ARGS.seed,ARGS.pretrained,scenario_out_dir,ARGS.margin,ARGS.num_samples,ARGS.batch_size,ARGS.gpuid,augmentation,training,threshold)



plan_regression()