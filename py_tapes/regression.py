import os
import shutil
import subprocess
import argparse

def regression_task(out_dir, unli_dir, dataset, gpu_id,augmentation,training):

    os.makedirs(out_dir, exist_ok=True)
    
    # os.environ['PYTHONPATH'] = unli_dir
    
    command = [
        'python', os.path.join(unli_dir, 'unli', 'commands', 'regression.py'),
        '--data', dataset,
        '--out', out_dir
    ]

    if gpu_id :
        command.extend(['--gpu', str(gpu_id)])

    if augmentation :
        command.extend(['--augmentation', augmentation])

    if training and augmentation is None:
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
    parser.add_argument("--outdir", type=str, default="", help="Output path to store the weights")
    parser.add_argument("--augmentation", type=str, help="Enable augmentation")  # action='store_true'
    parser.add_argument("--training_augmentation", action='store_true', help="trainig the augmentation you created")

    ARGS = parser.parse_args()
    unli_dir = os.getenv('PYTHONPATH') or ARGS.rootdir
    out_dir = ARGS.outdir

    augmentation  , training = ARGS.augmentation, ARGS.training_augmentation
    gpu_id = None

    dataset = {
        'combined': unli_dir + "/combined_dataset",
        # 'surrogate': unli_dir+"/surrogate_dataset",
        # 'usnli': unli_dir+"/usnli_dataset",
        # 'hyp-only-surrogate': '/path/to/hyp_only_surrogate_dataset',
        # 'hyp-only-combined': '/path/to/hyp_only_combined_dataset',
        # 'hyp-only-usnli': '/path/to/hyp_only_usnli_dataset'
    }
    
    for scenario, dataset_path in dataset.items():
        scenario_out_dir = os.path.join(out_dir  ,"comet" , scenario)
        regression_task(scenario_out_dir, unli_dir, dataset_path, gpu_id,augmentation,training )



plan_regression()