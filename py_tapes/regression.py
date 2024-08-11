import os
import shutil
import subprocess

def regression_task(out_dir, unli_dir, dataset, gpu_id):
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # Set PYTHONPATH to include unli directory
    os.environ['PYTHONPATH'] = unli_dir
    
    # Command to run regression.py script
    # command = [
    #     'python', os.path.join(unli_dir, 'unli', 'commands', 'regression.py'),
    #     '--data', dataset,
    #     '--out', out_dir
    # ]
    #
    # # Optionally specify GPU ID if provided
    # if gpu_id is not None:
    #     command.extend(['--gpu', str(gpu_id)])
    #
    # # subprocess.run(command, check=True)

    # Command to run regression.py script
    # command = f"/home/orisim/.virtualenvs/UNLI/bin/python {os.path.join(unli_dir, 'unli', 'commands', 'regression.py')} --data {dataset} --out {out_dir} --batch_size 64" # training_augmentation --augmentation --pretrained /sise/home/orisim/projects/UNLI/comet-atomic_2020_BART_aaai/pytorch_model.bin
    command = f"/home/orisim/.virtualenvs/UNLI/bin/python {os.path.join(unli_dir, 'unli', 'commands', 'regression.py')} --data {dataset} --out {out_dir} --batch_size 64 --augmentation bart" # training_augmentation --augmentation --pretrained /sise/home/orisim/projects/UNLI/comet-atomic_2020_BART_aaai/pytorch_model.bin"
    # command = f"/home/orisim/.virtualenvs/UNLI/bin/python {os.path.join(unli_dir, 'unli', 'commands', 'regression.py')} --data {dataset} --out {out_dir} --batch_size 64 --training_augmentation" # training_augmentation --augmentation --pretrained /sise/home/orisim/projects/UNLI/comet-atomic_2020_BART_aaai/pytorch_model.bin"

    # Optionally specify GPU ID if provided
    if gpu_id is not None:
        command += f" --gpu {gpu_id}"

    # Execute the command
    os.system(command)

def plan_regression():
    # Define datasets and other parameters
    unli_dir = r"/sise/home/orisim/projects/UNLI/"  # Replace with actual path
    out_dir =  r"/sise/home/orisim/projects/UNLI/"  # Replace with desired output path
    gpu_id = None  # Specify GPU ID if needed
    dataset = {
        # 'surrogate': unli_dir+"/surrogate_dataset",
        'combined': unli_dir+"/combined_dataset",
        # 'usnli': unli_dir+"/usnli_dataset",
        # 'hyp-only-surrogate': '/path/to/hyp_only_surrogate_dataset',
        # 'hyp-only-combined': '/path/to/hyp_only_combined_dataset',
        # 'hyp-only-usnli': '/path/to/hyp_only_usnli_dataset'
    }
    
    # Iterate over each dataset scenario
    for scenario, dataset_path in dataset.items():
        scenario_out_dir = os.path.join(out_dir  ,"comet" , scenario)
        regression_task(scenario_out_dir, unli_dir, dataset_path, gpu_id)

# Execute the plan
plan_regression()