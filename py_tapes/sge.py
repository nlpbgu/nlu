import os
import subprocess
import time

def create_wrapper():
    wrapper = "ducttape_sge_job.sh"
    with open(wrapper, 'w') as f:
        f.write("#!/usr/bin/env bash\n\n")
        f.write(f"#$ {resource_flags}\n")
        f.write(f"#$ {action_flags}\n")
        f.write("#$ -j y\n")
        f.write(f"#$ -o {os.getcwd()}/job.out\n")
        f.write(f"#$ -e {os.getcwd()}/job.err\n")
        f.write(f"#$ -N {TASK}[{REALIZATION}]{CONFIGURATION}\n\n")
        f.write("set -euo pipefail\n\n")
        f.write(TASK_VARIABLES.replace("=", "=\"").replace("\n", "\"\n") + "\n")
        f.write("cat >> {wrapper} <<'EOF'\n")
        f.write("set +u\n")
        f.write("if [[ ! -z ${pyenv:-} ]]; then\n")
        f.write("  virtualenv=$pyenv\n")
        f.write("  if [[ $virtualenv == conda:* ]]; then\n")
        f.write("    . /etc/profile.d/conda.sh\n")
        f.write("    target=$(echo $virtualenv | cut -d: -f2-)\n")
        f.write("    source deactivate\n")
        f.write("    conda activate $target\n")
        f.write("  else\n")
        f.write("    source $virtualenv\n")
        f.write("  fi\n")
        f.write("fi\n")
        f.write("set -u\n")
        f.write("EOF\n\n")
        f.write(f"cd {os.getcwd()}\n\n")
        f.write('echo "HOSTNAME: $(hostname)"\n')
        f.write("echo\n")
        f.write('echo "CUDA in ENV:"\n')
        f.write("env | grep CUDA\n")
        f.write("env | grep SGE\n\n")
        f.write(COMMANDS + "\n")
        f.write(f"echo $? > {os.getcwd()}/exitcode\n")

    return wrapper

def submit_job(wrapper):
    result = subprocess.run(f"qsub -V -S /bin/bash {wrapper} | grep -Eo 'Your job [0-9]+' | grep -Eo '[0-9]+'", shell=True, stdout=subprocess.PIPE)
    job_id = result.stdout.decode().strip()
    with open(f"{os.getcwd()}/job_id", 'w') as f:
        f.write(job_id)

    return job_id

def print_wrapper_contents(wrapper):
    with open(wrapper, 'r') as f:
        print(f.read())

def main():
    global resource_flags, action_flags, TASK, REALIZATION, TASK_VARIABLES, CONFIGURATION, COMMANDS
    # Assign your variables here
    resource_flags = "<resource_flags>"
    action_flags = "<action_flags>"
    TASK = "<TASK>"
    REALIZATION = "<REALIZATION>"
    TASK_VARIABLES = "<TASK_VARIABLES>"
    CONFIGURATION = "<CONFIGURATION>"
    COMMANDS = "<COMMANDS>"

    wrapper = create_wrapper()

    # Print the contents of the wrapper file
    print_wrapper_contents(wrapper)

    job_id = submit_job(wrapper)

    def exitfn():
        print(f"wait until I kill the job {job_id}")
        subprocess.run(f"qdel {job_id}", shell=True)
        exit()

    try:
        while True:
            result = subprocess.run(f"qstat -u $USER | grep {job_id}", shell=True, stdout=subprocess.PIPE)
            if not result.stdout:
                break
            time.sleep(15)
    except KeyboardInterrupt:
        exitfn()

    with open(f"{os.getcwd()}/exitcode", 'r') as f:
        exitcode = f.read().strip()

    assert exitcode == "0"

if __name__ == "__main__":
    main()