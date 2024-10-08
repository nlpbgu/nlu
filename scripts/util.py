from typing import *
import os
import shutil
import requests

def read_dict(f: str) -> Dict[str, str]:
    d = {}
    for l in open(f, 'r'):
        sid, sentence = l.strip().split("\t")
        d[canonicalize(sentence)] = sid
    return d


def read_qrels(f: str) -> Dict[Tuple[str, str], str]:
    d = {}
    for l in open(f, 'r'):
        pid, _, hid, cl = l.strip().split("\t")
        d[(pid, hid)] = cl
    return d


def canonicalize(s: str):
    return "".join(filter(lambda c: c.isalnum(), s))


def copy_files_to_new_directory(existing_dir, new_dir):

    # Create the new directory if it doesn't exist
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    else:
        # If the directory exists, clear its contents
        for filename in os.listdir(new_dir):
            file_path = os.path.join(new_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')


    # List all files in the existing directory
    files = os.listdir(existing_dir)

    # Move each file to the new directory
    for file_name in files:
        # Construct full file path
        old_file_path = os.path.join(existing_dir, file_name)
        new_file_path = os.path.join(new_dir, file_name)

        if os.path.isfile(old_file_path):
            # Move the file
            shutil.copy(old_file_path, new_file_path)
            print(f"Copied: {file_name} -> {new_file_path}")


def add_row_to_qrels(file_path, data , iter_str = 'ITER'):
    with open(file_path, 'a') as file:
        for row in data:
            new_row = f"{row['key_l']}\t{iter_str}\t{row['new_key_r']}\t{row['y']}\n"
            file.write(new_row)


def add_row_to_l(file_path, data):
    with open(file_path, 'a') as file:
        for row in data:
            new_row = f"{row['new_key_l']}\t{row['l']}\n"
            file.write(new_row)

def add_row_to_r(file_path, data):
    with open(file_path, 'a') as file:
        for row in data:
            new_row = f"{row['new_key_r']}\t{row['new_r']}\n"
            file.write(new_row)



def download_files(url, save_dir):

    files = ['.gitattributes','README.md','added_tokens.json','config.json','merges.txt','pytorch_model.bin','special_tokens_map.json','tokenizer_config.json','vocab.json']


    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for file_name in files:

        file_path = os.path.join(save_dir, file_name)

        if os.path.exists(file_path):
            print(f"File '{file_name}' already exists in '{save_dir}'. Skipping download.")
            continue

        # Download the file
        print(f"Downloading '{file_name}'...")
        response = requests.get(os.path.join(url,file_name))

        with open(file_path, "wb") as file:
            file.write(response.content)

        print(f"File '{file_name}' downloaded and saved in '{save_dir}'.")

def remove_dir(save_dir):

    if os.path.exists(save_dir):
        # Remove the directory and all its contents
        shutil.rmtree(save_dir)
        print(f"Directory '{save_dir}' has been removed.")
