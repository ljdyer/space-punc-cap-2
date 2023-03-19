import argparse
import pandas as pd
import more_itertools
import socket
from pathlib import Path
from datetime import datetime
from helper import tabulate_list_of_dicts, get_dict_by_value

parser = argparse.ArgumentParser(
    description='Show test results for feature restoration models',
    allow_abbrev=False
)
parser.add_argument(
    '--outputsdir', '-o', type=str, default='outputs',
    help='The folder where the model outputs are stored'
)


# ====================
def show_test_results(outputsdir):

    all_files = []
    outputsdir = Path(outputsdir)
    for training_folder in outputsdir.glob('*'):
        for model_folder in training_folder.glob('*'):
            results_files = [f for f in model_folder.glob('*.csv') if 'metrics' in f.name]
            for f in results_files:
                all_files.append({
                    'training': training_folder.name,
                    'model': model_folder.name,
                    'fname': f.name
                })
    all_files = [{'option': i, **m} for i, m in enumerate(all_files)]
    print(tabulate_list_of_dicts(all_files))
    choice = int(input('Which file do you wish to display? '))
    chosen_file = get_dict_by_value(all_files, 'option', choice)
    path = Path(outputsdir) / Path(chosen_file['training']) / Path(chosen_file['model']) / Path(chosen_file['fname'])
    print(pd.read_csv(path))


args = parser.parse_args()
outputsdir = args.outputsdir
show_test_results(outputsdir)