from pathlib import Path
from argparse import ArgumentParser
import shutil

parser = ArgumentParser()
parser.add_argument('root_dir', type=str)
parser.add_argument('exclude_items', nargs='+', type=str)
args = parser.parse_args()

root_dir = Path(args.root_dir)
exclude_items = args.exclude_items
delete_items = [item for item in root_dir.glob('*') if item.name not in exclude_items]

print('The following items will be deleted from {root_dir}:')
for item in delete_items:
    print(item)
if input("Is this OK? (type 'y' if OK)").lower() == 'y':
    for item in delete_items:
        shutil.rmtree(item)
