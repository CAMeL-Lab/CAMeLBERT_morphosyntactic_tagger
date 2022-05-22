import re
import sys
import shutil
from pathlib import Path

RE_ACCURACY = re.compile(u'(eval_accuracy = )(\d+(?:\.\d+)?)')

eval_results = []
for path in Path(sys.argv[1]).rglob('eval_results.txt'):
    if 'checkpoint-best' in str(path):
        sys.exit('Best checkpoint already chosen! Exit')
    with open(path, mode='r') as f:
        matched = RE_ACCURACY.search(f.read())
        if matched:
            acc = float(matched[2])
            eval_results.append((path, acc))
        else:
            print('No match! Something is wrong.')

max_so_far_path = None
max_so_far_acc = 0
for result in eval_results:
    if max_so_far_acc <= float(result[1]):
        max_so_far_acc = float(result[1])
        max_so_far_path = result[0]

source = str(max_so_far_path.parent)
target = str(max_so_far_path.parent.parent.joinpath('checkpoint-best'))

print(f'cp -r {source} {target}')

