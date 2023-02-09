

from training.dataset import get_simplegrab_dataset
from pathlib import Path

tfrecords_path = Path("data_toolbox/output_6.12.22/")
binvox_filepaths = list(tfrecords_path.rglob('*.tfrecords'))
dataset = get_simplegrab_dataset(binvox_filepaths)
#dataset = iter(dataset)
count = 0
for i in dataset:
    print(i[0][0].shape)
    count += 1
