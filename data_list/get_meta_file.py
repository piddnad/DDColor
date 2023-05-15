import glob
from tqdm import tqdm

output_name = 'dataset_name.txt'
data_path = '/path/to/dataset'

print(f'Generating {output_name} from {data_path} ...')

all_img_path = []
for ext in [ 'png', 'jpg', 'jpeg', 'JPG', 'JPEG', 'PNG']:
    all_img_path += list(glob.glob(f'{data_path}/*.{ext}'))

with open(output_name, 'w') as f:
    for path in tqdm(all_img_path):
        f.write(path + '\n')

print('Done.')