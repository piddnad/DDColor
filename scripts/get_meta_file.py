from tqdm import tqdm
import argparse
import os


def find_and_save_images(data_path, output_name):
    all_img_path = []

    for root, dirs, files in os.walk(data_path):
        for ext in ['png', 'jpg', 'jpeg', 'JPG', 'JPEG', 'PNG']:
            all_img_path += [os.path.join(root, file) for file in files if file.lower().endswith(f'.{ext}')]

    with open(output_name, 'w') as f:
        for path in tqdm(all_img_path):
            f.write(path + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-name', type=str, required=True, default='dataset_name.txt', help='Path output file.')
    parser.add_argument('--data-path', type=str, required=True, default='./', help='Path to dataset')

    args = parser.parse_args()

    print(f'Generating {args.output_name} from {args.data_path} ...')

    find_and_save_images(args.data_path, args.output_name)

    print('Done.')
