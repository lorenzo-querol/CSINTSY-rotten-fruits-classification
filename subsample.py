import os
import shutil
import random
from PIL import Image

dataset_dir = 'fruits_dataset'
subsampled_dir = 'subsampled_fruits_dataset'
subsample_rate = 0.25
jpeg_quality = 90
target_size = (224, 224)

shutil.rmtree(subsampled_dir, ignore_errors=True)

for split_name in ['train', 'test']:
    split_dir = os.path.join(dataset_dir, split_name)
    if not os.path.isdir(split_dir):
        continue

    subsampled_split_dir = os.path.join(subsampled_dir, split_name)
    os.makedirs(subsampled_split_dir, exist_ok=True)

    for class_name in os.listdir(split_dir):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        subsampled_class_dir = os.path.join(subsampled_split_dir, class_name)
        os.makedirs(subsampled_class_dir, exist_ok=True)

        image_filenames = os.listdir(class_dir)

        num_images = len(image_filenames)
        num_subsampled_images = int(num_images * subsample_rate)

        subsampled_image_filenames = random.sample(
            image_filenames, num_subsampled_images)

        for filename in subsampled_image_filenames:
            src_path = os.path.join(class_dir, filename)

            filename = os.path.splitext(filename)[0]
            filename = filename + '.jpg'
            dst_path = os.path.join(subsampled_class_dir, filename)
            with Image.open(src_path) as img:
                img = img.resize(target_size, resample=Image.BILINEAR)

                if img.mode == 'RGBA':
                    img = img.convert('RGB')

                img.save(dst_path, 'JPEG', quality=jpeg_quality)
