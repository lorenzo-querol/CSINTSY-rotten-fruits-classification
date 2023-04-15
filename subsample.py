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

# iterate over each split (train or test) in the dataset directory
for split_name in ['train', 'test']:
    split_dir = os.path.join(dataset_dir, split_name)
    if not os.path.isdir(split_dir):
        continue

    # create the corresponding split directory in the subsampled directory
    subsampled_split_dir = os.path.join(subsampled_dir, split_name)
    os.makedirs(subsampled_split_dir, exist_ok=True)

    # iterate over each class in the split directory
    for class_name in os.listdir(split_dir):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        # create the corresponding class directory in the subsampled split directory
        subsampled_class_dir = os.path.join(subsampled_split_dir, class_name)
        os.makedirs(subsampled_class_dir, exist_ok=True)

        # get the list of image filenames in the class directory
        image_filenames = os.listdir(class_dir)

        # calculate the number of images to subsample
        num_images = len(image_filenames)
        num_subsampled_images = int(num_images * subsample_rate)

        # subsample the images randomly
        subsampled_image_filenames = random.sample(
            image_filenames, num_subsampled_images)

        # copy, resize, and convert the subsampled images to the corresponding class directory in the subsampled split directory
        for filename in subsampled_image_filenames:
            src_path = os.path.join(class_dir, filename)

            # remove file extension
            filename = os.path.splitext(filename)[0]
            filename = filename + '.jpg'
            dst_path = os.path.join(subsampled_class_dir, filename)
            with Image.open(src_path) as img:
                # resize the image
                img = img.resize(target_size, resample=Image.BILINEAR)

                # convert RGBA to RGB
                if img.mode == 'RGBA':
                    img = img.convert('RGB')

                # save the image as JPEG
                img.save(dst_path, 'JPEG', quality=jpeg_quality)
