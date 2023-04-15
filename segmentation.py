import os
import cv2
import numpy as np
from alive_progress import alive_bar
import matplotlib.pyplot as plt

def segment_image(image):
    rgb_image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)

    # Convert to grayscale
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    # Blur the image
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply Otsu thresholding to image
    ret, thresh = cv2.threshold(
        blur_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply mask on image
    masked_image = cv2.bitwise_and(rgb_image, rgb_image, mask=thresh)

    plt.imshow('image', masked_image)
    plt.show()


main_dir = 'fruits_dataset'
train_dir = f'{main_dir}/train'
test_image = f'{train_dir}/freshapples/rotated_by_15_Screen Shot 2018-06-08 at 4.59.36 PM.png'


segment_image(test_image)


def plot_examples(images):
    fig, axs = plt.subplots(6, 5)
    fig.set_size_inches(15, 15)
    for i in range(6):
        for j in range(5):
            axs[i, j].imshow(images[i*5+j])
            axs[i, j].axis('off')
    plt.show()
    