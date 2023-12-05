############################## Import required libraries ###################################################

from PIL import Image
import random
from torchvision import transforms
import numpy as np
import imgaug.augmenters as iaa
import numpy as np
from PIL import ImageFilter
from scipy.ndimage import affine_transform
from scipy.ndimage import zoom
import os
import random
from torchvision import transforms
from tqdm import tqdm
import math
from PIL import ImageEnhance
from scipy.ndimage import gaussian_filter
from scipy.ndimage import map_coordinates
aug = iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255))

############################## Augmentation functions #########################################################


def apply_random_crop(image, output_size=(224,224)):
    width, height = image.shape[1:]
    crop_width, crop_height = output_size
    left = random.randint(0, width - crop_width)
    top = random.randint(0, height - crop_height)
    right = left + crop_width
    bottom = top + crop_height
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image


def apply_color_jitter(image, brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1):
    color_jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    jittered_image = color_jitter(image)
    return jittered_image


def apply_gaussian_noise(image):
    image_array = np.array(image)
    noisy_image = aug.augment_image(image_array)
    return Image.fromarray(noisy_image)


def apply_blur(image, radius=2):
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=radius))
    return blurred_image


def apply_contrast_stretching(image):
    stretched_image = image  # Replace with actual implementation
    return stretched_image

def apply_brightness_adjustment(image, factor=1.1):
    enhanced_image = ImageEnhance.Brightness(image).enhance(factor)
    return enhanced_image


def apply_random_erasing(image, probability=0.5, sl=0.02, sh=0.4, r1=0.3, r2=1/0.3, v_l=0, v_h=255):
    image = np.array(image)
    if random.random() > probability:
        return Image.fromarray(image)
    
    c, h, w = image.shape
    S = h * w
    while True:
        S_e = S * random.uniform(sl, sh)
        r_e = random.uniform(r1, r2)
        H_e = int(round(math.sqrt(S_e * r_e)))
        W_e = int(round(math.sqrt(S_e / r_e)))
        x = random.randint(0, h)
        y = random.randint(0, w)
        if x + H_e <= h and y + W_e <= w:
            break

    image[x:x + H_e, y:y + W_e, :] = random.randint(v_l, v_h)
    return Image.fromarray(image)


def delete_square(img, pixels=800):
    """Delete random square from image"""
    if random.randint(0, 10) < 2:
        return img
    img = np.array(img)
    h, w, channels = np.shape(img)

    #Random starting pixel
    rh = random.randint(0, h)
    rw = random.randint(0, w)

    sub = round(pixels/2)
    add = pixels-sub

    #Boundries for square
    hmin = max(rh-sub//50, 0)
    hmax = min(rh+add//50, h-1)
    vmin = max(rw-sub, 0)
    vmax = min(rw+add, w-1)

    # Turn pixel within range black
    img[hmin:hmax, vmin:vmax] = np.array([np.random.randint(
        0, 1), np.random.randint(0, 1), np.random.randint(0, 1)])

    return img


######################## Augmenting function ##########################################################

# Specify your augmentation options
augmentation_options = {
    'random_crop': True,
    'color_jitter': True,
    'gaussian_noise': True,
    'blur': True,
    'contrast_stretching': True,
    'brightness_adjustment': True,
    'elastic_deformation': True,
    'random_erasing': True,
    'delete_square': True,
}

# Specify the target image count
target_image_count = 1000

# Function to apply augmentations
def apply_augmentations(image):
    augmentations = []

    # if augmentation_options['random_crop'] and random.random() < 1:
    #     augmentations.append(apply_random_crop)
    # if augmentation_options['color_jitter'] and random.random() < 0.5:
    #     augmentations.append(apply_color_jitter)


    if augmentation_options['gaussian_noise'] and random.random() < 0.5:
        augmentations.append(apply_gaussian_noise)
    if augmentation_options['blur'] and random.random() < 0.5:
        augmentations.append(apply_blur)
    # if augmentation_options['contrast_stretching'] and random.random() < 0.5:
    #     augmentations.append(apply_contrast_stretching)
    if augmentation_options['brightness_adjustment'] and random.random() < 0.5:
        augmentations.append(apply_brightness_adjustment)


    # if augmentation_options['elastic_deformation'] and random.random() < 1:
    #     augmentations.append(apply_elastic_deformation)
    # if augmentation_options['random_erasing'] and random.random() < 0.5:
    #     augmentations.append(apply_random_erasing)

    augmented_images = []
    for augmentation in augmentations:
        augmented_image = delete_square(augmentation(image))
        pil_image = Image.fromarray(np.uint8(augmented_image))
        augmented_images.append(pil_image)

    return augmented_images

############################## Our Main function ################################################


# Set the path to the 'data_aug' folder
data_aug_folder = 'C:\\Users\\krish\\Downloads\\flowers_dataset\\data_aug'

# Iterate through the 'data_aug' folder and augment the existing images
for folder in os.listdir(data_aug_folder):
    folder_path = os.path.join(data_aug_folder, folder)
    if os.path.isdir(folder_path):
        num_images = len(os.listdir(folder_path))
        if num_images >= target_image_count:
            continue  # Skip this folder if it has enough images
        j = 0
        while len(os.listdir(folder_path)) < target_image_count:
            for image_file in os.listdir(folder_path):
                if image_file.split('_')[-1] != 'aug.png':
                # if not is_augmented:
                    image_path = os.path.join(folder_path, image_file)
                    image = Image.open(image_path)
                    augmented_images = apply_augmentations(image)
                    if len(os.listdir(folder_path)) >= target_image_count:
                        continue  # Skip this folder if it has enough images
                    # Save augmented images to the 'data_aug' folder
                    for i, augmented_image in enumerate(augmented_images):
                        image_file  = image_file.split('.')[0]
                        save_path = os.path.join(data_aug_folder, folder, f'{image_file}_{i}{j}_aug.png')
                        augmented_image.save(save_path)
            j+=1
print("Images are augmented")
