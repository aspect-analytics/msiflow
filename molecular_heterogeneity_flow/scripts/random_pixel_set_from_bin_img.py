import numpy as np
import tifffile as tiff
import os
import argparse

def select_random_positive_pixels(input_path, output_path, n_pixels):
    # Load the binary image
    image = tiff.imread(input_path)

    # Get coordinates of positive pixels
    positive_coords = np.column_stack(np.nonzero(image))
    total_positive = len(positive_coords)

    if n_pixels > total_positive:
        raise ValueError(f"Requested {n_pixels} pixels, but only {total_positive} positive pixels available.")

    # Randomly choose n pixels
    selected_indices = np.random.choice(total_positive, size=n_pixels, replace=False)
    selected_coords = positive_coords[selected_indices]

    # Create new binary image with same shape
    new_image = np.zeros_like(image, dtype=np.uint8)

    # Set selected pixels to 1
    for y, x in selected_coords:
        new_image[y, x] = 255

    # Save the result
    tiff.imwrite(output_path, new_image)
    print(f"New image saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Randomly select a set of pixels from binary image')
    parser.add_argument('input_img', type=str, help='binary input image as tif')
    parser.add_argument('output_img', type=str, help='binary output image as tif')
    parser.add_argument('n_pixels_to_select', type=int, help='number of pixels to select randomly from input image')
    args = parser.parse_args()

    select_random_positive_pixels(args.input_img, args.output_img, args.n_pixels_to_select)
