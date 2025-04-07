import argparse
import matplotlib.pyplot as plt
import tifffile
from skimage import measure, draw
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pkg.plot import plot_ion_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot borders of a binary image')
    parser.add_argument('bin_img', type=str, help='binary image file')
    parser.add_argument('out_img', type=str, help='output image file with borders')
    parser.add_argument('-imzML_fl', type=str, default='', help='define imzML filename to plot ion image together with border image')
    parser.add_argument('-mz', type=float, default='', help='set m/z value for ion image')
    parser.add_argument('-plot', type=bool, default=False, help='set to true if output should be plotted')
    args = parser.parse_args()

    bin_img = tifffile.imread(args.bin_img)

    # Find contours
    contours = measure.find_contours(bin_img, level=0.8)

    # Create an empty image for saving borders
    border_image = np.zeros_like(bin_img, dtype=np.uint8)

    # Draw contours on the border image
    for contour in contours:
        rr, cc = draw.polygon_perimeter(contour[:, 0], contour[:, 1], shape=border_image.shape)
        border_image[rr, cc] = 255  # Set contour pixels to white

    # Save the border image as a TIFF file
    tifffile.imwrite(args.out_img, border_image.astype(np.uint8))

    # Display the result
    if args.plot:
        if args.imzML_fl != '' and args.mz != '':
            # Plot ion image with contours
            ion_img = plot_ion_image(args.mz, args.imzML_fl, tol=0.01, unit='Da', pyimzml=False, CLAHE=False,
                                     return_img=True)

            fig, ax = plt.subplots(figsize=(8, 6))
            #print(ion_img.shape)
            #print(ion_img.min(), ion_img.max())
            im = ax.imshow(ion_img.astype(np.float32), cmap='inferno', interpolation='nearest')

            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Intensity")

            # Draw contours
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], 'w-', linewidth=2)  # Red outline

            ax.set_title(str(args.mz) + ' m/z ± 0.01 Da')
            ax.axis("off")
            plt.show()

        else:
            plt.figure(figsize=(8, 6))
            plt.imshow(border_image, cmap="gray")
            plt.title("border image")
            plt.axis("off")
            plt.show()