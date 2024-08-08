#!/usr/bin/env python3

"""
6.101 Lab:
Image Processing
"""

import math

from PIL import Image

# NO ADDITIONAL IMPORTS ALLOWED!


def get_pixel(image, row, col):
    """
    Given row index and column index, this function returns the pixel from 
    image["pixels"] based on the boundary behavior
    """
    index = row * image["width"] + col              ##
    return image["pixels"][index]                   ##
    
    # return image["pixels"][col, row]


def set_pixel(image, row, col, color):
    index = row * image["width"] + col
    image["pixels"][index] = color


def apply_per_pixel(image, func):
    """
    Apply the given function to each pixel in an image and return the resulting image.

    Parameters:
    image (dict): A dictionary representing an image, with keys:
                    - "height" (int): The height of the image.
                    - "width" (int): The width of the image.
                    - "pixels" (list): A list of pixel values.
    func (callable): A function that takes a single pixel value and returns a new pixel value.

    Returns:
    dict: A new image dictionary with the same dimensions as the input image, but with each
            pixel value modified by the given function.
    """

    new_image = {
        "height": image["height"],
        "width": image["width"],
        "pixels": [0] * (image["height"] * image["width"])  # Initialize with the correct size
    }

    # Apply the function to each pixel in the input image
    for row in range(image["height"]):
        for col in range(image["width"]):
            new_pixel = func(get_pixel(image, row, col))
            set_pixel(new_image, row, col, new_pixel)
            
            # new_image["pixels"].append(func(get_pixel(image, row, col)))

    return new_image
    raise NotImplementedError


def inverted(image):
    return apply_per_pixel(image, lambda color: 255-color)


# HELPER FUNCTIONS

def correlate(image, kernel, boundary_behavior):
    """
    Compute the result of correlating the given image with the given kernel.
    `boundary_behavior` will one of the strings "zero", "extend", or "wrap",
    and this function will treat out-of-bounds pixels as having the value zero,
    the value of the nearest edge, or the value wrapped around the other edge
    of the image, respectively.

    if boundary_behavior is not one of "zero", "extend", or "wrap", return
    None.

    Otherwise, the output of this function should have the same form as a 6.101
    image (a dictionary with "height", "width", and "pixels" keys), but its
    pixel values do not necessarily need to be in the range [0,255], nor do
    they need to be integers (they should not be clipped or rounded at all).

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.

    DESCRIBE YOUR KERNEL REPRESENTATION HERE
    """
    if boundary_behavior not in {"zero", "extend", "wrap"}:
        return None

    kernel_height = len(kernel)
    kernel_width = len(kernel[0])
    image_height = image["height"]
    image_width = image["width"]


    def get_pixel_zero(row, col):
        if 0 <= row < image_height and 0 <= col < image_width:
            return get_pixel(image, row, col)
        return 0
    
    def get_pixel_wrap(row, col):
        row = row % image_height
        col = col % image_width
        return get_pixel(image, row, col)
    
    def get_pixel_extended(row, col):
        if row < 0:
            row = 0
        elif row >= image_height:
            row = image_height - 1
        if col < 0:
            col = 0
        elif col >= image_width:
            col = image_width - 1
        return get_pixel(image, row, col)

    if boundary_behavior == "zero":
        get_pixel_func = get_pixel_zero
    elif boundary_behavior == "wrap":
        get_pixel_func = get_pixel_wrap
    elif boundary_behavior == "extend":
        get_pixel_func = get_pixel_extended

    output = {
        "height": image_height,
        "width": image_width,
        "pixels": [0] * (image_height * image_width),
    }

    kernel_center_y = kernel_height // 2
    kernel_center_x = kernel_width // 2

    for y in range(image_height):
        for x in range(image_width):
            new_value = 0
            for ky in range(kernel_height):
                for kx in range(kernel_width):
                    pixel_y = y + ky - kernel_center_y
                    pixel_x = x + kx - kernel_center_x
                    new_value += kernel[ky][kx] * get_pixel_func(pixel_y, pixel_x)
            set_pixel(output, y, x, new_value)

    return output
    raise NotImplementedError


def round_and_clip_image(image):
    """
    Given a dictionary, ensure that the values in the "pixels" list are all
    integers in the range [0, 255].

    All values should be converted to integers using Python's `round` function.

    Any locations with values higher than 255 in the input should have value
    255 in the output; and any locations with values lower than 0 in the input
    should have value 0 in the output.
    """
    rounded_clipped_pixels = []

    for pixel in image["pixels"]:
        rounded_pixel = round(pixel)
        clipped_pixel = max(0, min(255, rounded_pixel))
        rounded_clipped_pixels.append(clipped_pixel)

    return {
        "height": image["height"],
        "width": image["width"],
        "pixels": rounded_clipped_pixels
    }
    raise NotImplementedError

def box_blur_kernel(n):
    return [[1 / (n * n) for _ in range(n)] for _ in range(n)]


# FILTERS
    
def blurred(image, kernel_size):
    """
    Return a new image representing the result of applying a box blur (with the
    given kernel size) to the given input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    """
    # first, create a representation for the appropriate n-by-n kernel (you may
    # wish to define another helper function for this)

    # then compute the correlation of the input image with that kernel

    # and, finally, make sure that the output is a valid image (using the
    # helper function from above) before returning it.

    kernel = box_blur_kernel(kernel_size)
    return round_and_clip_image(correlate(image, kernel, boundary_behavior="extend"))
    raise NotImplementedError

def sharpened(image, n):
    """
    Return a new image which is sharper than the input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    Output list should be clipped.
    """
    raise NotImplementedError

def edges(image):
    """
    Return a new image with all the edges distincly and clearly detectable.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    Output list should be clipped.
    """
    raise NotImplementedError

def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns a dictionary
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image("test_images/cat.png")
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith("RGB"):
            pixels = [round(.299 * p[0] + .587 * p[1] + .114 * p[2])
                      for p in img_data]
        elif img.mode == "LA":
            pixels = [p[0] for p in img_data]
        elif img.mode == "L":
            pixels = list(img_data)
        else:
            raise ValueError(f"Unsupported image mode: {img.mode}")
        width, height = img.size
        return {"height": height, "width": width, "pixels": pixels}


def save_greyscale_image(image, filename, mode="PNG"):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the "mode" parameter.
    """
    out = Image.new(mode="L", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


if __name__ == "__main__":
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place for generating images, etc.
    
    ''' # Self-Trials
    dict1_pixel = load_greyscale_image("test_images/centered_pixel.png")
    dict1_pattern = load_greyscale_image("test_images/pattern.png")
    dict1_cat = load_greyscale_image("test_images/cat.png")
    # print(i)

    # save_greyscale_image(dict1_cat, "mytests/cat1.png")

    print(inverted(dict1_pattern))
    print()

    kernel_identity = [[0,0,0],[0,1,0],[0,0,0]]
    kernel_translation = [[0,0,0,0,0],[0,0,0,0,0],[1,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
    kernel_average = [[0., 0.2, 0.], [0.2, 0.2, 0.2], [0., 0.2, 0.]]                   # average of the 5 nearest pixels of the input image


    print(correlate(inverted(dict1_pattern), kernel_translation, "extend"))

    '''
    
    dict_test1 = load_greyscale_image("test_images/pigbird.png")
    kernel_test1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ]
    
    d1 = round_and_clip_image(correlate(dict_test1, kernel_test1, "extend"))
    d2 = round_and_clip_image(correlate(dict_test1, kernel_test1, "zero"))
    d3 = round_and_clip_image(correlate(dict_test1, kernel_test1, "wrap"))

    # save_greyscale_image(d1, "mytests/pigbird_extend.png")
    # save_greyscale_image(d2, "mytests/pigbird_zero.png")
    # save_greyscale_image(d3, "mytests/pigbird_wrap.png")

    dict_test2 = load_greyscale_image("test_images/cat.png")
    d4 = blurred(dict_test2, 13)
    # save_greyscale_image(d4, "mytests/cat_blurred_extend.png")
    # save_greyscale_image(d4, "mytests/cat_blurred_wrap.png")
    # save_greyscale_image(d4, "mytests/cat_blurred_zero.png")

    
    # pass
