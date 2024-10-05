import cv2
import numpy as np

"""
1. Open an image (lena.tif), display its size, plot/write the image
"""
def display_image_data(image_path: str): 
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    (height, width) = image.shape[:2]
    print(f"Image width: {width} pixels")
    print(f"Image height: {height} pixels")

    cv2.imshow('Lena', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
2. Apply filters that blur/sharpen the image (https://learnopencv.com/image-filtering-usingconvolution-in-opencv/). Test these functions with at least 2 values for the parameters
(when the function has at least one parameter). Plot the results (or save the images)
"""
def apply_filters_on_image(image_path: str):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    blur_kernel = np.ones((5, 5), np.float32) / 25
    sharpen_kernel = np.array([[-1,-1,-1], 
                               [-1, 9,-1],
                               [-1,-1,-1]])
    
    cv2.imshow('Normal', image)
    
    blurred_image = cv2.filter2D(src=image, ddepth=-1, kernel=blur_kernel)
    cv2.imshow('Blurred', blurred_image)

    sharpened_image = cv2.filter2D(src=image, ddepth=-1, kernel=sharpen_kernel)
    cv2.imshow('Sharpened', sharpened_image)    
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
"""
3. Apply the following filter: [[0, -2, 0], [], [-2, 8, -2], [0, -2, 0]]
"""
def apply_custom_filter(image_path: str):
    image = cv2.imread(image_path)
    
    custom_kernel = np.array([[0, -2, 0], 
                              [-2, 8, -2], 
                              [0, -2, 0]])
    
    cv2.imshow('Normal', image)
    
    modified_image = cv2.filter2D(src=image, ddepth=-1, kernel=custom_kernel)
    cv2.imshow('Modified', modified_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

"""
4. Rotate an image using different angles, clockwise and counterclockwise. How can an
image rotation function be implemented?
"""
def rotate_image(image_path: str):
    image = cv2.imread(image_path)
    
    cv2.imshow('Normal', image)
    
    rotated_90_clockwise = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow('90 clockwise', rotated_90_clockwise)
    
    rotated_90_counter_clockwise = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imshow('90 counter-clockwise', rotated_90_counter_clockwise)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
"""
5. Write a function that crops a rectangular part of an image. The parameters of this function
are the position of the upper, left pixel in the image, where the cropping starts, the width
and the length of the rectangle.
"""
def crop_image(image_path: str,
               starting_point_coordinates: tuple[int, int],
               rectangle_width: int,
               rectangle_length: int):
    image = cv2.imread(image_path)
    cv2.imshow('Normal', image)
    
    print(image.shape[:2])
    
    cropped_image = image[starting_point_coordinates[0]: starting_point_coordinates[0] + rectangle_length,
                          starting_point_coordinates[1]: starting_point_coordinates[1] + rectangle_width]
    cv2.imshow('Cropped Image', cropped_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
"""
6. Create an emoticon image (emoji) using OpenCV functions. Include this image in the
archive that you’ll send at the end of the semester (save it as your_name.jpg)
"""
def create_emoticon():
    # Create white 'canvas'
    img = np.ones((500, 500, 3), dtype="uint8") * 255
    
    # Draw a yellow circle (the face) in the center
    cv2.circle(img, (250, 250), 200, (0, 255, 255), -1) 
    
    # Draw 2 black circles (eyes)
    cv2.circle(img, (175, 175), 30, (0, 0, 0), -1)  # Left eye
    cv2.circle(img, (325, 175), 30, (0, 0, 0), -1)  # Right eye
    
    # Draw 2 rectangles around the eyes (glasses) and the bridge between the lenses
    cv2.rectangle(img, (135, 135), (215, 215), (0, 0, 0), 5)  # Left lens
    cv2.rectangle(img, (285, 135), (365, 215), (0, 0, 0), 5)  # Right lens
    cv2.line(img, (215, 175), (285, 175), (0, 0, 0), 5)
    
    # Draw a line (nose)
    cv2.line(img, (250, 225), (250, 275), (0, 0, 0), 5)
    
    # Draw an arc (the smile)
    cv2.ellipse(img, (250, 300), (100, 50), 0, 0, 180, (0, 0, 0), 5)
    
    cv2.imwrite('Valentin_Istrate_emoji.jpg', img)
    
    cv2.imshow("My emoji", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

file_path: str = "E:\Master\ComputerVision\Week1\lena.tif"
# display_image_data("E:\Master\ComputerVision\Week1\lena.tif")
# apply_filters_on_image(file_path)
# apply_custom_filter(file_path)
# rotate_image(file_path)
# crop_image(file_path, (0, 100), 256, 256)
# create_emoticon()
