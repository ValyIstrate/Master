import cv2
import numpy as np
import random 


"""
Simple averaging (in 2 modes)
"""
def simple_average_grayscale(image_path: str):
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    gray_image = image.copy()
    
    for i in range(height):
        for j in range(width):
            r, g, b = image[i, j]
            gray_value = r/3 + g/3 + b/3
            gray_image[i, j] = [gray_value, gray_value, gray_value]
            
    gray_image_different_method = np.mean(image, axis=2).astype(np.uint8)
            
    cv2.imshow('Original Image', image)
    cv2.imshow('Grayscale Image (Per Pixel)', gray_image)
    cv2.imshow('Grayscale other method', gray_image_different_method)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
"""
Weighted average
"""
def weighted_average(image_path: str):
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    gray_image1 = image.copy()
    gray_image2 = image.copy()
    gray_image3 = image.copy()
    
    for i in range(height):
        for j in range(width):
            r, g, b = image[i, j]
            gv1 = r*0.3 + g*0.59 + b*0.11
            gv2 = r*0.2126 + g*0.7152 + b*0.0722
            gv3 = r*0.299 + g*0.587 + b*0.114
            gray_image1[i, j] = [gv1, gv1, gv1]
            gray_image2[i, j] = [gv2, gv2, gv2]
            gray_image3[i, j] = [gv3, gv3, gv3]
    
    cv2.imshow('Original Image', image)
    cv2.imshow('Grayscale Image F1', gray_image1)
    cv2.imshow('Grayscale Image F2', gray_image2)
    cv2.imshow('Grayscale Image F3', gray_image3)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
"""
Desaturation
"""
def desaturation(image_path: str):
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    gray_image = image.copy()
    
    for i in range(height):
        for j in range(width):
            r, g, b = image[i, j]
            gv = (min(r, g, b) + max(r, g, b)) / 2
            gray_image[i, j] = [gv, gv, gv]
    
    cv2.imshow('Original Image', image)
    cv2.imshow('Desaturated Image', gray_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
"""
Decomposition
"""
def decomposition(image_path: str):
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    gray_image_min = image.copy()
    gray_image_max = image.copy()
    
    for i in range(height):
        for j in range(width):
            r, g, b = image[i, j]
            gv_max = max(r, g, b)
            gv_min = min(r, g, b)
            gray_image_max[i, j] = [gv_max, gv_max, gv_max]
            gray_image_min[i, j] = [gv_min, gv_min, gv_min]
    
    cv2.imshow('Original Image', image)
    cv2.imshow('Decomposition Image - Max', gray_image_max)
    cv2.imshow('Decomposition Image - Min', gray_image_min)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
"""
Single Colour Channel
"""
def single_colour_channel(image_path: str):
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    gray_image_R = image.copy()
    gray_image_G = image.copy()
    gray_image_B = image.copy()
    
    for i in range(height):
        for j in range(width):
            r, g, b = image[i, j]
            gray_image_R[i, j] = [r, r, r]
            gray_image_G[i, j] = [g, g, g]
            gray_image_B[i, j] = [b, b, b]
            
    cv2.imshow('Original Image', image)
    cv2.imshow('Single Channel - R', gray_image_R)
    cv2.imshow('Single Channel - G', gray_image_G)    
    cv2.imshow('Single Channel - B', gray_image_B)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
"""
Custom Number of Grey Shades
"""
def custom_number_of_shades(image_path: str, shades: int, equal_intervals: bool):
    image = cv2.imread(image_path)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if equal_intervals:
        a = np.linspace(0, 255, shades + 1, dtype=int)
    else:
        a = sorted(random.sample(range(1, 255), shades - 1))
        a = [0] + a + [255]
        
    print("Intervals (a):", a)
    
    for i in range(1, shades + 1):
        avg_value = (a[i-1] + a[i]) // 2

        mask = (grayscale_image >= a[i-1]) & (grayscale_image < a[i])
        grayscale_image[mask] = avg_value
        
    cv2.imshow('Original Image', image)
    cv2.imshow('Grayscale', grayscale_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
"""
Custom numbers of grey shades - Floyd-Steinberg
"""
def floyd_steinberg_dither(image_path: str, shades: int):
    image = cv2.imread(image_path)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    height, width = grayscale_image.shape

    scale_factor = 255 / (shades - 1)
    
    for y in range(height):
        for x in range(width):
            old_pixel = grayscale_image[y, x]
            
            new_pixel = round(old_pixel / scale_factor) * scale_factor
            grayscale_image[y, x] = new_pixel
            
            error = old_pixel - new_pixel

            if x + 1 < width:
                grayscale_image[y, x + 1] = np.clip(grayscale_image[y, x + 1] + error * 7 / 16, 0, 255)
            if y + 1 < height and x - 1 >= 0:
                grayscale_image[y + 1, x - 1] = np.clip(grayscale_image[y + 1, x - 1] + error * 3 / 16, 0, 255)
            if y + 1 < height:
                grayscale_image[y + 1, x] = np.clip(grayscale_image[y + 1, x] + error * 5 / 16, 0, 255)
            if y + 1 < height and x + 1 < width:
                grayscale_image[y + 1, x + 1] = np.clip(grayscale_image[y + 1, x + 1] + error * 1 / 16, 0, 255)
                
    cv2.imshow('Original Image', image)
    cv2.imshow('Floyd-Steinberg', grayscale_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
"""
Custom number of grey shades - Stucki
"""    
def stucki_dither(image_path: str, num_shades: int):
    image = cv2.imread(image_path)
    
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = grayscale_image.shape

    scale_factor = 255 / (num_shades - 1)

    for y in range(height):
        for x in range(width):
            old_pixel = grayscale_image[y, x]
            
            new_pixel = round(old_pixel / scale_factor) * scale_factor
            grayscale_image[y, x] = new_pixel
            
            error = old_pixel - new_pixel

            # Right (x + 1)
            if x + 1 < width:
                grayscale_image[y, x + 1] = np.clip(grayscale_image[y, x + 1] + error * 8 / 42, 0, 255)
            # Right next (x + 2)
            if x + 2 < width:
                grayscale_image[y, x + 2] = np.clip(grayscale_image[y, x + 2] + error * 4 / 42, 0, 255)
            # Below row (y + 1)
            if y + 1 < height:
                if x - 2 >= 0:
                    grayscale_image[y + 1, x - 2] = np.clip(grayscale_image[y + 1, x - 2] + error * 2 / 42, 0, 255)
                if x - 1 >= 0:
                    grayscale_image[y + 1, x - 1] = np.clip(grayscale_image[y + 1, x - 1] + error * 4 / 42, 0, 255)
                grayscale_image[y + 1, x] = np.clip(grayscale_image[y + 1, x] + error * 8 / 42, 0, 255)
                if x + 1 < width:
                    grayscale_image[y + 1, x + 1] = np.clip(grayscale_image[y + 1, x + 1] + error * 4 / 42, 0, 255)
                if x + 2 < width:
                    grayscale_image[y + 1, x + 2] = np.clip(grayscale_image[y + 1, x + 2] + error * 2 / 42, 0, 255)
            # Below next row (y + 2)
            if y + 2 < height:
                if x - 2 >= 0:
                    grayscale_image[y + 2, x - 2] = np.clip(grayscale_image[y + 2, x - 2] + error * 1 / 42, 0, 255)
                if x - 1 >= 0:
                    grayscale_image[y + 2, x - 1] = np.clip(grayscale_image[y + 2, x - 1] + error * 2 / 42, 0, 255)
                grayscale_image[y + 2, x] = np.clip(grayscale_image[y + 2, x] + error * 4 / 42, 0, 255)
                if x + 1 < width:
                    grayscale_image[y + 2, x + 1] = np.clip(grayscale_image[y + 2, x + 1] + error * 2 / 42, 0, 255)
                if x + 2 < width:
                    grayscale_image[y + 2, x + 2] = np.clip(grayscale_image[y + 2, x + 2] + error * 1 / 42, 0, 255)
                    
    cv2.imshow('Original Image', image)
    cv2.imshow('Stucki', grayscale_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
"""
Convert grayscale to RGB
"""
def gray_to_rgb(image_path: str):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    height, width = gray_image.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            gv = gray_image[i, j]
            rgb_image[i, j] = [gv, gv, gv]
            
    cv2.imshow('Grayscale Image', gray_image)
    cv2.imshow('RGB Image', rgb_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
# simple_average_grayscale('E:\Master\ComputerVision\Week2\colored.jpg')
# weighted_average('E:\Master\ComputerVision\Week2\colored.jpg')
# desaturation('E:\Master\ComputerVision\Week2\colored.jpg')
# decomposition('E:\Master\ComputerVision\Week2\colored.jpg')
# single_colour_channel('E:\Master\ComputerVision\Week2\colored.jpg')
# custom_number_of_shades('E:\Master\ComputerVision\Week2\colored.jpg', 8, False)
# floyd_steinberg_dither('E:\Master\ComputerVision\Week2\colored.jpg', 8)
# stucki_dither('E:\Master\ComputerVision\Week2\colored.jpg', 8)
gray_to_rgb('E:\Master\ComputerVision\Week2\colored.jpg')
    