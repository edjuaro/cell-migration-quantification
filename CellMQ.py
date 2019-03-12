import cv2
import numpy as np
from skimage import draw
from skimage import io
 
# Read image
im_in = cv2.imread("analyses/MDA231_stopper_1_c3.tif", cv2.IMREAD_GRAYSCALE);
 
# Threshold.
# Set values equal to or above 220 to 0.
# Set values below 220 to 255.
 
th, im_th = cv2.threshold(im_in, 20, 255, cv2.THRESH_BINARY_INV);
 
# Copy the thresholded image.
im_floodfill = im_th.copy()
 
# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = im_th.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
 
# Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (0,0), 255);
 
# Invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(im_floodfill)
 
# Combine the two images to get the foreground.
im_out = im_th | im_floodfill_inv
io.imsave(fname='temp_output.png', arr=im_out)

# im_out_inv = cv2.bitwise_not(im_out)


# dilate the mask:
k_size = 2
k_half = k_size/2
kernel = np.ones((k_size,k_size),np.uint8)
coords = draw.circle(k_half, k_half, k_half, shape=im_th.shape)
kernel[coords] = 1 
erosion = cv2.erode(im_out,kernel,iterations = 1)
dilation = cv2.dilate(cv2.bitwise_not(erosion),kernel,iterations = 1)
dilation = cv2.bitwise_not(dilation)

# io.imshow(dilation)
io.imsave(fname='mask.png', arr=dilation)
 
# Display images.
# io.imsave(fname='mask.png', arr=im_out)

# # mostly from http://nickc1.github.io/python,/matlab/2016/05/17/Standard-Deviation-(Filters)-in-Matlab-and-Python.html

# import cv2
# from skimage import draw

# from skimage import io
# filename = 'analyses/MDA231_stopper_1_c3.tif'
# plate = io.imread(filename,as_grey=True)
# image = plate
# #io.imshow(image)
# # io.imsave(fname='temp_output.png', arr=image)



# import numpy as np

# # img = cv2.imread('....') # Read in the image
# sobelx = cv2.Sobel(image,cv2.CV_64F,1,0) # Find x and y gradients
# sobely = cv2.Sobel(image,cv2.CV_64F,0,1)
# # Find magnitude and angle
# I2 = np.sqrt(sobelx**2.0 + sobely**2.0)
# # angle = np.arctan2(sobely, sobelx) * (180 / np.pi)
# # io.imshow(I2)
# # io.imsave(fname='temp_output.png', arr=I2)


# from scipy.ndimage.filters import uniform_filter
# import numpy as np

# def window_stdev(X, window_size):
    # c1 = uniform_filter(X, window_size, mode='reflect')
    # c2 = uniform_filter(X*X, window_size, mode='reflect')
    # return np.sqrt(c2 - c1*c1)

# # x = np.arange(16).reshape(4,4).astype('float')
# kernel_size = 3
# I1 = window_stdev(I2,kernel_size)*np.sqrt(kernel_size**2/(kernel_size**2 - 1))

# # io.imshow(I1)
# # io.imsave(fname='temp_output.png', arr=I1)


# from scipy.signal import medfilt2d
# I1 = medfilt2d(I1, kernel_size=3)
# # io.imshow(I1)
# # io.imsave(fname='temp_output.png', arr=I1)


# import numpy as np
# from skimage.morphology import reconstruction
# from skimage.exposure import rescale_intensity

# # image = rescale_intensity(I1, in_range=(50, 200))
# image = I1
# seed = np.copy(image)
# seed[1:-1, 1:-1] = image.max()
# mask = image

# filled = reconstruction(seed, mask, method='erosion')
# io.imsave(fname='temp_output.png', arr=filled)




# # kernel = np.zeros((80,80),np.uint8)
# # coords = draw.circle(40, 40, 40, shape=image.shape)
# # kernel[coords] = 1 
# # erosion = cv2.erode(I1,kernel,iterations = 1)
# # # io.imshow(erosion)
# # # # kernel = np.ones((40,40),np.uint8)
# # # # erosion = cv2.erode(I1,kernel,iterations = 1)
# # # # io.imshow(erosion)
# # # io.imsave(fname='temp_output.png', arr=erosion)


# # from skimage.morphology import reconstruction
# # fill = reconstruction(I1, erosion, method='erosion')
# # # io.imshow(fill)
# # # io.imsave(fname='temp_output.png', arr=fill)

# # dilation = cv2.dilate(fill,kernel,iterations = 1)
# # # io.imshow(dilation)
# # io.imsave(fname='temp_output.png', arr=dilation)