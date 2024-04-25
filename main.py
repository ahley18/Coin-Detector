import numpy as np
from skimage import color, exposure
from skimage.restoration import denoise_tv_chambolle
from skimage.filters import sobel
from skimage.feature import canny
import matplotlib.pyplot as plt
import cv2 as cv

# Load an example image
read_img = cv.imread(cv.samples.findFile(r"files\coins3.jpg"))
image = exposure.equalize_adapthist(read_img, clip_limit = 5)
# Convert the image to grayscale
image_gray = color.rgb2gray(image)

#other image processing
denoised_image = denoise_tv_chambolle(image_gray, weight = 5)
canny_image = canny(denoised_image, sigma = 4)
sobel_image = sobel(denoised_image)
cv.imwrite(r'files\canny_image.jpg', (canny_image * 255).astype(np.uint8))

#count contours
img = cv.imread(r'files\canny_image.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, cv.THRESH_BINARY)
contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(img, contours, -1, (0,255,0), 3)

count = 0
# Iterate through contours
for contour in contours:
    # Approximate the contour to reduce the number of vertices
    #peri = cv.arcLength(contour, True)
    epsilon = 0.03 * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)

    # Get the number of vertices
    num_vertices = len(approx)

    # Determine the shape based on the number of vertices
    shape = "Unknown"
    if num_vertices >= 8:
        shape = "Coin"
        count += 1
    else:
        shape = "Not a coin"
    # Draw the shape name on the image
    cv.putText(img, shape, (approx.ravel()[0], approx.ravel()[1]), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 10)
coin_count = 'Coin count: ' + str(count)
cv.putText(img, coin_count, (500,500), 1, 6,(255,255,255),4,cv.LINE_AA)
# plotting
cv.imwrite(r'files\Coin_Detection.jpg', (img).astype(np.uint8))
#plt.imshow(canny_image, cmap = 'gray')

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 6), sharex=True, sharey=True)
ax1.imshow(read_img, cmap = 'Accent')
ax1.set_title('Original Image')
ax2.imshow(image_gray, cmap='gray')
ax2.set_title('Grayscale Image')
ax3.imshow(canny_image, cmap='gray')
ax3.set_title('Canny Edge Detection')
ax4.imshow(img)
ax4.set_title('Coin Detection')

#save figure
fig.savefig(r'files\preprocessing.jpg')
plt.show()