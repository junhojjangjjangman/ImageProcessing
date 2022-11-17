import cv2
import numpy as np
import scipy as sp

import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.ndimage as nd

# cv2의 imread 함수로 example.png 파일을 gray 이미지로 읽어 오세요.
# 변수는 gray_img
dir = r'C:/Users/15/Desktop/DataSet/example.png'
gray_img = cv2.imread(r'C:/Users/15/Desktop/DataSet/example.png', cv2.IMREAD_GRAYSCALE)
plt.imshow(gray_img, cmap="gray"); plt.axis("off"); plt.title("Grayscale Buzz");
plt.show()
# cv2의 imread 함수로 example.png 파일을 컬러 이미지로 읽어 오세요.
# 변수는 color_img

color_img = cv2.imread(r'C:/Users/15/Desktop/DataSet/example.png', cv2.IMREAD_COLOR)
plt.imshow(color_img, cmap="gray"); plt.axis("off"); plt.title("COLOR Buzz");
plt.show()

channels = blue, green, red = np.moveaxis(color_img, 2, 0)
plt.figure(figsize=(12, 4))
plt.subplot(131); plt.imshow(channels[0], cmap="gray"); plt.axis("off"); plt.title("Blue Channel");
plt.subplot(132); plt.imshow(channels[1], cmap="gray"); plt.axis("off"); plt.title("Green Channel");
plt.subplot(133); plt.imshow(channels[2], cmap="gray"); plt.axis("off"); plt.title("Red Channel");
plt.show()
# cv2의 cvtColor 함수를 이용하여 수정된 이미지(RGB채널 순서가 올바른)를 읽어오세요.
# 변수는 rgb_img

rgb_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_img, cmap="gray"); plt.axis("off"); plt.title("COLOR_BGR2RGB Buzz");
plt.show()

# imwite 함수를 이용하여 각 채널별 이미지를 저장하세요.
# red.png, green.png, blue.png
cv2.imwrite('red.png', red)
cv2.imwrite('green.png', green)
cv2.imwrite('blue.png', blue)

img = cv2.imread(r'C:/Users/15/Desktop/DataSet/example.png', cv2.IMREAD_GRAYSCALE)

# Show the type information for an array
print(type(img))
print(img.dtype)

# Example ndarray object properties
# img 의 ndim, shape, size 함수를 이용하여 정보를 확인해 보세요.
img_dims = img.ndim
img_shape = img.shape
img_size = img.size

print("\n{:^70}".format("NDARRAY ATTRIBUTES"))
print("{:^70}\n".format("===================="))
print("{:^30}{:^20}{:^20}".format("Description", "Example", "Value"))
print("{:^30}{:^20}{:^20}".format("-------------", "---------", "-------"))
print("{:^30}{:^20}{:^20}".format("Number of dimensions", "img.ndim", img_dims))
print("{:^30}{:^20}{:^20}".format("Image Shape", "img.shape", str(img_shape)))
print("{:^30}{:^20}{:^20}".format("Pixel count", "img.size", img_size))

print(img[37][73])
print(color_img[37][73][:])
print(img[:3, -4:],img[:3, -4:].dtype)

print("Original img.shape:", img.shape)
print("Shape of the first column of pixels (all rows, column 0): ", img[:, 0].shape)
print("Shape of the first row of pixels (row 0, all columns): ", img[0, :].shape)

# Index arrays
rows = [0, 2, 4, 6, 8]
cols = [1, 3, 5, 7, 9]
print("rows =", rows)
print("cols =", cols)
print("img[rows, cols]: ", img[rows, cols])

height, width, depth = rgb_img.shape

# Create an array of ones, the same shape as the image.
mask = np.ones((height,width), dtype=bool)

# Set the middle pixel to 0.
half_height = height//2
half_width = width//2
mask[half_height,half_width] = 0

# Calculate the distance to the middle pixel for every pixel.
distance_mask = nd.morphology.distance_transform_edt(mask)
max_distance = np.max(distance_mask)

# Normalize for display.
distance_mask_display = (distance_mask*255./max_distance).astype(dtype=np.uint8)

# Create a circle mask (in_range) based on distance from the center.
in_range = distance_mask < half_width

# For display purposes, to see the pixels in range.
circle_mask = np.zeros((rgb_img.shape[0], rgb_img.shape[1], rgb_img.shape[2]), np.uint8)
cv2.circle(
    circle_mask,
    (half_width,
     half_height),
     int(circle_mask.shape[1]/2),
     (255, 255, 255),
     -1
)#circle 인자가 열, 행 순인거 같음! [0]/2 [1]/2 순서가 아닌 이유

# Set all of the pixels where the mask is out of range to 0.
masked_buzz = np.zeros((rgb_img.shape[0], rgb_img.shape[1], rgb_img.shape[2]), np.uint8)
cv2.bitwise_and(rgb_img, circle_mask, masked_buzz)

plt.figure(figsize=(12, 6))
plt.subplot(141); plt.imshow(rgb_img, cmap="gray"); plt.axis("off"); plt.title("Buzz");
plt.subplot(142); plt.imshow(distance_mask_display, cmap="gray"); plt.axis("off"); plt.title("Distances");
plt.subplot(143); plt.imshow(circle_mask, cmap="gray"); plt.axis("off"); plt.title("Circle Mask");
plt.subplot(144); plt.imshow(masked_buzz, cmap="gray"); plt.axis("off"); plt.title("Buzz With Mask");
plt.show()

print(img[img > 230])

print("Minimum pixel intensity:", img.min())
print("Maximum pixel intensity:", img.max())
print("Mean pixel intensity:", img.mean())
print("Cumulative pixel intensity:", img.sum())

print("maximum column sum:", img.sum(axis=0, dtype=float).max())

# Convert from uint8 -> float64, use .astype()
img64 = img.astype(np.float64)
print("old dtype:", img.dtype)
print("new dtype:", img64.dtype)

img3d = np.atleast_3d(img)  # equivalent to img[:, :, np.newaxis]
print("old shape:", img.shape)
print("new shape:", img3d.shape)

color_img = cv2.imread(r'C:/Users/15/Desktop/DataSet/example.png', cv2.IMREAD_COLOR) #type: uint8

bluePlusGreen = color_img[:, :, 0] + color_img[:, :, 1]
redTimesBlue = color_img[:, :, 2] * color_img[:, :, 0]
greenMinusRed = color_img[:, :, 1] - color_img[:, :, 2]

plt.figure(figsize=(12, 6))
plt.subplot(131); plt.imshow(bluePlusGreen, cmap="gray"); plt.axis("off"); plt.title("Sum");
plt.subplot(132); plt.imshow(redTimesBlue, cmap="gray"); plt.axis("off"); plt.title("Product");
plt.subplot(133); plt.imshow(greenMinusRed, cmap="gray"); plt.axis("off"); plt.title("Difference");
plt.show()

_img = color_img.astype(float) # type: #float, height, width, channels
bluePlusGreen = _img[:, :, 0] + _img[:, :, 1]
redTimesBlue = _img[:, :, 2] * _img[:, :, 0]
greenMinusRed = _img[:, :, 1] - _img[:, :, 2]

plt.figure(figsize=(12, 6))
plt.subplot(131); plt.imshow(bluePlusGreen, cmap="gray"); plt.axis("off"); plt.title("Sum");
plt.subplot(132); plt.imshow(redTimesBlue, cmap="gray"); plt.axis("off"); plt.title("Product");
plt.subplot(133); plt.imshow(greenMinusRed, cmap="gray"); plt.axis("off"); plt.title("Difference");
plt.show()

img = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)

#  your code here 이미지 각도 돌리기
# img90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # 시계방향으로 90도 회전
img270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 반시계방향으로 90도 회전
imgFlip = cv2.flip(img, 1)
img_Flip_180 = cv2.rotate(imgFlip, cv2.ROTATE_180)  # Flip_180도 회전
# = 시계방향으로 270도 회전

plt.figure(figsize=(16, 6))
plt.subplot(141); plt.imshow(img, cmap="gray"); plt.axis("off"); plt.title("Original");
plt.subplot(142); plt.imshow(img270, cmap="gray"); plt.axis("off"); plt.title("Transpose");
plt.subplot(143); plt.imshow(imgFlip, cmap="gray"); plt.axis("off"); plt.title("Flip Horizontal");
plt.subplot(144); plt.imshow(img_Flip_180, cmap="gray"); plt.axis("off"); plt.title("Flip vertical");
plt.show()

sub_img = img[::10, ::10]
res_img = cv2.resize(img, sub_img.shape[::-1])
plt.figure(figsize=(9, 4));
plt.subplot(131); plt.imshow(img, cmap="gray"); plt.axis("off"); plt.title("Original");
plt.subplot(132); plt.imshow(sub_img, cmap="gray"); plt.axis("off"); plt.title("Subsampled");
plt.subplot(133); plt.imshow(res_img, cmap="gray"); plt.axis("off"); plt.title("cv2.resize - Linear");
plt.show()# 크기가 줄어드는게 아니고 해상도 줄어드는 것
print("Original size:", img.shape)
print("Reduced size:", sub_img.shape)

bs_img = cv2.GaussianBlur(img, ksize=(13, 13), sigmaX=2.5)[::10, ::10]
interp_img = cv2.resize(img, sub_img.shape[::-1], interpolation=cv2.INTER_AREA)
plt.figure(figsize=(12, 4));
plt.subplot(141); plt.imshow(img, cmap="gray"); plt.axis("off"); plt.title("Original");
plt.subplot(142); plt.imshow(sub_img, cmap="gray"); plt.axis("off"); plt.title("Subsampled");
plt.subplot(143); plt.imshow(bs_img, cmap="gray"); plt.axis("off"); plt.title("Blur + Subsampled");
plt.subplot(144); plt.imshow(interp_img, cmap="gray"); plt.axis("off"); plt.title("cv2.resize - Area Interpolation");
plt.show()

dy = np.diff(img.astype(float), axis=0)  # dy
dx = np.diff(img.astype(float), axis=1)  # dx
plt.figure(figsize=(12, 6));
plt.subplot(131); plt.imshow(img, cmap="gray"); plt.axis("off"); plt.title("Original");
plt.subplot(132); plt.imshow(dy, cmap="gray"); plt.axis("off"); plt.title("dy");
plt.subplot(133); plt.imshow(dx, cmap="gray"); plt.axis("off"); plt.title("dx");
plt.show()

dy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
dx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
plt.figure(figsize=(12, 6));
plt.subplot(131); plt.imshow(img, cmap="gray"); plt.axis("off"); plt.title("Original");
plt.subplot(132); plt.imshow(dy, cmap="gray"); plt.axis("off"); plt.title("dy");
plt.subplot(133); plt.imshow(dx, cmap="gray"); plt.axis("off"); plt.title("dx");
plt.show()

# generate a test image
gradient = np.arange(0, 9)

# Adding 200 will cause values to "wrap around" above 255.  This is intended here.
# It might be worthwhile to note the use of np.newaxis.  There are other ways to accomplish this.
# (See np.atleast_2d)
vals = (gradient * gradient[:, np.newaxis]) * 255. / 64. + 200
img = vals.astype(dtype=np.uint8)

# make a copy of the original that shares the same scale as the outputs -- the white border
# will not appear in the jupyter notebook
_img = cv2.copyMakeBorder(img, 4, 4, 4, 4, borderType=cv2.BORDER_CONSTANT, value=255)

#
zPadded = cv2.copyMakeBorder(img, 4, 4, 4, 4, borderType=cv2.BORDER_CONSTANT, value=127)
rPadded = cv2.copyMakeBorder(img, 4, 4, 4, 4, borderType=cv2.BORDER_REFLECT)
r101Padded = cv2.copyMakeBorder(img, 4, 4, 4, 4, borderType=cv2.BORDER_REFLECT_101)

plt.figure(figsize=(12, 6));
plt.subplot(141);
plt.imshow(_img, cmap="gray");
plt.axis("off");
plt.title("Original");
plt.subplot(142);
plt.imshow(zPadded, cmap="gray");
plt.axis("off");
plt.title("Constant Fill (127)");
plt.subplot(143);
plt.imshow(rPadded, cmap="gray");
plt.axis("off");
plt.title("Border Reflect");
plt.subplot(144);
plt.imshow(r101Padded, cmap="gray");
plt.axis("off");
plt.title("Border Reflect 101");
plt.show()
