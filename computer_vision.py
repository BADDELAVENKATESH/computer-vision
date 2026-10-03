import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# COMPUTER VISION - DAY 1
# All 25 Questions in One Python Program
# ============================================================

IMAGE_PATH = "images/sample.jpg"
OUTPUT_DIR = "outputs"

# User-selected coordinates
X = 100
Y = 100

# ROI coordinates
X1 = 100
Y1 = 100
X2 = 500
Y2 = 400


# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Q1. Read image using OpenCV and display it
# ============================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}\n"
        "Make sure images/sample.jpg exists."
    )

cv2.imwrite(f"{OUTPUT_DIR}/q01_original.jpg", image)

print("\nQ1: Image loaded and saved as q01_original.jpg")

# Display image
cv2.imshow("Q1 - Original Image", image)
cv2.waitKey(1000)
cv2.destroyAllWindows()


# ============================================================
# Q2. Check if image loaded successfully
# ============================================================

if image is not None:
    print("Q2: Image loaded successfully.")
else:
    print("Q2: Error - image could not be loaded.")


# ============================================================
# Q3. Print image height, width and number of channels
# ============================================================

height, width, channels = image.shape

print("\nQ3:")
print("Height:", height)
print("Width:", width)
print("Channels:", channels)


# ============================================================
# Q4. Calculate total number of pixels
# ============================================================

total_pixels = height * width

print("\nQ4:")
print("Total pixels:", total_pixels)


# ============================================================
# Q5. Print image data type
# ============================================================

print("\nQ5:")
print("Image data type:", image.dtype)


# ============================================================
# Q6. Save image using a different filename
# ============================================================

q6_path = f"{OUTPUT_DIR}/q06_saved_image.jpg"
cv2.imwrite(q6_path, image)

print("\nQ6:")
print("Image saved as:", q6_path)


# ============================================================
# Q7. Read image in grayscale and display
# ============================================================

gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if gray is None:
    raise FileNotFoundError("Could not load grayscale image.")

cv2.imwrite(f"{OUTPUT_DIR}/q07_grayscale.jpg", gray)

print("\nQ7: Grayscale image saved.")

cv2.imshow("Q7 - Grayscale Image", gray)
cv2.waitKey(1000)
cv2.destroyAllWindows()


# ============================================================
# Q8. Convert color image to grayscale using cv2.cvtColor
# ============================================================

gray_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite(
    f"{OUTPUT_DIR}/q08_converted_grayscale.jpg",
    gray_cvt
)

print("\nQ8: Color image converted to grayscale using cv2.cvtColor.")


# ============================================================
# Q9. Display image using Matplotlib and hide axis
# ============================================================

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 6))
plt.imshow(rgb_image)
plt.axis("off")
plt.title("Q9 - Image using Matplotlib")
plt.savefig(
    f"{OUTPUT_DIR}/q09_matplotlib.png",
    bbox_inches="tight"
)
plt.close()

print("\nQ9: Matplotlib image saved.")


# ============================================================
# Q10. Resize image to 50%
# ============================================================

new_width = int(width * 0.5)
new_height = int(height * 0.5)

resized = cv2.resize(
    image,
    (new_width, new_height)
)

cv2.imwrite(
    f"{OUTPUT_DIR}/q10_resized_50_percent.jpg",
    resized
)

print("\nQ10:")
print("Original resolution:", width, "x", height)
print("50% resolution:", new_width, "x", new_height)


# ============================================================
# Q11. Access and print pixel value at user-provided (x,y)
# ============================================================

if 0 <= X < width and 0 <= Y < height:

    pixel = image[Y, X]

    print("\nQ11:")
    print(f"Pixel at (x={X}, y={Y}):", pixel)

else:
    print("\nQ11: Coordinates are outside image boundaries.")


# ============================================================
# Q12. Modify selected pixel and save image
# ============================================================

modified_image = image.copy()

if 0 <= X < width and 0 <= Y < height:

    # OpenCV uses BGR.
    # [0, 0, 255] = Red
    modified_image[Y, X] = [0, 0, 255]

    cv2.imwrite(
        f"{OUTPUT_DIR}/q12_modified_pixel.jpg",
        modified_image
    )

    print("\nQ12:")
    print(f"Pixel at ({X}, {Y}) changed to red.")
    print("Modified image saved.")


# ============================================================
# Q13. Print B, G and R values separately
# ============================================================

if 0 <= X < width and 0 <= Y < height:

    B, G, R = image[Y, X]

    print("\nQ13:")
    print("Blue (B):", B)
    print("Green (G):", G)
    print("Red (R):", R)


# ============================================================
# Q14. Split image into B, G and R channels and display
# ============================================================

B, G, R = cv2.split(image)

cv2.imwrite(f"{OUTPUT_DIR}/q14_blue_channel.jpg", B)
cv2.imwrite(f"{OUTPUT_DIR}/q14_green_channel.jpg", G)
cv2.imwrite(f"{OUTPUT_DIR}/q14_red_channel.jpg", R)

print("\nQ14:")
print("Blue, Green and Red channel images saved.")


# ============================================================
# Q15. Merge the three channels
# ============================================================

merged_image = cv2.merge([B, G, R])

cv2.imwrite(
    f"{OUTPUT_DIR}/q15_merged_channels.jpg",
    merged_image
)

print("\nQ15: Three channels merged successfully.")


# ============================================================
# Q16. Find minimum and maximum intensity in grayscale image
# ============================================================

min_intensity = np.min(gray_cvt)
max_intensity = np.max(gray_cvt)

print("\nQ16:")
print("Minimum intensity:", min_intensity)
print("Maximum intensity:", max_intensity)


# ============================================================
# Q17. Calculate mean intensity of grayscale image
# ============================================================

mean_intensity = np.mean(gray_cvt)

print("\nQ17:")
print("Mean intensity:", mean_intensity)


# ============================================================
# Q18. Calculate mean and standard deviation
# ============================================================

mean_value = np.mean(gray_cvt)
std_value = np.std(gray_cvt)

print("\nQ18:")
print("Mean:", mean_value)
print("Standard deviation:", std_value)


# ============================================================
# Q19. Create 256x256 grayscale image with intensity 128
# ============================================================

image_128 = np.full(
    (256, 256),
    128,
    dtype=np.uint8
)

cv2.imwrite(
    f"{OUTPUT_DIR}/q19_constant_128.jpg",
    image_128
)

print("\nQ19:")
print("Created 256x256 grayscale image with intensity 128.")


# ============================================================
# Q20. Create grayscale ramp from 0 to 255
# ============================================================

ramp = np.tile(
    np.arange(256, dtype=np.uint8),
    (256, 1)
)

cv2.imwrite(
    f"{OUTPUT_DIR}/q20_grayscale_ramp.jpg",
    ramp
)

print("\nQ20:")
print("Created grayscale ramp from 0 to 255.")


# ============================================================
# Q21. Convert 8-bit grayscale image to 4-bit quantized image
# ============================================================

quantized_4bit = (gray_cvt // 16) * 16

quantized_4bit = quantized_4bit.astype(np.uint8)

cv2.imwrite(
    f"{OUTPUT_DIR}/q21_4bit_quantized.jpg",
    quantized_4bit
)

print("\nQ21:")
print("8-bit grayscale converted to 4-bit quantized image.")


# ============================================================
# Q22. Convert 8-bit grayscale image to 2-bit quantized image
# ============================================================

quantized_2bit = (gray_cvt // 64) * 64

quantized_2bit = quantized_2bit.astype(np.uint8)

cv2.imwrite(
    f"{OUTPUT_DIR}/q22_2bit_quantized.jpg",
    quantized_2bit
)

print("\nQ22:")
print("8-bit grayscale converted to 2-bit quantized image.")


# ============================================================
# Q23. Downsample image by factor of 2
# ============================================================

downsampled = cv2.resize(
    image,
    (width // 2, height // 2)
)

cv2.imwrite(
    f"{OUTPUT_DIR}/q23_downsampled.jpg",
    downsampled
)

down_height, down_width = downsampled.shape[:2]

print("\nQ23:")
print("Original resolution:", width, "x", height)
print("Downsampled resolution:", down_width, "x", down_height)


# ============================================================
# Q24. Crop rectangular Region of Interest (ROI)
# ============================================================

if (
    0 <= X1 < X2 <= width
    and
    0 <= Y1 < Y2 <= height
):

    roi = image[Y1:Y2, X1:X2]

    cv2.imwrite(
        f"{OUTPUT_DIR}/q24_cropped_roi.jpg",
        roi
    )

    print("\nQ24:")
    print(
        f"ROI cropped from ({X1},{Y1}) "
        f"to ({X2},{Y2})."
    )

else:
    print("\nQ24: ROI coordinates are outside image boundaries.")


# ============================================================
# Q25. Rotate image by 90 degrees
# ============================================================

rotated = cv2.rotate(
    image,
    cv2.ROTATE_90_CLOCKWISE
)

cv2.imwrite(
    f"{OUTPUT_DIR}/q25_rotated_90.jpg",
    rotated
)

print("\nQ25:")
print("Image rotated 90 degrees clockwise.")
print("Rotated image saved.")


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("ALL 25 QUESTIONS COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nAll generated files are available in:")
print(os.path.abspath(OUTPUT_DIR))