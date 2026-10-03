import cv2

# Read the image
image = cv2.imread("images/sample.jpg")

# Check whether the image loaded successfully
if image is None:
    print("Error: Could not load the image.")
else:
    print("Image loaded successfully.")

    # Display the image
    cv2.imshow("Q1 - Original Image", image)

    # Wait until a key is pressed
    cv2.waitKey(0)

    # Close the image window
    cv2.destroyAllWindows()