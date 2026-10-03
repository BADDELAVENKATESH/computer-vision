import streamlit as st  # pyright: ignore[reportMissingImports]
import cv2
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Computer Vision Day 1",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ Computer Vision – Day 1")
st.write("Interactive demonstration of all 25 OpenCV and NumPy questions.")

IMAGE_PATH = Path("images/sample.jpg")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# Load image
# ------------------------------------------------------------

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    st.error(f"Could not load image: {IMAGE_PATH}")
    st.stop()

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

height, width, channels = image.shape

# ------------------------------------------------------------
# Original image
# ------------------------------------------------------------

st.header("Q1–Q10: Image Basics")

st.image(rgb, caption="Q1 – Original Image", use_container_width=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Height", height)

with col2:
    st.metric("Width", width)

with col3:
    st.metric("Channels", channels)

st.write("**Q2:** Image loaded successfully.")
st.write(f"**Q3:** Resolution = {width} × {height}, Channels = {channels}")
st.write(f"**Q4:** Total pixels = {height * width}")
st.write(f"**Q5:** Data type = `{image.dtype}`")

st.image(
    rgb,
    caption="Q6 – Saved Image",
    use_container_width=True
)

st.image(
    gray,
    caption="Q7/Q8 – Grayscale Image",
    use_container_width=True
)

resized = cv2.resize(image, (width // 2, height // 2))
resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

st.image(
    resized_rgb,
    caption="Q10 – Resized to 50%",
    use_container_width=True
)

# ------------------------------------------------------------
# Pixel operations
# ------------------------------------------------------------

st.header("Q11–Q18: Pixel and Intensity Operations")

x = st.number_input(
    "X coordinate",
    min_value=0,
    max_value=width - 1,
    value=min(100, width - 1)
)

y = st.number_input(
    "Y coordinate",
    min_value=0,
    max_value=height - 1,
    value=min(100, height - 1)
)

x = int(x)
y = int(y)

B, G, R = image[y, x]

st.write(f"**Q11:** Pixel at ({x}, {y}) = `{image[y, x].tolist()}`")
st.write(f"**Q13:** B = `{B}`, G = `{G}`, R = `{R}`")

modified = image.copy()
modified[y, x] = [0, 0, 255]

modified_rgb = cv2.cvtColor(modified, cv2.COLOR_BGR2RGB)

st.image(
    modified_rgb,
    caption="Q12 – Selected Pixel Changed to Red",
    use_container_width=True
)

blue, green, red = cv2.split(image)

col1, col2, col3 = st.columns(3)

with col1:
    st.image(blue, caption="Q14 – Blue Channel")

with col2:
    st.image(green, caption="Q14 – Green Channel")

with col3:
    st.image(red, caption="Q14 – Red Channel")

merged = cv2.merge([blue, green, red])
merged_rgb = cv2.cvtColor(merged, cv2.COLOR_BGR2RGB)

st.image(
    merged_rgb,
    caption="Q15 – Merged Channels",
    use_container_width=True
)

st.write(f"**Q16:** Minimum intensity = `{np.min(gray)}`")
st.write(f"**Q16:** Maximum intensity = `{np.max(gray)}`")
st.write(f"**Q17:** Mean intensity = `{np.mean(gray):.2f}`")
st.write(f"**Q18:** Standard deviation = `{np.std(gray):.2f}`")

# ------------------------------------------------------------
# Quantization and image creation
# ------------------------------------------------------------

st.header("Q19–Q25: Image Processing")

constant_128 = np.full((256, 256), 128, dtype=np.uint8)

st.image(
    constant_128,
    caption="Q19 – 256×256 Image with Intensity 128"
)

ramp = np.tile(
    np.arange(256, dtype=np.uint8),
    (256, 1)
)

st.image(
    ramp,
    caption="Q20 – Grayscale Ramp 0–255"
)

quantized_4bit = ((gray // 16) * 16).astype(np.uint8)

st.image(
    quantized_4bit,
    caption="Q21 – 4-bit Quantized Image",
    use_container_width=True
)

quantized_2bit = ((gray // 64) * 64).astype(np.uint8)

st.image(
    quantized_2bit,
    caption="Q22 – 2-bit Quantized Image",
    use_container_width=True
)

downsampled = cv2.resize(
    image,
    (width // 2, height // 2)
)

downsampled_rgb = cv2.cvtColor(
    downsampled,
    cv2.COLOR_BGR2RGB
)

st.image(
    downsampled_rgb,
    caption=f"Q23 – Downsampled: {width // 2} × {height // 2}",
    use_container_width=True
)

x1 = st.number_input(
    "ROI X1",
    min_value=0,
    max_value=width - 2,
    value=min(100, width - 2)
)

y1 = st.number_input(
    "ROI Y1",
    min_value=0,
    max_value=height - 2,
    value=min(100, height - 2)
)

x2 = st.number_input(
    "ROI X2",
    min_value=1,
    max_value=width,
    value=min(500, width)
)

y2 = st.number_input(
    "ROI Y2",
    min_value=1,
    max_value=height,
    value=min(400, height)
)

x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

if x2 > x1 and y2 > y1:
    roi = image[y1:y2, x1:x2]
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    st.image(
        roi_rgb,
        caption="Q24 – Cropped Region of Interest",
        use_container_width=True
    )

rotated = cv2.rotate(
    image,
    cv2.ROTATE_90_CLOCKWISE
)

rotated_rgb = cv2.cvtColor(
    rotated,
    cv2.COLOR_BGR2RGB
)

st.image(
    rotated_rgb,
    caption="Q25 – Rotated 90° Clockwise",
    use_container_width=True
)

# ------------------------------------------------------------
# Download results
# ------------------------------------------------------------

st.header("📁 Generated Results")

st.success("All 25 Computer Vision Day 1 questions are implemented.")

st.write(
    "The original Python program generates the files in the "
    "`outputs/` directory."
)