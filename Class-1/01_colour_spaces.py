import cv2
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
#  Change this path to your own image
import os

IMAGE_PATH = r"C:\Users\LENOVO\Downloads\image_analysis_codes\demo.jpg"
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"File missing, create or update `IMAGE_PATH`: {IMAGE_PATH}")
# ─────────────────────────────────────────────

# ── Load the image ────────────────────────────────────────────────────────────
# OpenCV always loads images in BGR order (Blue, Green, Red)
img_bgr = cv2.imread(IMAGE_PATH)

if img_bgr is None:
    raise FileNotFoundError(f"Could not load image at: {IMAGE_PATH}")

print(f"Image shape : {img_bgr.shape}")   # (Height, Width, 3)
print(f"Image dtype : {img_bgr.dtype}")   # uint8  →  values 0-255


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — The BGR vs RGB bug
#  OpenCV loads BGR.  Matplotlib expects RGB.
#  Without conversion, red and blue channels are swapped in the display.
# ─────────────────────────────────────────────────────────────────────────────
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_bgr = cv2.resize(img_bgr, (800, 500))  # Resize for display purposes
# Show the problem live — one window correct, one wrong
cv2.imshow("original image", img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — Split individual channels
#  Each channel viewed alone looks like a greyscale image.
#  Bright = a lot of that colour present.  Dark = very little.
# ─────────────────────────────────────────────────────────────────────────────
b_channel, g_channel, r_channel = cv2.split(img_bgr)

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — Convert to HSV
#  H = Hue (actual colour, 0-179)
#  S = Saturation (how vivid, 0-255)
#  V = Value / brightness (0-255)
#  Useful when you need to detect objects by colour under varying lighting.
# ─────────────────────────────────────────────────────────────────────────────
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
h, s, v  = cv2.split(img_hsv)

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — LAB and YCrCb (for reference)
# ─────────────────────────────────────────────────────────────────────────────
img_lab   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)


# ─────────────────────────────────────────────────────────────────────────────
#  END-OF-SECTION COMPARISON GRID
#  All colour space outputs side by side
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
fig.suptitle("Colour Spaces — Overview", fontsize=16, fontweight="bold")

# Row 1 — Original and colour spaces
axes[0, 0].imshow(img_rgb);             axes[0, 0].set_title("Original (RGB correct)")
axes[0, 1].imshow(img_bgr);             axes[0, 1].set_title("BGR shown in Matplotlib\n(red-blue swapped!)")
axes[0, 2].imshow(img_hsv);             axes[0, 2].set_title("HSV colour space")
axes[0, 3].imshow(img_lab);             axes[0, 3].set_title("LAB colour space")

# Row 2 — Individual BGR channels
axes[1, 0].imshow(b_channel, cmap="Blues");  axes[1, 0].set_title("Blue channel")
axes[1, 1].imshow(g_channel, cmap="Greens"); axes[1, 1].set_title("Green channel")
axes[1, 2].imshow(r_channel, cmap="Reds");   axes[1, 2].set_title("Red channel")
axes[1, 3].imshow(img_ycrcb);               axes[1, 3].set_title("YCrCb colour space")

# Row 3 — HSV individual channels
axes[2, 0].imshow(h, cmap="hsv");   axes[2, 0].set_title("H — Hue (what colour)")
axes[2, 1].imshow(s, cmap="gray");  axes[2, 1].set_title("S — Saturation (how vivid)")
axes[2, 2].imshow(v, cmap="gray");  axes[2, 2].set_title("V — Value (brightness)")
axes[2, 3].axis("off")

for row in axes:
    for ax in row:
        ax.axis("off")

plt.tight_layout()
plt.show()
