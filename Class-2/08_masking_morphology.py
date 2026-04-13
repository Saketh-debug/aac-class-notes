import cv2
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
IMAGE_PATH = r"C:\Users\LENOVO\Downloads\image_analysis_codes\demo.jpg"
# ─────────────────────────────────────────────

img_bgr = cv2.imread(IMAGE_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — Create a binary mask via thresholding
#  A mask is a binary image — same size as original.
#  White pixels (255) = "keep this"
#  Black pixels (0)   = "discard this"
# ─────────────────────────────────────────────────────────────────────────────
_, mask_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

cv2.imshow("Binary mask from thresholding", mask_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — Apply the mask to the original image
#  bitwise_and keeps original pixel values where mask = 255
#  and sets pixels to 0 (black) where mask = 0
# ─────────────────────────────────────────────────────────────────────────────
masked_result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_thresh)

cv2.imshow("Masked result — only bright regions kept", masked_result)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — Colour-based mask using HSV
#  Use HSV colour space to isolate pixels by colour
#  Here we isolate warm/orange tones (road/skin/desert tones in the demo image)
# ─────────────────────────────────────────────────────────────────────────────
hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# Adjust these values to target a specific colour in your own image
lower_warm = np.array([5,  40, 60])
upper_warm = np.array([30, 255, 255])

mask_colour = cv2.inRange(hsv, lower_warm, upper_warm)
masked_colour = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_colour)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — Morphological Operations
#  Masks from thresholding are often imperfect:
#    - small noise specks outside objects
#    - small holes inside objects
#    - ragged boundaries
#  Morphological ops clean these up.
#
#  The kernel defines the neighbourhood size for each operation.
#  Larger kernel = stronger effect.
# ─────────────────────────────────────────────────────────────────────────────
kernel = np.ones((7, 7), np.uint8)

# Dilation — expands white regions
# Fills small holes inside objects, connects broken edges
dilation = cv2.dilate(mask_thresh, kernel, iterations=1)

# Erosion — shrinks white regions
# Removes small noise specks, separates lightly touching objects
erosion = cv2.erode(mask_thresh, kernel, iterations=1)

# Opening = erosion then dilation
# Removes small noise specks WITHOUT significantly changing large object size
opening = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

# Closing = dilation then erosion
# Fills small holes inside large objects WITHOUT significantly changing their size
closing = cv2.morphologyEx(mask_thresh, cv2.MORPH_CLOSE, kernel)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 — Apply cleaned mask back to original
#  Use closing to clean the mask (fill holes), then apply it
# ─────────────────────────────────────────────────────────────────────────────
cleaned_mask  = closing
final_result  = cv2.bitwise_and(img_bgr, img_bgr, mask=cleaned_mask)


# ─────────────────────────────────────────────────────────────────────────────
#  END-OF-SECTION COMPARISON GRID  (2 rows)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("Masking + Morphological Operations", fontsize=15, fontweight="bold")

# Row 1 — Masking
axes[0, 0].imshow(img_rgb);                                               axes[0, 0].set_title("Original image")
axes[0, 1].imshow(mask_thresh, cmap="gray");                              axes[0, 1].set_title("Binary mask\n(from thresholding)")
axes[0, 2].imshow(cv2.cvtColor(masked_result, cv2.COLOR_BGR2RGB));        axes[0, 2].set_title("Mask applied\n(bitwise_and)")
axes[0, 3].imshow(cv2.cvtColor(masked_colour, cv2.COLOR_BGR2RGB));        axes[0, 3].set_title("Colour-based mask\n(HSV inRange)")

# Row 2 — Morphological operations on the mask
axes[1, 0].imshow(mask_thresh, cmap="gray");  axes[1, 0].set_title("Original mask\n(before morphology)")
axes[1, 1].imshow(dilation,    cmap="gray");  axes[1, 1].set_title("Dilation\n(expands white — fills holes)")
axes[1, 2].imshow(erosion,     cmap="gray");  axes[1, 2].set_title("Erosion\n(shrinks white — removes specks)")
axes[1, 3].imshow(opening,     cmap="gray");  axes[1, 3].set_title("Opening\n(erosion → dilation, removes noise)")

for row in axes:
    for ax in row:
        ax.axis("off")

plt.tight_layout()
plt.show()

# ─── Separate figure for closing + final result ───────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(14, 5))
fig2.suptitle("Closing + Final Masked Output", fontsize=14, fontweight="bold")

axes2[0].imshow(mask_thresh, cmap="gray");                              axes2[0].set_title("Original mask")
axes2[1].imshow(closing,     cmap="gray");                              axes2[1].set_title("Closing\n(dilation → erosion, fills holes)")
axes2[2].imshow(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB));         axes2[2].set_title("Cleaned mask applied\nto original image")

for ax in axes2:
    ax.axis("off")

plt.tight_layout()
plt.show()
