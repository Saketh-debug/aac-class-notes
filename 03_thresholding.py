import cv2
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
IMAGE_PATH = "demo.jpg"
# ─────────────────────────────────────────────

img_bgr = cv2.imread(IMAGE_PATH)
img_bgr = cv2.resize(img_bgr, (800, 500))
# Thresholding always works on greyscale — convert first
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — Simple (Global) Thresholding — all 5 types
#  One value T applied to every pixel in the entire image.
#  Works well with even, consistent lighting.
# ─────────────────────────────────────────────────────────────────────────────
T = 127   # threshold value — pixels above this → white, below → black

_, thresh_binary     = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)
_, thresh_binary_inv = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY_INV)
_, thresh_trunc      = cv2.threshold(gray, T, 255, cv2.THRESH_TRUNC)
_, thresh_tozero     = cv2.threshold(gray, T, 255, cv2.THRESH_TOZERO)
_, thresh_tozero_inv = cv2.threshold(gray, T, 255, cv2.THRESH_TOZERO_INV)

# Show one result live to let students observe it closely
cv2.imshow("THRESH_BINARY  (T=127)", thresh_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — Adaptive Thresholding
#  Computes a different T for every small local region of the image.
#  Works well with uneven lighting, shadows, real-world photos.
#
#  blockSize : size of the local neighbourhood (must be odd)
#  C         : constant subtracted from the local average
# ─────────────────────────────────────────────────────────────────────────────
adaptive_mean     = cv2.adaptiveThreshold(gray, 255,
                        cv2.ADAPTIVE_THRESH_MEAN_C,
                        cv2.THRESH_BINARY, blockSize=11, C=2)

adaptive_gaussian = cv2.adaptiveThreshold(gray, 255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, blockSize=11, C=2)

# Show adaptive result live
cv2.imshow("Adaptive (Gaussian) — handles uneven lighting", adaptive_gaussian)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
#  END-OF-SECTION COMPARISON GRID
#  All threshold types + adaptive — side by side
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("Thresholding — All Types Compared", fontsize=15, fontweight="bold")

# Row 1 — Global thresholding types
axes[0, 0].imshow(gray,              cmap="gray"); axes[0, 0].set_title("Greyscale input")
axes[0, 1].imshow(thresh_binary,     cmap="gray"); axes[0, 1].set_title("THRESH_BINARY\nPixel>T → 255, else 0")
axes[0, 2].imshow(thresh_binary_inv, cmap="gray"); axes[0, 2].set_title("THRESH_BINARY_INV\nPixel>T → 0, else 255")
axes[0, 3].imshow(thresh_trunc,      cmap="gray"); axes[0, 3].set_title("THRESH_TRUNC\nPixel>T → set to T")

# Row 2 — Remaining types + adaptive
axes[1, 0].imshow(thresh_tozero,     cmap="gray"); axes[1, 0].set_title("THRESH_TOZERO\nPixel>T → unchanged, else 0")
axes[1, 1].imshow(thresh_tozero_inv, cmap="gray"); axes[1, 1].set_title("THRESH_TOZERO_INV\nPixel>T → 0, else unchanged")
axes[1, 2].imshow(adaptive_mean,     cmap="gray"); axes[1, 2].set_title("Adaptive — MEAN\n(local T per region)")
axes[1, 3].imshow(adaptive_gaussian, cmap="gray"); axes[1, 3].set_title("Adaptive — GAUSSIAN\n(best for real-world photos)")

for row in axes:
    for ax in row:
        ax.axis("off")

plt.tight_layout()
plt.show()
