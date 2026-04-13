import cv2
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
IMAGE_PATH = r"C:\Users\LENOVO\Downloads\image_analysis_codes\demo.jpg"
# ─────────────────────────────────────────────

img_bgr = cv2.imread(IMAGE_PATH)
img_bgr = cv2.resize(img_bgr, (800, 500))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — Gaussian Blur
#  Real images always contain noise — tiny random pixel variations.
#  Canny is sensitive enough to detect noise as false edges.
#  Gaussian Blur smooths these out BEFORE edge detection runs.
#
#  The (5,5) is the kernel size — how many neighbouring pixels to consider.
#  Must be odd. Larger kernel = stronger blur.
#  The last 0 tells OpenCV to auto-calculate the spread (sigma).
# ─────────────────────────────────────────────────────────────────────────────
blur_3x3 = cv2.GaussianBlur(gray, (3, 3), 0)   # mild blur
blur_5x5 = cv2.GaussianBlur(gray, (5, 5), 0)   # standard — use this most of the time
blur_9x9 = cv2.GaussianBlur(gray, (9, 9), 0)   # heavy blur

# Show the effect of blur live
cv2.imshow("Original Greyscale", gray)
cv2.imshow("After Gaussian Blur (5x5)", blur_5x5)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — Canny Edge Detection
#  Takes greyscale, returns a binary image — white = edge, black = not edge.
#
#  threshold1 (50)  : lower bound — gradients weaker than this are rejected (noise)
#  threshold2 (150) : upper bound — gradients stronger than this are definite edges
#  Gradients in between are kept only if connected to a strong edge.
#
#  Lower both values → detects more edges (but also more noise)
#  Raise both values → fewer, cleaner edges
# ─────────────────────────────────────────────────────────────────────────────

# Without blur — notice the extra noise/false edges
edges_no_blur    = cv2.Canny(gray,     50, 150)

# With blur — much cleaner, only real structural edges
edges_with_blur  = cv2.Canny(blur_5x5, 50, 150)

# Effect of different threshold values
edges_sensitive  = cv2.Canny(blur_5x5, 20, 60)   # low thresholds → more edges
edges_strict     = cv2.Canny(blur_5x5, 100, 200)  # high thresholds → fewer edges

# Show clean edge result live
cv2.imshow("Canny Edges (blur + thresholds 50-150)", edges_with_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
#  END-OF-SECTION COMPARISON GRID
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("Gaussian Blur + Canny Edge Detection", fontsize=15, fontweight="bold")

# Row 1 — Blur effect
axes[0, 0].imshow(gray,     cmap="gray"); axes[0, 0].set_title("Original Greyscale")
axes[0, 1].imshow(blur_3x3, cmap="gray"); axes[0, 1].set_title("Gaussian Blur 3×3\n(mild)")
axes[0, 2].imshow(blur_5x5, cmap="gray"); axes[0, 2].set_title("Gaussian Blur 5×5\n(standard)")
axes[0, 3].imshow(blur_9x9, cmap="gray"); axes[0, 3].set_title("Gaussian Blur 9×9\n(heavy)")

# Row 2 — Canny comparisons
axes[1, 0].imshow(img_rgb,          );   axes[1, 0].set_title("Original Colour")
axes[1, 1].imshow(edges_no_blur,  cmap="gray"); axes[1, 1].set_title("Canny WITHOUT blur\n(noisy false edges)")
axes[1, 2].imshow(edges_with_blur,cmap="gray"); axes[1, 2].set_title("Canny WITH blur 5×5\n(clean edges ✓)")
axes[1, 3].imshow(edges_sensitive,cmap="gray"); axes[1, 3].set_title("Low thresholds (20/60)\nMore edges detected")

for row in axes:
    for ax in row:
        ax.axis("off")

plt.tight_layout()
plt.show()
