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
#  Harris Corner Detection
#  A corner is a point where brightness changes sharply in multiple directions.
#  More distinctive and stable than a plain edge (which changes in one direction).
#  Used for: feature matching, image stitching, motion tracking.
#
#  blockSize : size of neighbourhood considered around each pixel
#  ksize     : kernel size for the Sobel gradient computation
#  k         : Harris sensitivity parameter (typical range 0.04 – 0.06)
#              higher k = fewer corners detected
# ─────────────────────────────────────────────────────────────────────────────

# Harris requires float32
gray_float = np.float32(gray)

harris = cv2.cornerHarris(gray_float, blockSize=5, ksize=3, k=0.04)

# Dilate the result so corner markers are more visible when we draw them
harris_dilated = cv2.dilate(harris, None)

# Mark corners on a copy — any pixel where the Harris response is strong enough
# gets coloured red (0, 0, 255) in BGR
harris_result = img_bgr.copy()
harris_result[harris_dilated > 0.01 * harris_dilated.max()] = [255, 0, 0] # blue dots = corners (in BGR)

# Show result live
cv2.imshow("Harris Corner Detection  (blue dots = corners)", harris_result)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
#  Shi-Tomasi (goodFeaturesToTrack) — often more practical
#  Returns exact (x, y) coordinates of corners you can use in code.
#  maxCorners    : maximum number of corners to return
#  qualityLevel  : minimum strength relative to best corner (0.01 = 1%)
#  minDistance   : corners must be this many pixels apart
# ─────────────────────────────────────────────────────────────────────────────
corners = cv2.goodFeaturesToTrack(gray, maxCorners=50, qualityLevel=0.01, minDistance=20)

shi_result = img_bgr.copy()
if corners is not None:
    corners = np.int32(corners)
    for corner in corners:
        x, y = corner.ravel()   # ravel() extracts the x,y from the nested array
        cv2.circle(shi_result, (x, y), 6, (0, 255, 0), -1)   # green filled dot


# ─────────────────────────────────────────────────────────────────────────────
#  END-OF-SECTION COMPARISON GRID
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle("Corner Detection — Harris vs Shi-Tomasi", fontsize=15, fontweight="bold")

axes[0].imshow(img_rgb);                                       axes[0].set_title("Original")
axes[1].imshow(cv2.cvtColor(harris_result, cv2.COLOR_BGR2RGB)); axes[1].set_title("Harris Corner Detection\n(blue = corners)")
axes[2].imshow(cv2.cvtColor(shi_result,   cv2.COLOR_BGR2RGB)); axes[2].set_title("Shi-Tomasi (goodFeaturesToTrack)\n(green dots = corners)")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
