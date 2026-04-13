import cv2
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
IMAGE_PATH = r"C:\Users\LENOVO\Downloads\image_analysis_codes\contours.png"
# ─────────────────────────────────────────────

img_bgr = cv2.imread(IMAGE_PATH)
img_bgr = cv2.resize(img_bgr, (800, 500))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — Produce a binary image first (required by findContours)
#  Contour detection traces the boundaries of WHITE regions in a binary image.
#  The white region = the object.  Black region = background.
# ─────────────────────────────────────────────────────────────────────────────
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

cv2.imshow("Binary image (input to contour detection)", binary)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — Find Contours
#  RETR_EXTERNAL : only outermost boundaries (ignores shapes inside shapes)
#  CHAIN_APPROX_SIMPLE : stores only endpoints of straight lines — efficient
#
#  Result: a Python list where each element is one contour
#          each contour = NumPy array of (x,y) boundary coordinates
# ─────────────────────────────────────────────────────────────────────────────
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Total contours found: {len(contours)}")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — Draw ALL contours on a copy of the original
# ─────────────────────────────────────────────────────────────────────────────
all_contours_img = img_bgr.copy()
cv2.drawContours(all_contours_img, contours, -1, (255, 0, 0), 3)
# -1 = draw all contours.  Replace with index (0, 1, 2...) to draw just one.


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — Filter by area and draw bounding boxes
#  Small contours are almost always noise. Filter them out.
#  cv2.contourArea() returns the area of a contour in pixels.
# ─────────────────────────────────────────────────────────────────────────────
MIN_AREA = 3200   # ignore contours smaller than this

filtered_img = img_bgr.copy()
bounding_img = img_bgr.copy()

large_contours = []

for contour in contours:
    area = cv2.contourArea(contour)

    if area > MIN_AREA:
        large_contours.append(contour)

        # Get the bounding rectangle that fully encloses this contour
        x, y, w, h = cv2.boundingRect(contour)

        # Draw the bounding rectangle on the image
        cv2.rectangle(bounding_img, (x, y), (x + w, y + h), (0, 0, 255), 3)

        # Label with the area
        cv2.putText(bounding_img, f"{int(area)}px", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

print(f"Contours after filtering (area > {MIN_AREA}): {len(large_contours)}")

# Draw only the filtered contours in green
cv2.drawContours(filtered_img, large_contours, -1, (255, 0, 0), 3)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 — Contour from Canny edges (alternative to thresholding)
#  Canny edges can also be used as input to findContours
# ─────────────────────────────────────────────────────────────────────────────
blurred      = cv2.GaussianBlur(gray, (5, 5), 0)
edges        = cv2.Canny(blurred, 50, 150)
contours_edge, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

canny_contour_img = img_bgr.copy()
cv2.drawContours(canny_contour_img, contours_edge, -1, (255, 100, 0), 3)


# ─────────────────────────────────────────────────────────────────────────────
#  END-OF-SECTION COMPARISON GRID
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Contour Detection — Full Pipeline", fontsize=15, fontweight="bold")

axes[0, 0].imshow(img_rgb);                                             axes[0, 0].set_title("Original image")
axes[0, 1].imshow(binary, cmap="gray");                                 axes[0, 1].set_title("Binary mask\n(input to findContours)")
axes[0, 2].imshow(cv2.cvtColor(all_contours_img, cv2.COLOR_BGR2RGB));  axes[0, 2].set_title(f"All contours drawn\n({len(contours)} found)")

axes[1, 0].imshow(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB));      axes[1, 0].set_title(f"Filtered (area > {MIN_AREA}px)\n{len(large_contours)} kept")
axes[1, 1].imshow(cv2.cvtColor(bounding_img, cv2.COLOR_BGR2RGB));      axes[1, 1].set_title("Bounding boxes\ncv2.boundingRect()")
axes[1, 2].imshow(cv2.cvtColor(canny_contour_img, cv2.COLOR_BGR2RGB)); axes[1, 2].set_title("Contours from Canny edges\n(alternative approach)")

for row in axes:
    for ax in row:
        ax.axis("off")

plt.tight_layout()
plt.show()
