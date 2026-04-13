import cv2
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
IMAGE_PATH = r"C:\Users\LENOVO\Downloads\image_analysis_codes\demo.jpg"
# ─────────────────────────────────────────────

img_bgr = cv2.imread(IMAGE_PATH)
img_bgr = cv2.resize(img_bgr, (800, 500))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# IMPORTANT: all drawing functions modify the array IN-PLACE.
# Always work on a copy so you do not destroy the original.
canvas = img_bgr.copy()


# ─────────────────────────────────────────────────────────────────────────────
#  NOTE ON COLOUR
#  OpenCV colour is always BGR, NOT RGB.
#  Green  (0, 255, 0)     Blue  (255, 0, 0)     Red    (0, 0, 255)
#  White  (255, 255, 255) Black (0, 0, 0)        Yellow (0, 255, 255)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — Rectangle
#  cv2.rectangle(image, top-left corner, bottom-right corner, colour, thickness)
#  thickness = -1  →  fills the rectangle solid
# ─────────────────────────────────────────────────────────────────────────────
rect_outline = img_bgr.copy()
cv2.rectangle(rect_outline, (20, 150), (750, 450), (0, 255, 0), 3)   # outline, green

rect_filled = img_bgr.copy()
cv2.rectangle(rect_filled, (20, 150), (750, 450), (200, 250, 250), -1)   # filled, 

cv2.imshow("Rectangle outline (thickness=3)", rect_outline)
cv2.waitKey(0)
cv2.imshow("Rectangle filled (thickness=-1)", rect_filled)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — Circle
#  cv2.circle(image, centre, radius, colour, thickness)
# ─────────────────────────────────────────────────────────────────────────────
circle_outline = img_bgr.copy()
cv2.circle(circle_outline, (450, 350), 70, (200, 200, 255), 3)   # outline,

circle_filled = img_bgr.copy()
cv2.circle(circle_filled, (450, 350), 70, (200, 200, 255), -1)   # filled, 


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — Line
#  cv2.line(image, start point, end point, colour, thickness)
# ─────────────────────────────────────────────────────────────────────────────
line_img = img_bgr.copy()
cv2.line(line_img, (50, 50), (600, 400), (255, 0, 0), 4)   # blue line, 4px thick


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — Polygon (triangle, pentagon, any shape)
#  pts must be a NumPy array of shape (N, 1, 2) — N corner points
#  isClosed=True  connects the last point back to the first
# ─────────────────────────────────────────────────────────────────────────────
poly_img = img_bgr.copy()

# Triangle
triangle_pts = np.array([[300, 50], [150, 300], [450, 300]], dtype=np.int32)
triangle_pts = triangle_pts.reshape((-1, 1, 2))   # required shape for polylines
cv2.polylines(poly_img, [triangle_pts], isClosed=True, color=(0, 255, 255), thickness=3)

# Pentagon
pentagon_pts = np.array([[500, 100], [600, 180], [570, 300], [430, 300], [400, 180]], dtype=np.int32)
pentagon_pts = pentagon_pts.reshape((-1, 1, 2))
cv2.polylines(poly_img, [pentagon_pts], isClosed=True, color=(255, 0, 255), thickness=3)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 — Filled polygon
#  cv2.fillPoly fills the interior with a solid colour
# ─────────────────────────────────────────────────────────────────────────────
filled_poly = img_bgr.copy()

filled_triangle = np.array([[300, 50], [150, 300], [450, 300]], dtype=np.int32)
cv2.fillPoly(filled_poly, [filled_triangle], color=(0, 200, 100))   # filled green triangle
cv2.imshow("Filled triangle", filled_poly)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6 — Text
#  cv2.putText(image, text, origin, font, scale, colour, thickness)
#  Origin is the BOTTOM-LEFT corner of the text block.
# ─────────────────────────────────────────────────────────────────────────────
text_img = img_bgr.copy()
cv2.putText(text_img, "Hello OpenCV!", (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
cv2.putText(text_img, "AAC - MLOps", (50, 140),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)


# ─────────────────────────────────────────────────────────────────────────────
#  END-OF-SECTION COMPARISON GRID
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("Drawing Shapes in OpenCV", fontsize=15, fontweight="bold")

axes[0, 0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB));          axes[0, 0].set_title("Original image")
axes[0, 1].imshow(cv2.cvtColor(rect_outline, cv2.COLOR_BGR2RGB));     axes[0, 1].set_title("Rectangle — outline\ncv2.rectangle(..., thickness=3)")
axes[0, 2].imshow(cv2.cvtColor(rect_filled, cv2.COLOR_BGR2RGB));      axes[0, 2].set_title("Rectangle — filled\ncv2.rectangle(..., thickness=-1)")
axes[0, 3].imshow(cv2.cvtColor(circle_outline, cv2.COLOR_BGR2RGB));   axes[0, 3].set_title("Circle — outline\ncv2.circle(..., thickness=3)")

axes[1, 0].imshow(cv2.cvtColor(circle_filled, cv2.COLOR_BGR2RGB));    axes[1, 0].set_title("Circle — filled\ncv2.circle(..., thickness=-1)")
axes[1, 1].imshow(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB));         axes[1, 1].set_title("Line\ncv2.line()")
axes[1, 2].imshow(cv2.cvtColor(poly_img, cv2.COLOR_BGR2RGB));         axes[1, 2].set_title("Polygon outlines\ncv2.polylines()")
axes[1, 3].imshow(cv2.cvtColor(text_img, cv2.COLOR_BGR2RGB));         axes[1, 3].set_title("Text\ncv2.putText()")

for row in axes:
    for ax in row:
        ax.axis("off")

plt.tight_layout()
plt.show()
