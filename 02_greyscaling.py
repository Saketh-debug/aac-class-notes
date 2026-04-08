import cv2
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
IMAGE_PATH = "demo.jpg"
# ─────────────────────────────────────────────

img_bgr = cv2.imread(IMAGE_PATH)
img_bgr = cv2.resize(img_bgr, (800, 500))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — Correct greyscale using OpenCV (Luminosity Formula)
#  Formula:  Grey = 0.114×B + 0.587×G + 0.299×R
#  Green has the highest weight because the human eye is most sensitive to it.
# ─────────────────────────────────────────────────────────────────────────────
gray_correct = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

print(f"Original shape : {img_bgr.shape}")      # (H, W, 3)
print(f"Greyscale shape: {gray_correct.shape}")  # (H, W)  — no 3rd dimension


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — Manual average (WRONG way — for comparison only)
#  Equal weights give perceptually inaccurate results.
# ─────────────────────────────────────────────────────────────────────────────
b, g, r     = cv2.split(img_bgr)
gray_wrong  = ((b.astype(np.float32) + g.astype(np.float32) + r.astype(np.float32)) / 3).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — Show a single greyscale result live
# ─────────────────────────────────────────────────────────────────────────────
cv2.imshow("Greyscale (correct  luminosity formula)", gray_correct)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — Pixel value difference between correct and wrong conversion
# ─────────────────────────────────────────────────────────────────────────────
diff = cv2.absdiff(gray_correct, gray_wrong)
print(f"\nMax pixel difference (correct vs naive avg): {diff.max()}")
print("Bright pixels in the diff image = where the two methods disagree most.")


# ─────────────────────────────────────────────────────────────────────────────
#  END-OF-SECTION COMPARISON GRID
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(20, 7))
fig.suptitle("Greyscaling — Correct vs Naive Average", fontsize=15, fontweight="bold")

axes[0, 0].imshow(img_rgb);                  axes[0, 0].set_title("Original (colour)")
axes[0, 1].imshow(gray_correct, cmap="gray"); axes[0, 1].set_title("Correct\n(Luminosity formula)")
axes[1, 0].imshow(gray_wrong,   cmap="gray"); axes[1, 0].set_title("Naive average\n(Equal weights — wrong)")
axes[1, 1].imshow(diff,         cmap="hot");  axes[1, 1].set_title("Difference\n(Bright = disagreement)")

for row in axes:
    for ax in row:
        ax.axis("off")

plt.tight_layout()
plt.show()
