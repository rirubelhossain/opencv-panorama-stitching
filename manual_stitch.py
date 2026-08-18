from pathlib import Path

import cv2
import numpy as np


project_folder = Path(__file__).resolve().parent
images_folder = project_folder / "images"
output_folder = project_folder / "output"

output_folder.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Load images
# ---------------------------------------------------------

photo_a = cv2.imread(str(images_folder / "photo_a.jpg"))
photo_b = cv2.imread(str(images_folder / "photo_b.jpg"))

if photo_a is None or photo_b is None:
    raise FileNotFoundError("The input photographs could not be opened.")


# ---------------------------------------------------------
# Detect SIFT features
# ---------------------------------------------------------

gray_a = cv2.cvtColor(photo_a, cv2.COLOR_BGR2GRAY)
gray_b = cv2.cvtColor(photo_b, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

keypoints_a, descriptors_a = sift.detectAndCompute(gray_a, None)
keypoints_b, descriptors_b = sift.detectAndCompute(gray_b, None)


# ---------------------------------------------------------
# Match descriptors
# ---------------------------------------------------------

matcher = cv2.BFMatcher(cv2.NORM_L2)

possible_matches = matcher.knnMatch(
    descriptors_a,
    descriptors_b,
    k=2
)

good_matches = []

for first_match, second_match in possible_matches:
    if first_match.distance < 0.75 * second_match.distance:
        good_matches.append(first_match)

if len(good_matches) < 10:
    raise RuntimeError("Not enough matches to create a panorama.")

print("Good matches:", len(good_matches))


# ---------------------------------------------------------
# Calculate homography: Photo A -> Photo B
# ---------------------------------------------------------

points_a = np.float32([
    keypoints_a[match.queryIdx].pt
    for match in good_matches
]).reshape(-1, 1, 2)

points_b = np.float32([
    keypoints_b[match.trainIdx].pt
    for match in good_matches
]).reshape(-1, 1, 2)

homography, ransac_mask = cv2.findHomography(
    points_a,
    points_b,
    cv2.RANSAC,
    5.0
)

if homography is None:
    raise RuntimeError("Homography calculation failed.")

inliers = int(ransac_mask.sum())

print("RANSAC inliers:", inliers)


# ---------------------------------------------------------
# Calculate the required output canvas
# ---------------------------------------------------------

height_a, width_a = photo_a.shape[:2]
height_b, width_b = photo_b.shape[:2]

corners_a = np.float32([
    [0, 0],
    [width_a, 0],
    [width_a, height_a],
    [0, height_a]
]).reshape(-1, 1, 2)

corners_b = np.float32([
    [0, 0],
    [width_b, 0],
    [width_b, height_b],
    [0, height_b]
]).reshape(-1, 1, 2)


# Transform Photo A's corners into Photo B's coordinate system
transformed_corners_a = cv2.perspectiveTransform(
    corners_a,
    homography
)

all_corners = np.concatenate(
    (transformed_corners_a, corners_b),
    axis=0
)

minimum_x, minimum_y = np.floor(
    all_corners.min(axis=0).ravel()
).astype(int)

maximum_x, maximum_y = np.ceil(
    all_corners.max(axis=0).ravel()
).astype(int)

canvas_width = maximum_x - minimum_x
canvas_height = maximum_y - minimum_y


# Translation prevents negative coordinates
translation = np.array([
    [1, 0, -minimum_x],
    [0, 1, -minimum_y],
    [0, 0, 1]
], dtype=np.float64)

print("Canvas size:", canvas_width, "x", canvas_height)


# ---------------------------------------------------------
# Warp both photographs onto the common canvas
# ---------------------------------------------------------

warped_a = cv2.warpPerspective(
    photo_a,
    translation @ homography,
    (canvas_width, canvas_height)
)

warped_b = cv2.warpPerspective(
    photo_b,
    translation,
    (canvas_width, canvas_height)
)


# Create masks independently from image colours
mask_a_original = np.full(
    (height_a, width_a),
    255,
    dtype=np.uint8
)

mask_b_original = np.full(
    (height_b, width_b),
    255,
    dtype=np.uint8
)

mask_a = cv2.warpPerspective(
    mask_a_original,
    translation @ homography,
    (canvas_width, canvas_height)
)

mask_b = cv2.warpPerspective(
    mask_b_original,
    translation,
    (canvas_width, canvas_height)
)

mask_a = mask_a > 0
mask_b = mask_b > 0


# ---------------------------------------------------------
# Feather blending
# ---------------------------------------------------------

mask_a_uint8 = mask_a.astype(np.uint8) * 255
mask_b_uint8 = mask_b.astype(np.uint8) * 255

distance_a = cv2.distanceTransform(
    mask_a_uint8,
    cv2.DIST_L2,
    5
)

distance_b = cv2.distanceTransform(
    mask_b_uint8,
    cv2.DIST_L2,
    5
)

total_distance = distance_a + distance_b + 0.000001

weight_a = distance_a / total_distance
weight_b = distance_b / total_distance

weight_a = weight_a[:, :, np.newaxis]
weight_b = weight_b[:, :, np.newaxis]

manual_panorama = (
    warped_a.astype(np.float32) * weight_a
    + warped_b.astype(np.float32) * weight_b
)

manual_panorama = np.clip(
    manual_panorama,
    0,
    255
).astype(np.uint8)


# ---------------------------------------------------------
# Crop unused black space
# ---------------------------------------------------------

combined_mask = (mask_a | mask_b).astype(np.uint8) * 255

coordinates = cv2.findNonZero(combined_mask)

if coordinates is not None:
    x, y, width, height = cv2.boundingRect(coordinates)

    manual_panorama = manual_panorama[
        y:y + height,
        x:x + width
    ]


# ---------------------------------------------------------
# Save result
# ---------------------------------------------------------

output_path = output_folder / "feather_panorama.jpg"

success = cv2.imwrite(
    str(output_path),
    manual_panorama
)

if not success:
    raise RuntimeError("The panorama could not be saved.")

print("Manual panorama created successfully!")
print("Saved at:", output_path)