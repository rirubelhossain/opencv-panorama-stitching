from pathlib import Path

import cv2
import numpy as np


project_folder = Path(__file__).resolve().parent
images_folder = project_folder / "images"
output_folder = project_folder / "output"

output_folder.mkdir(exist_ok=True)


# Load the photographs
photo_a = cv2.imread(str(images_folder / "photo_a.jpg"))
photo_b = cv2.imread(str(images_folder / "photo_b.jpg"))

if photo_a is None or photo_b is None:
    raise FileNotFoundError("The input photographs could not be opened.")


# Convert to grayscale
gray_a = cv2.cvtColor(photo_a, cv2.COLOR_BGR2GRAY)
gray_b = cv2.cvtColor(photo_b, cv2.COLOR_BGR2GRAY)


# Detect SIFT features
sift = cv2.SIFT_create()

keypoints_a, descriptors_a = sift.detectAndCompute(gray_a, None)
keypoints_b, descriptors_b = sift.detectAndCompute(gray_b, None)


# Find possible matches
matcher = cv2.BFMatcher(cv2.NORM_L2)

possible_matches = matcher.knnMatch(
    descriptors_a,
    descriptors_b,
    k=2
)


# Apply Lowe's ratio test
good_matches = []

for first_match, second_match in possible_matches:
    if first_match.distance < 0.75 * second_match.distance:
        good_matches.append(first_match)

print("Good matches before RANSAC:", len(good_matches))


if len(good_matches) < 10:
    raise RuntimeError("Not enough good matches to calculate homography.")


# Extract corresponding coordinates
points_a = np.float32([
    keypoints_a[match.queryIdx].pt
    for match in good_matches
]).reshape(-1, 1, 2)

points_b = np.float32([
    keypoints_b[match.trainIdx].pt
    for match in good_matches
]).reshape(-1, 1, 2)


# Calculate homography using RANSAC
homography, ransac_mask = cv2.findHomography(
    points_a,
    points_b,
    cv2.RANSAC,
    5.0
)

if homography is None:
    raise RuntimeError("The homography could not be calculated.")


# Count geometrically consistent matches
inlier_count = int(ransac_mask.sum())
inlier_percentage = 100 * inlier_count / len(good_matches)

print("RANSAC inliers:", inlier_count)
print(f"Inlier percentage: {inlier_percentage:.2f}%")

print("\nHomography matrix:")
print(homography)


# Keep only the first 100 matches for a clear visualization
number_to_draw = min(100, len(good_matches))

selected_matches = good_matches[:number_to_draw]
selected_mask = ransac_mask.ravel().tolist()[:number_to_draw]


# Draw only RANSAC-approved matches
ransac_visualization = cv2.drawMatches(
    photo_a,
    keypoints_a,
    photo_b,
    keypoints_b,
    selected_matches,
    None,
    matchColor=(0, 255, 0),
    singlePointColor=(0, 0, 255),
    matchesMask=selected_mask,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


output_path = output_folder / "ransac_matches.jpg"

cv2.imwrite(
    str(output_path),
    ransac_visualization
)

print("\nRANSAC visualization saved at:")
print(output_path)