from pathlib import Path

import cv2


project_folder = Path(__file__).resolve().parent
images_folder = project_folder / "images"
output_folder = project_folder / "output"

output_folder.mkdir(exist_ok=True)


# Load photographs
photo_a = cv2.imread(str(images_folder / "photo_a.jpg"))
photo_b = cv2.imread(str(images_folder / "photo_b.jpg"))

if photo_a is None or photo_b is None:
    raise FileNotFoundError("The input photographs could not be opened.")


# Convert photographs to grayscale
gray_a = cv2.cvtColor(photo_a, cv2.COLOR_BGR2GRAY)
gray_b = cv2.cvtColor(photo_b, cv2.COLOR_BGR2GRAY)


# Create the SIFT feature detector
sift = cv2.SIFT_create()


# Detect keypoints and calculate descriptors
keypoints_a, descriptors_a = sift.detectAndCompute(gray_a, None)
keypoints_b, descriptors_b = sift.detectAndCompute(gray_b, None)

print("Keypoints in Photo A:", len(keypoints_a))
print("Keypoints in Photo B:", len(keypoints_b))


# Draw keypoints
keypoint_image_a = cv2.drawKeypoints(
    photo_a,
    keypoints_a,
    None,
    color=(0, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

keypoint_image_b = cv2.drawKeypoints(
    photo_b,
    keypoints_b,
    None,
    color=(0, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)


# Save keypoint visualizations
cv2.imwrite(
    str(output_folder / "keypoints_a.jpg"),
    keypoint_image_a
)

cv2.imwrite(
    str(output_folder / "keypoints_b.jpg"),
    keypoint_image_b
)


# Match SIFT descriptors
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

print("Possible matches:", len(possible_matches))
print("Good matches:", len(good_matches))


# Display no more than 100 matches
matches_to_draw = good_matches[:100]

matching_image = cv2.drawMatches(
    photo_a,
    keypoints_a,
    photo_b,
    keypoints_b,
    matches_to_draw,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


# Save matching visualization
matches_path = output_folder / "feature_matches.jpg"

cv2.imwrite(
    str(matches_path),
    matching_image
)

print("Feature analysis completed!")
print("Results saved inside:", output_folder)