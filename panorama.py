from pathlib import Path

import cv2


project_folder = Path(__file__).resolve().parent
images_folder = project_folder / "images"
output_folder = project_folder / "output"

photo_a_path = images_folder / "photo_a.jpg"
photo_b_path = images_folder / "photo_b.jpg"
panorama_path = output_folder / "first_panorama.jpg"

output_folder.mkdir(exist_ok=True)


# Load input images
photo_a = cv2.imread(str(photo_a_path))
photo_b = cv2.imread(str(photo_b_path))

if photo_a is None or photo_b is None:
    raise FileNotFoundError("One or both photographs could not be opened.")

print("Both photographs loaded successfully.")
print("Starting panorama stitching...")


# Create OpenCV panorama stitcher
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

status, panorama = stitcher.stitch([photo_a, photo_b])


# Check the result
if status == cv2.Stitcher_OK:
    cv2.imwrite(str(panorama_path), panorama)

    print("Panorama created successfully!")
    print("Saved at:", panorama_path)
else:
    print("Panorama stitching failed.")
    print("OpenCV status code:", status)

    if status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
        print("The photographs do not contain enough matching information.")
    elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
        print("OpenCV could not calculate a reliable homography.")
    elif status == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
        print("OpenCV could not estimate the camera parameters.")