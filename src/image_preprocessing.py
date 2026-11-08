
import cv2
import numpy as np


def preprocess_image(image):
    """
    Preprocess a handwritten digit image for the trained SVM model.

    Input:
        image: PIL Image or NumPy image array

    Output:
        NumPy array of shape (1, 784), with pixel values from 0 to 255.
    """

    # 1. Convert input image to NumPy array
    if hasattr(image, "convert"):
        image = np.array(image.convert("RGB"))
    else:
        image = np.asarray(image)

    # 2. Convert image to grayscale
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 3. Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # 4. Apply Otsu thresholding
    _, thresholded = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 5. Detect contours
    contours, _ = cv2.findContours(
        thresholded,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError("No handwritten digit detected.")

    # 6. Select the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest_contour) < 10:
        raise ValueError("No clear handwritten digit detected.")

    # 7. Crop the digit using its bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    digit = thresholded[y:y + h, x:x + w]

    # 8. Resize while maintaining aspect ratio
    scale = 20.0 / max(w, h)

    new_width = max(1, int(round(w * scale)))
    new_height = max(1, int(round(h * scale)))

    resized_digit = cv2.resize(
        digit,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    # 9. Place the digit in the centre of a 28x28 black canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_width) // 2
    y_offset = (28 - new_height) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized_digit

    # 10. Calculate the centre of mass
    moments = cv2.moments(canvas)

    if moments["m00"] != 0:
        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]

        shift_x = int(round(13.5 - center_x))
        shift_y = int(round(13.5 - center_y))

        # 11. Shift digit towards the centre
        transformation = np.float32([
            [1, 0, shift_x],
            [0, 1, shift_y]
        ])

        canvas = cv2.warpAffine(
            canvas,
            transformation,
            (28, 28)
        )

    # 12. Slightly increase stroke thickness
    kernel = np.ones((2, 2), dtype=np.uint8)

    canvas = cv2.dilate(
        canvas,
        kernel,
        iterations=1
    )

    # 13. Flatten into 784 features
    result = canvas.reshape(1, 784)

    return result
