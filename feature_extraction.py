import cv2
import numpy as np


def extract_features(image):
    # convert to LAB image
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # split channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # blur the channel
    blur_image = cv2.GaussianBlur(b_channel, (5, 5), 0)

    # Apply Otsu thresholding to image
    ret, thresh = cv2.threshold(
        blur_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, hierarchy = cv2.findContours(
        opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, cnt in enumerate(contours):
        # get contour area
        area = round(cv2.contourArea(cnt), 2)

        # get contour length (blob perimeter)
        perimeter = round(cv2.arcLength(cnt, True), 2)

        # circularity = 4 * pi * area / (perimeter)**2
        try:
            circularity = round(
                (4 * 3.14 * area) / (perimeter**2), 2)
        except ZeroDivisionError:
            circularity = 0

        # get contour hull and area (convexity)
        hull = cv2.convexHull(cnt, False)
        convex_area = cv2.contourArea(hull, True)

        try:
            convexity = round(area / convex_area, 2)
        except ZeroDivisionError:
            circularity = 0

    # Apply mask on image
    mask = (cv2.merge([opened, opened, opened]))
    segmented_image = 255 * (mask * image)

    # Color moments feature extraction
    red_mean, green_mean, blue_mean = np.mean(image, axis=(0, 1))
    red_std, green_std, blue_std = np.std(image, axis=(0, 1))

    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    red_skew = np.mean((R - red_mean)**3) / (red_std**3)
    green_skew = np.mean((G - green_mean)**3) / (green_std**3)
    blue_skew = np.mean((B - blue_mean)**3) / (blue_std**3)

    red_kurt = np.mean((R - red_mean)**4) / (red_std**4)
    green_kurt = np.mean((G - green_mean)**4) / (green_std**4)
    blue_kurt = np.mean((B - blue_mean)**4) / (blue_std**4)

    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h_mean, s_mean, v_mean = np.mean(hsv_img, axis=(0, 1))
    h_std, s_std, v_std = np.std(hsv_img, axis=(0, 1))

    H = hsv_img[:, :, 0]
    S = hsv_img[:, :, 1]
    V = hsv_img[:, :, 2]

    h_skew = np.mean((H - h_mean)**3) / (h_std**3)
    s_skew = np.mean((S - s_mean)**3) / (s_std**3)
    v_skew = np.mean((V - v_mean)**3) / (v_std**3)

    h_kurt = np.mean((H - h_mean)**4) / (h_std**4)
    s_kurt = np.mean((S - s_mean)**4) / (s_std**4)
    v_kurt = np.mean((V - v_mean)**4) / (v_std**4)

    feature_vector = {
        'area': area,
        'perimeter': perimeter,
        'circularity': circularity,
        'convexity': convexity,
        'red_mean': red_mean,
        'green_mean': green_mean,
        'blue_mean': blue_mean,
        'red_std': red_std,
        'green_std': green_std,
        'blue_std': blue_std,
        'red_skew': red_skew,
        'green_skew': green_skew,
        'blue_skew': blue_skew,
        'red_kurt': red_kurt,
        'green_kurt': green_kurt,
        'blue_kurt': blue_kurt,
        'h_mean': h_mean,
        's_mean': s_mean,
        'v_mean': v_mean,
        'h_std': h_std,
        's_std': s_std,
        'v_std': v_std,
        'h_skew': h_skew,
        's_skew': s_skew,
        'v_skew': v_skew,
        'h_kurt': h_kurt,
        's_kurt': s_kurt,
        'v_kurt': v_kurt
    }

    return segmented_image, feature_vector
