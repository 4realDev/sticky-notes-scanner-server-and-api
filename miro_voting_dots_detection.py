import math
import os
import numpy as np
import cv2

# source: https://solarianprogrammer.com/2015/05/08/detect-red-circles-image-using-opencv/
# IMPORTANT: currently only works for the red voting dots and does NOT work with pink or redish sticky notes

img_with_circles_count = 0


def get_voting_dots(img_path: str, save_folder: str, DEBUG=False):
    # reads image 'test_img.png' as grayscale
    img = cv2.imread(img_path)

    # Convert input image to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image, keep only the red pixels
    lower_red_hue_range = cv2.inRange(img_hsv, (0, 100, 100), (10, 255, 255))
    upper_red_hue_range = cv2.inRange(
        img_hsv, (150, 100, 100), (180, 255, 255))

    # Combine the above two images
    red_hue_image = cv2.addWeighted(
        lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0)
    red_hue_image_gaussian = cv2.GaussianBlur(red_hue_image, (9, 9), 2, 2)

    # Use the Hough transform to detect circles in the combined threshold image
    circles = []
    try:
        # https://dsp.stackexchange.com/questions/22648/in-opecv-function-hough-circles-how-does-parameter-1-and-2-affect-circle-detecti
        # param1 = will set sensitivity; how strong edges of circles need to be.
        # Too high and it won't detect anything, too low and it will find too much clutter.
        # param2 = will set how many edge points it needs to find to declare that it's found a circle.
        # Too high will detect nothing, too low will declare anything to be a circle.
        # The ideal value of param 2 will be related to the circumference of the circles.
        circles = cv2.HoughCircles(red_hue_image_gaussian, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=45, param2=25, minRadius=0, maxRadius=0)[0]
    except:
        return 0

    # https://www.geeksforgeeks.org/check-two-given-circles-touch-intersect/
    # Loop over all detected circles and outline them on the original image
    # mark the overlapping, false detected circles red and substract them from the detected circles count
    # and mark the correct detected circles green
    detected_circles = len(circles)
    if len(circles) > 0:
        i = 0
        while i < len(circles):
            x1 = round(circles[i][0])
            y1 = round(circles[i][1])
            r1 = round(circles[i][2])
            color = (0, 255, 0)
            k = i + 1
            while k < len(circles):
                x2 = round(circles[k][0])
                y2 = round(circles[k][1])
                r2 = round(circles[k][2])
                distance_between_circles = math.sqrt(
                    (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
                if distance_between_circles <= r1 + r2:
                    # subtract the intersecting circle to prevent false multiple circle detection
                    detected_circles -= 1
                    color = (0, 0, 255)
                k += 1
            i += 1

            img = cv2.circle(img, (x1, y1), r1, color, 5)

        if DEBUG:
            global img_with_circles_count
            cv2.imwrite(os.path.join(
                save_folder, f"image_with_circles_detection_{img_with_circles_count}.png"), img)
            img_with_circles_count += 1

    return detected_circles
