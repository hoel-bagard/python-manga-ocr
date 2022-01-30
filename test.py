from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import scipy


def show_img(img: np.ndarray, window_name: str = "Image"):
    """Displays an image until the user presses the "q" key.

    Args:
        img: The image that is to be displayed.
        window_name (str): The name of the window in which the image will be displayed.
    """
    while True:
        # Make the image full screen if it's above a given size (assume the screen isn't too small^^)
        if any(img.shape[:2] > np.asarray([1080, 1440])):
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.imshow(window_name, img)
        key = cv2.waitKey(10)
        if key == ord("q"):
            cv2.destroyAllWindows()
            break


def text_detect(img):
    img = img.copy()
    show_img(img)
    # text cropping rectangle
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.Sobel(img, cv2.CV_8U, 1, 0)  # same as default,None,3,1,0,cv2.BORDER_DEFAULT)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    show_img(img)

    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    show_img(img)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    show_img(img)
    exit()
    # noiseSizeParam = int(ele_size[0]/3)
    # contours = [i for i in contours if i.shape[0] > noiseSizeParam**2]
    # rect = [cv2.boundingRect(i) for i in contours]  # no padding, box    #x,y,w,h
    # rect_p = [(max(int(i[0]-noiseSizeParam), 0),
    #            max(int(i[1]-noiseSizeParam), 0),
    #            min(int(i[0]+i[2]+noiseSizeParam), img.shape[1]),
    #            min(int(i[1]+i[3]+noiseSizeParam), img.shape[0])) for i in rect]  # with padding, box  x1,y1,x2,y2

    # return rect_p, rect


import connected_components as cc
import defaults


def form_canny_mask(img, mask=None):
    edges = cv2.Canny(img, 128, 255, apertureSize=3)
    if mask is not None:
        mask = mask*edges
    else:
        mask = edges
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    temp_mask = np.zeros(img.shape,np.uint8)
    for c in contours:
        # also draw detected contours into the original image in green
        # cv2.drawContours(img,[c],0,(0,255,0),1)
        hull = cv2.convexHull(c)
        cv2.drawContours(temp_mask,[hull],0,255,-1)
        # cv2.drawContours(temp_mask,[c],0,255,-1)
        # polygon = cv2.approxPolyDP(c,0.1*cv2.arcLength(c,True),True)
        # cv2.drawContours(temp_mask,[polygon],0,255,-1)
    return temp_mask


def clean_page(img, max_scale=defaults.CC_SCALE_MAX, min_scale=defaults.CC_SCALE_MIN):
    # img = cv2.imread(sys.argv[1])
    h, w, d = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # create gaussian filtered and unfiltered binary images
    sigma = 1.5
    print('Binarizing image with sigma value of ' + str(sigma))

    def binarize(img, threshold=190, white=255):
        (t, binary) = cv2.threshold(img, threshold, white, cv2.THRESH_BINARY_INV )
        return binary

    print("AAAAAAAAAAAAAAAAA")
    show_img(gray)
    gaussian_filtered = scipy.ndimage.gaussian_filter(gray, sigma=sigma)
    show_img(gaussian_filtered)
    binary_threshold = 190
    print('Binarizing image with sigma value of ' + str(sigma))
    gaussian_binary = binarize(gaussian_filtered, threshold=binary_threshold)
    show_img(gaussian_binary)
    binary = binarize(gray, threshold=binary_threshold)

    # exit()
    # Draw out statistics on average connected component size in the rescaled, binary image
    average_size = cc.average_size(gaussian_binary)
    # print('Initial mask average size is ' + str(average_size))
    max_size = average_size*max_scale
    min_size = average_size*min_scale

    # primary mask is connected components filtered by size
    mask = cc.form_mask(gaussian_binary, max_size, min_size)

    # secondary mask is formed from canny edges
    canny_mask = form_canny_mask(gaussian_filtered, mask=mask)

    # final mask is size filtered connected components on canny mask
    final_mask = cc.form_mask(canny_mask, max_size, min_size)

    # apply mask and return images
    cleaned = cv2.bitwise_not(final_mask * binary)
    return (cv2.bitwise_not(binary), final_mask, cleaned)


def get_blurbs(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.bitwise_not(cv2.adaptiveThreshold(img_gray, 255, cv2.THRESH_BINARY,
                                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 75, 10))

    kernel = np.ones((2, 2), np.uint8)
    img_gray = cv2.erode(img_gray, kernel, iterations=2)
    img_gray = cv2.bitwise_not(img_gray)
    contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    pruned_contours = []
    mask = np.zeros_like(img)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    height, width, channel = img.shape

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100 and area < ((height / 3) * (width / 3)):
            pruned_contours.append(cnt)

    # find contours for the mask for a second pass after pruning the large and small contours
    cv2.drawContours(mask, pruned_contours, -1, (255, 255, 255), 1)
    # show_img(mask)
    contours2, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask, pruned_contours, -1, (255, 255, 255), 1)

    # mask = np.zeros_like(img)
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # cv2.drawContours(mask, contours2, -1, (255, 255, 255), 1)
    # show_img(mask)

    final_mask = cv2.cvtColor(np.zeros_like(img), cv2.COLOR_BGR2GRAY)

    for cnt in contours2:
        area = cv2.contourArea(cnt)
        if area > 1000 and area < ((height / 3) * (width / 3)):
            draw_mask = cv2.cvtColor(np.zeros_like(img), cv2.COLOR_BGR2GRAY)
            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
            # pickle.dump(approx, open("approx.pkl", mode="w"))
            cv2.fillPoly(draw_mask, [approx], (255, 0, 0))
            cv2.fillPoly(final_mask, [approx], (255, 0, 0))
            image = cv2.bitwise_and(draw_mask, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            # draw_mask_inverted = cv2.bitwise_not(draw_mask)
            # image = cv2.bitwise_or(image, draw_mask_inverted)
            y = approx[:, 0, 1].min()
            h = approx[:, 0, 1].max() - y
            x = approx[:, 0, 0].min()
            w = approx[:, 0, 0].max() - x
            image = image[y:y+h, x:x+w]
            # show_img(image)

    show_img(final_mask)


def main():
    parser = ArgumentParser(description="Download script for the unsplash dataset")
    parser.add_argument("img_path", type=Path, help="Path to the dataset")
    args = parser.parse_args()

    img_path: Path = args.img_path
    # img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(str(img_path))

    # get_blurbs(img)

    show_img(img)
    binary, mask, cleaned = clean_page(img)
    # show_img(binary)
    # show_img(mask)
    show_img(cleaned)


if __name__ == "__main__":
    main()
