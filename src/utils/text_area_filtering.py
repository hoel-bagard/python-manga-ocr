import logging

import cv2
import numpy as np

from src.utils.connected_components import get_cc_average_size, get_connected_components

# def text_like_histogram(img, area, average_size):
#     if not arg.boolean_value('additional_filtering'):
#         return True
#     (x, y, w, h) = dimensions_2d_slice(area)
#     x_subimage = np.copy(img)
#     x_histogram = np.zeros(w,int)
#     y_subimage = np.copy(img)
#     y_histogram = np.zeros(h,int)

#     aoi = img[area]

#     ccs = cc.get_connected_components(aoi)
#     if( len(ccs) < 2):
#         return False

#     #avg = average_size
#     avg = cc.average_size(aoi)
#     mean_width = cc.mean_width(aoi)
#     mean_height = cc.mean_height(aoi)
#     if arg.boolean_value('verbose'):
#         print('average size = ' + str(avg) + ' mean width = ' + str(mean_width) + ' mean height = ' + str(mean_height))
#     if math.isnan(avg) or avg==0:
#         if arg.boolean_value('verbose'):
#             print('Rejecting area since average size is NaN')
#         #return False

#     #in a text area, the average size of a blob (cc) will reflect
#     #that of the used characters/typeface. Thus if there simply aren't
#     #enough pixels per character, we can drop this as a text candidate
#     #note the following is failing in odd situations, probably due to incorrect
#     #calculation of 'avg size'
#     #TODO: replace testing against "average size" with testing against
#     #hard thresholds for connected component width and height. i.e.
#     #if they're all thin small ccs, we can drop this area

#     #if avg < defaults.MINIMUM_TEXT_SIZE_THRESHOLD:
#     if mean_width < defaults.MINIMUM_TEXT_SIZE_THRESHOLD or \
#         mean_height < defaults.MINIMUM_TEXT_SIZE_THRESHOLD:
#         if arg.boolean_value('verbose'):
#             print('Rejecting area since average width or height is less than threshold.')
#         return False

#     #check the basic aspect ratio of the ccs
#     if mean_width/mean_height < 0.5 or mean_width/mean_height > 2:
#         if arg.boolean_value('verbose'):
#             print('Rejecting area since mean cc aspect ratio not textlike.')
#         return False

#     width_multiplier = float(avg)
#     height_multiplier = float(avg)

#     #gaussian filter the subimages in x,y directions to emphasise peaks and troughs
#     x_subimage  = scipy.ndimage.filters.gaussian_filter(x_subimage,(0.01*width_multiplier,0))
#     y_subimage  = scipy.ndimage.filters.gaussian_filter(y_subimage,(0,0.01*height_multiplier))

#     #put together the histogram for black pixels over the x directon (along columns) of the component
#     for i,col in enumerate(range(x,x+w)):
#         black_pixel_count = np.count_nonzero(y_subimage[y:y+h,col])
#         x_histogram[i] = black_pixel_count

#     #and along the y direction (along rows)
#     for i,row in enumerate(range(y,y+h)):
#         black_pixel_count = np.count_nonzero(x_subimage[row,x:x+w])
#         y_histogram[i] = black_pixel_count

#     h_white_runs = get_white_runs(x_histogram)
#     num_h_white_runs = len(h_white_runs)
#     h_black_runs = get_black_runs(x_histogram)
#     num_h_black_runs = len(h_black_runs)
#     (h_spacing_mean, h_spacing_variance) = slicing_list_stats(h_white_runs)
#     (h_character_mean, h_character_variance) = slicing_list_stats(h_black_runs)
#     v_white_runs = get_white_runs(y_histogram)
#     num_v_white_runs = len(v_white_runs)
#     v_black_runs = get_black_runs(y_histogram)
#     num_v_black_runs = len(v_black_runs)
#     (v_spacing_mean, v_spacing_variance) = slicing_list_stats(v_white_runs)
#     (v_character_mean, v_character_variance) = slicing_list_stats(v_black_runs)

#     if arg.boolean_value('verbose'):
#         print('x ' + str(x) + ' y ' +str(y) + ' w ' + str(w) + ' h ' + str(h))
#         print('white runs ' + str(len(h_white_runs)) + ' ' + str(len(v_white_runs)))
#         print('white runs mean ' + str(h_spacing_mean) + ' ' + str(v_spacing_mean))
#         print('white runs std  ' + str(h_spacing_variance) + ' ' + str(v_spacing_variance))
#         print('black runs ' + str(len(h_black_runs)) + ' ' + str(len(v_black_runs)))
#         print('black runs mean ' + str(h_character_mean) + ' ' + str(v_character_mean))
#         print('black runs std  ' + str(h_character_variance) + ' ' + str(v_character_variance))

#     if num_h_white_runs < 2 and num_v_white_runs < 2:
#         if arg.boolean_value('verbose'):
#             print('Rejecting area since not sufficient amount post filtering whitespace.')
#         return False

#     if v_spacing_variance > defaults.MAXIMUM_VERTICAL_SPACE_VARIANCE:
#         if arg.boolean_value('verbose'):
#             print('Rejecting area since vertical inter-character space variance too high.')
#         return False

#     if v_character_mean < avg*0.5 or v_character_mean > avg*2.0:
#         pass
#         #return False
#     if h_character_mean < avg*0.5 or h_character_mean > avg*2.0:
#         pass
#         #return False

#     return True


def filter_text_like_areas(img: np.ndarray, segmentation: np.ndarray, average_size: float, logger: logging.Logger):
    # See if a given rectangular area (2d slice) is very text like
    # First step is to estimate furigana like elements so they can be masked
    furigana_areas = furigana.estimate_furigana(img, segmentation)
    furigana_mask = np.array(furigana_areas == 0, 'B')

    # Binarize the image, clean it via the segmentation and remove furigana too
    binary_threshold = 190
    logger.debug(f"binarizing images with threshold value of {binary_threshold}")
    _, binary = cv2.threshold(img, binary_threshold, 255, cv2.THRESH_BINARY_INV)

    binary_average_size = get_cc_average_size(binary)
    logger.debug(f"average cc size for binaryized grayscale image is {binary_average_size:.2f}")
    segmentation_mask = np.array(segmentation != 0, 'B')
    cleaned = binary * segmentation_mask * furigana_mask

    areas = get_connected_components(segmentation)
    text_like_areas = []
    nontext_like_areas = []
    for area in areas:
        if text_like_histogram(cleaned, area, average_size):
            text_like_areas.append(area)
        else:
            nontext_like_areas.append(area)

    return (text_like_areas, nontext_like_areas)

