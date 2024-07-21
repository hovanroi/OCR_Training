import cv2
import pytesseract
import numpy as np

def rotate_image(thresh_img, debug=False):
    """ Rotate an image. Required input to be a binary image."""
    im_h, im_w = thresh_img.shape[0:2]
    if im_h > im_w:
        # Not yet implemented for vertical side code
        return thresh_img

    tmp = np.where(thresh_img > 0)
    row, col = tmp
    coords = np.column_stack((col, row))
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if debug:
        box_points = cv2.boxPoints(rect)
        box_points = np.int_(box_points)  # Changed np.int0 to np.int_
        debug_box_img = cv2.drawContours(thresh_img.copy(), [box_points], 0, (255, 255, 255), 2)
    
    if angle > 45:
        angle = 270 + angle

    (h, w) = thresh_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(thresh_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    return rotated

def is_contour_bad(c, src_img):
    im_h, im_w = src_img.shape[0:2]
    box = cv2.boundingRect(c)
    x, y, w, h = box[0], box[1], box[2], box[3]
    
    if im_w > im_h:
        if h >= 0.6 * im_h:
            return True
        if x < 0.4 * im_w and y > 0.6 * im_h:
            return True
        if w * h < 0.002 * im_h * im_w:
            return True
        if x <= 1 or x >= (im_w - 1) or y <= 1 or y >= (im_h - 1):
            if w * h < 0.05 * im_h * im_w:
                return True
    else:
        if w * h < 0.001 * im_h * im_w:
            return True
        if x + w >= 0.5 * im_w and y + h >= 0.4 * im_h:
            return True
    return False

def remove_noise(cnts, thresh, src_img, debug=False):
    mask = np.ones(thresh.shape[:2], dtype="uint8") * 255
    for c in cnts:
        if debug:
            box = cv2.boundingRect(c)
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(src_img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=1)
        if is_contour_bad(c, src_img):
            cv2.drawContours(mask, [c], -1, 0, -1)
    result = cv2.bitwise_and(thresh, thresh, mask=mask)
    return result

def otsu_threshold(src_img):
    #Convert color image to gray image
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    #Threshold process
    _, thresh_img1 = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_img2 = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 17)
    thresh_img = cv2.bitwise_and(thresh_img1, thresh_img2)
    mean_intensity = np.mean(thresh_img)
    if mean_intensity < 128:
        thresh_img = cv2.bitwise_not(thresh_img)
    return thresh_img

def make_sure_it_bbwt(thresh_img, depth=2):
    im_h, im_w = thresh_img.shape[0:2]
    total_pixel_value = np.sum(thresh_img)
    center_img = thresh_img[depth:im_h-depth, depth:im_w-depth]
    center_pixel_value = np.sum(center_img)
    border_bw_value = (total_pixel_value - center_pixel_value) / (im_h * im_w - center_img.size)
    if border_bw_value > 127:
        cv2.bitwise_not(thresh_img, thresh_img)

image_path = r"C:\Users\user\Desktop\Training OCR\Data\Example2.jpg"
image = cv2.imread(image_path) # reading the image
cv2.imshow("Input Image", image)

### Image processing
clean = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
cv2.imshow("Cleaned Image", clean)
thresh = otsu_threshold(clean)
cv2.imshow("Threshold Image", thresh)
rotated = rotate_image(thresh, debug=False)
cv2.imshow("Rotated Image", rotated)
cv2.waitKey(0)


### OCR Technology
# Set the path to the Tesseract executable (adjust as needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
text = pytesseract.image_to_string(rotated) # applying the ocr

# Print the text
print(text)