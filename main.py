import cv2
import pytesseract
import imutils
import numpy as np


pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

image = cv2.imread('C:\\Users\\Antonio\\Desktop\\POZE VA\\NP85.jpg', cv2.IMREAD_COLOR)
#image = cv2.resize(image, (600, 400))

img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img_gray = cv2.bilateralFilter(img_gray, 13, 15, 15)

img_edge = cv2.Canny(img_gray, 30, 200)
find_contours = cv2.findContours(img_edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
find_contours = imutils.grab_contours(find_contours)
find_contours = sorted(find_contours, key=cv2.contourArea, reverse=True)[:10]
img_contour = None

for c in find_contours:

    perimeter = cv2.arcLength(c, True)
    approximation_poly = cv2.approxPolyDP(c, 0.018 * perimeter, True)

    if len(approximation_poly) == 4:
        img_contour = approximation_poly
        break

if img_contour is None:
    contour_found = False
    print("The contour of the image was not found")
else:
    contour_found = True

if contour_found == True:
    cv2.drawContours(image, [img_contour], -1, (0, 0, 255), 3)

img_mask = np.zeros(img_gray.shape, np.uint8)
new_img = cv2.drawContours(img_mask, [img_contour], 0, 255, -1, )
new_img = cv2.bitwise_and(image, image, mask=img_mask)

(x, y) = np.where(img_mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
roi_cropped = img_gray[topx:bottomx + 1, topy:bottomy + 1]

text = pytesseract.image_to_string(roi_cropped, config='--psm 11')
print("The license plate number is:", text)
img = cv2.resize(image, (500, 300))
cropped_img = cv2.resize(roi_cropped, (400, 200))
cv2.imshow('Original Image', image)
cv2.imshow('Final Result', roi_cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()