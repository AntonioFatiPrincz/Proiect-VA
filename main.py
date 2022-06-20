import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract'
# Citire imagine
img = cv2.imread('NP2.JPG')

# Grayscale Image
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detectie margini
edge_detect = cv2.Canny(img_gray, 170, 200)
cv2.imshow("EDGE",edge_detect)

find_contours, new = cv2.findContours(edge_detect.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
find_contours = sorted(find_contours, key=cv2.contourArea, reverse=True)[:30]


# Conturul placutei de inmatriculare
license_contour = None
license_plate_image = None # imaginea conturului placutei

# Coordonate imagine placuta de inmatriculare
x = None
y = None
l = None #latime
i = None #inaltime

# Aici iteram prin toate contururile gasite anterior si incercam sa gasim
# acele contururi care au 4 colturi

for c in find_contours:

    # aici aflam perimetrul conturului
    the_perim = cv2.arcLength(c,True) # gasire perimetru
    approx_shape_poly = cv2.approxPolyDP(c, 0.03 * the_perim, True) # aici se verifica forma conturului/ daca este un poligon
    # este folosita o acuratete de 0.013

    # verificam daca este un poligon inchis
    if len(approx_shape_poly) == 4:
        license_contour = approx_shape_poly
        x,y,l,i = cv2.boundingRect(c)
        license_plate_image = img_gray[y:y+i, x:x+l]
        break

# Se indeparteaza Noise-ul din cadrul imaginii pentru a sporii claritatea acestea
#eliminare noise
license_plate_image = cv2.bilateralFilter(license_plate_image,11,17,17)
# eliminare culori si accentuare contrast
(thresh, license_plate_image) = cv2.threshold(license_plate_image, 150, 255,cv2.THRESH_BINARY)

# Detectie text
license_text = pytesseract.image_to_string(license_plate_image)

# evidentiere placuta
img = cv2.rectangle(img, (x,y), (x+l,y+i), (10,120,230), 3)
img = cv2.putText(img, license_text, (x-60, y-50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 6, cv2.LINE_AA)

print("Text detectat :", license_text)

cv2.imshow("Detectat", img)
cv2.waitKey(0)










