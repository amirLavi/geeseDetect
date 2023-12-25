import cv2

# print(cv2.__version__)
# Load the pre-trained Haarcascade classifier for detecting faces
goose_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the input image
image_path = './images/Amir.jpg'  # Replace with the path to your image
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect Canadian geese in the image
geese = goose_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around the detected geese
print(geese)
for (x, y, w, h) in geese:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the result
cv2.imshow('Canadian Geese Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()