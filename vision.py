import cv2
import numpy as np
import matplotlib.pyplot as plt

#reading in the input image
plate = cv2.imread("polo.webp")



#function that shows the image
def display(img, cmap = 'gray'):
    fig = plt.figure(figsize = (12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap = 'gray')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("boxed.jpg",img)

def detect_plate(img):
    
    boxed_img = plate.copy()
    
    #gets the points of where the classifier detects a plate
    plate_rects = plate_cascade.detectMultiScale(boxed_img, scaleFactor = 1.2, minNeighbors = 10)

    #draws the rectangle around it
    for (x,y,w,h) in plate_rects:
        cv2.rectangle(boxed_img, (x,y), (x+w, y+h), (255,0,0), 5)

    return boxed_img

#detects the plate and zooms in on it
def detect_zoom_plate(img, kernel):
    
    boxed_img = img.copy()
    
    #gets the points of where the classifier detects a plate
    plate_rects = plate_cascade.detectMultiScale(boxed_img, scaleFactor = 1.2, minNeighbors = 10)
    
    for (x,y,w,h) in plate_rects:
        x_offset = x
        y_offset = y
        
        x_end = x+w
        y_end = y+h
        
        #getting the points that show the license plate
        zoom_img = boxed_img[y_offset:y_end, x_offset:x_end]
        #increasing the size of the image
        zoom_img = cv2.resize(zoom_img, (0,0),fx = 2, fy = 2)
        zoom_img = zoom_img[7:-7, 7:-7]
        #sharpening the image to make it look clearer
        zoom_img = cv2.filter2D(zoom_img, -1, kernel)
        
        zy = (40 - (y_end - y_offset))//2
        zx = (144 - (x_end-x_offset))//2
        
        ydim = (y_end+zy-50) - (y_offset-zy-50)
        xdim = (x_end+zx) - (x_offset-zx)
       
        zoom_img = cv2.resize(zoom_img,(xdim,ydim))
        
        #putting the zoomed in image above where the license plate is located
        try:
            boxed_img[y_offset-zy-55:y_end+zy-55, x_offset-zx:x_end+zx] = zoom_img
        except:
            pass
         
        #drawing a rectangle
        for (x,y,w,h) in plate_rects:
            cv2.rectangle(boxed_img, (x,y), (x+w, y+h), (255,0,0), 2)
            
    return boxed_img


def get_plate(img):
    
    boxed_img = img.copy()
    plate_rects = plate_cascade.detectMultiScale(boxed_img, scaleFactor = 1.2, minNeighbors = 3)

    for (x,y,w,h) in plate_rects:
        x_offset = x
        y_offset = y
    
        x_end = x+w
        y_end = y+h
        
        zoom_img = boxed_img[y_offset:y_end, x_offset:x_end]
        boxed_img[y_offset:y_end, x_offset:x_end] = zoom_img
        
        for (x,y,w,h) in plate_rects:
            cv2.rectangle(boxed_img, (x,y), (x+w, y+h), (255,0,0), 5)

    cv2.imwrite("plate.jpeg", zoom_img)
    return boxed_img

#need to change color of picture from BGR to RGB
plate = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)

#Cascade Classifier where there are hundreds of samples of license plates
plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

result = detect_plate(plate)
    
#matrix needed to sharpen the image
kernel = np.array([[-1,-1,-1],
                   [-1,9,-1],
                   [-1,-1,-1]])
    
result = detect_zoom_plate(plate, kernel)
display(result)

result = get_plate(plate)

