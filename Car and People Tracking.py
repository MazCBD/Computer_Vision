# Car and People Tracking
# Note: we can use haar features xml files of cat, dogs, football etc

import cv2  # calling opencv

# Our image
img_file = "C:\\Users\\ASUS\\Desktop\\car_img.jpg"

# Our video
video = cv2.VideoCapture(r'C:\Users\ASUS\Desktop\human_cars.mp4')


# Pre-trained car and human classfier
car_tracker_file = r'C:\Users\ASUS\Desktop\car_detector.xml'
human_tracker_file = r'C:\Users\ASUS\Desktop\human_detector.xml'


# create car and human classifier (detecting the car in video/image, red box)
car_tracker = cv2.CascadeClassifier(car_tracker_file)

human_tracker = cv2.CascadeClassifier(human_tracker_file)

# Iterate over frames
while True:

    # Read the current frame
    (read_successful, frame) = video.read()

    # Safe coding
    if read_successful:
        # Must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # detect cars and human of any size and scale
    cars = car_tracker.detectMultiScale(grayscaled_frame)

    humans = human_tracker.detectMultiScale(grayscaled_frame)

    # Draw rectangles around the car with the coordinates and w,h above
    # for cars, [289 188 191 191] --> Array index 0 & Loop over these
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x+1, y+2), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    for (a, b, c, d) in humans:
        cv2.rectangle(frame, (a, b), (a+c, b+d), (0, 255, 255), 2)

    # Display the image with the faces spotted
    cv2.imshow('Detecting Cars and Humans', frame)

    # Don't autoclose (wait here in the code and listen for a key press)
    key = cv2.waitKey(1)

    # Stop if Q key is pressed, for lowecase is 81 and uppercase is 113
    if key == 81 or key == 113:
        break

# Release the VideoCapture object
video.release()

print("Code Completed")

'''

# create opencv image
img = cv2.imread(img_file)

# convert image to grayscale for haar cascade
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# create car classifier (detecting the car in image, red box)
car_tracker = cv2.CascadeClassifier(classifier_file) 

# detect cars of any size and scale
cars = car_tracker.detectMultiScale(black_n_white)


print(cars) # give us the coordinates(x,y) and width, height of the car in image 


# Draw rectangles around the car with the coordinates and w,h above
for (x,y,w,h) in cars: # [289 188 191 191] --> Array index 0 & Loop over these
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255),2)



car1 = cars[0] # the x,y and w,h of the first car
car2 = cars[1] 

(x,y,w,h) = car2 # unpacking 
cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255),2)


# Display the img
cv2.imshow('Car Detector',img)

# Don't autoclose (wait here in the code and listen for a key press)
cv2.waitKey()
'''
