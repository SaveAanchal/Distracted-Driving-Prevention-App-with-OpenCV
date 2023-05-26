import cv2
import numpy as np
import winsound
import random
frequency = 2500
duration = 1000
sounds = ['Sound1.wav','Sound2.wav','Sound3.wav','Sound4.wav','Sound5.wav']

# --------------------------------------------This code was found here: https://medium.com/@stepanfilonov/tracking-your-eyes-with-python-3952------------------

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector_params.filterByConvexity = False
detector_params.filterByInertia = False
detector = cv2.SimpleBlobDetector_create(detector_params)
start = False
facecheck = True
def detect_faces(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
        frame = img[y:y + h, x:x + w]
    return frame


def detect_eyes(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = cascade.detectMultiScale(gray_frame, 1.3, 5)  # detect eyes
    width = np.size(img, 1)  # get face frame width
    height = np.size(img, 0)  # get face frame height
    left_eye = None
    right_eye = None
    for (x, y, w, h) in eyes:
        if y > height / 2:
            pass
        eyecenter = x + w / 2  # get the eye center
        if eyecenter < width * 0.5:
            left_eye = img[y:y + h, x:x + w]
        else:
            right_eye = img[y:y + h, x:x + w]
    return left_eye, right_eye


def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)

    return img


def blob_process(img, threshold, detector):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2)
    img = cv2.dilate(img, None, iterations=4)
    img = cv2.medianBlur(img, 5)
    keypoints = detector.detect(img)
    #print(keypoints)
    return keypoints


def nothing(x):
    pass

#---------------------------------------------------------------The end of the code credited above-----------------------------------------------------------

#Define the check_eyes function with parameters face_frame, start, facecheck
def check_eyes(face_frame, start, facecheck):
    
#If facecheck is true, and there is a face in frame, find eyes
#Make a for each loop for eyes and check if there is an eye in frame
#Get the threshold from the threshold trackbar and cut eyebrows in the eye
#Draw keypoints on the eye and define the eye as the keypoint location
#If the user has started the program, check if the eye at 1,1,0 is less than or equal to 10 or the greater than or equal to 150
#If true return true, else false
#If there is not an eye in frame return true
#If there is no face in frame return true
#Do the same thing if facecheck is false, but if there is no face, don't return anything
    
    if facecheck:
        if face_frame is not None:
            eyes = detect_eyes(face_frame, eye_cascade)
            for eye in eyes:
                if eye is not None:
                    threshold = r = cv2.getTrackbarPos('threshold', 'image')
                    eye = cut_eyebrows(eye)
                    keypoints = blob_process(eye, threshold, detector)
                    eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    if start:
                        if eye[1][1][0] <= 10 or eye[1][1][0] >= 150:
                            return True
                        else:
                            return False
                else:
                    return True
        else:
            return True
    else:
        if face_frame is not None:
            eyes = detect_eyes(face_frame, eye_cascade)
            for eye in eyes:
                if eye is not None:
                    threshold = r = cv2.getTrackbarPos('threshold', 'image')
                    eye = cut_eyebrows(eye)
                    keypoints = blob_process(eye, threshold, detector)
                    eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    if start:
                        if eye[1][1][0] <= 10 or eye[1][1][0] >= 150:
                            return True
                        else:
                            return False
                else:
                    return True

#Define the result function with no parameters
def result():
        winsound.Beep(frequency, duration) #Beep
        filename = sounds[random.randint(0,4)] #Find a random sound from the list sounds
        winsound.PlaySound(filename, winsound.SND_FILENAME) #Play the sound

def main():
    #cap = cv2.VideoCapture(0) << Would be used in the case of a video capture
    cap = cv2.VideoCapture('Video.mp4')
    while(cap.isOpened()):
        cv2.namedWindow('image')
        cv2.createTrackbar('threshold', 'image', 0, 255, nothing)
        cv2.createTrackbar('start', 'image',0,1,nothing)
        cv2.createTrackbar('face check', 'image',0,1,nothing)
        counter = 0
        #Get trackbar position to see if the user has started running the program
        #Get trackbar position to see if the user would like facecheck to be on
        #Facecheck: If there are no faces in the frame, and facecheck is on, the program will warn you
        #Otherwise, if facecheck is off, the program will not
        #Check if the user has started the program
        while True:
            s = cv2.getTrackbarPos('start', 'image')
            f = cv2.getTrackbarPos('face check', 'image')
            if s == 1:
                start = True
            else:
                start = False
                
            _, frame = cap.read()
            face_frame = detect_faces(frame, face_cascade)

            #Check if facecheck is on
            #Increase the counter if check_eyes returns true, if it does not, set counter to 0
            if f == 1:              
                if check_eyes(face_frame, start, True):
                    counter = counter + 1
                else:
                    counter = 0
                
            else:
                if check_eyes(face_frame, start, False):
                    counter = counter + 1
                else:
                    counter = 0
            #If the user would like the program to be started and the counter is at least 10
            #Call result and set counter to 0
            if start == True and counter >= 10:
                result()
                counter = 0
            cv2.imshow('image', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
