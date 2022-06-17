import sys
import cv2

def lab(vdFile, nFrameFast, fTime):
    path = 'G://CUNY/CV/Final Project/Data/'
    cap = cv2.VideoCapture(path + vdFile)
    frameTime = 5   # time of each frame in ms
    nFrame = 0          # store the number of frame from beginning

    while(cap.isOpened()):
        print(nFrame)
        ret, frame = cap.read()
        cv2.putText(frame, "Frame #" + str(nFrame), (300, 200), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
        # set time of frame
        if nFrame > nFrameFast:
            frameTime = fTime
        cv2.imshow('frame',frame)
        nFrame += 1
        if cv2.waitKey(frameTime) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print('The label.py need 3 arguments: ')
        print('python label.py <video name> <the frame number before which run fast> <time between frames in ms>')
    else:
        lab(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))