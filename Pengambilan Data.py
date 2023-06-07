import cv2
img_counter = 0
cam=cv2.VideoCapture(0)        #webcam
while True:
    key = cv2.waitKey(10) & 0xFF
    ret, frame=cam.read()
    cv2.imshow('webcam', frame)
    #if k%256 == 32:
    if key == ord('c'):
        # SPACE pressed
        img_name = "Tissue Front_{}.png".format(img_counter)
        cv2.imwrite(img_name,frame)
        print("{} written!".format(img_name))
        img_counter += 1
    elif cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
