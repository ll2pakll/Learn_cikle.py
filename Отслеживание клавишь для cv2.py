import cv2
'''Код для получения номеров клавишь'''
img_path = 'd:\Work Area\Xseg_exstract\\frames\\0000.jpg'

img = cv2.imread(img_path) # load a dummy image
while(1):
    cv2.imshow('img', img)
    k = cv2.waitKey(33)
    if k==27:    # Esc key to stop
        break
    elif k==-1:  # normally -1 returned,so don't print it
        continue
    else:
        print (k) # else print its value