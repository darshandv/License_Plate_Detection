# Author : Darshan D V

import cv2
import os

# Converts a video into a set of images and stores them in a file
def convert(filename,n_frames,im_format):
    video = cv2.VideoCapture(str(filename))
    os.system('mkdir '+str(filename[:-4])+'_images')
    os.chdir(os.getcwd()+'/'+str(filename[:-4])+'_images/')
    i=0
    j=0
    while True:
        ret,frame = video.read()
        if ret:
            if not j:
                cv2.imwrite(str(i)+'.'+str(im_format),frame)
                i=i+1
            j=(j+1)%n_frames
        else:
            break
    os.chdir('../')
    print ('Done')
    return i


convert('Traffic2.mp4',25,'jpg')
