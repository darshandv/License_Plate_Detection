import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

video = cv2.VideoCapture('Traffic2.mp4')

def radian_to_degree(angle):
    return angle*180/np.pi

def order_points(pts):
    ''' 
        The function returns the 4 corner points in a set of points and returns the rectangle formed by the four
        co-ordinates. 
        
        The input are a set of points (pts).
        
        'crop' is the amount of pixels that are subtracted from the points obtained which is kind of padding 
        so that we don't lose the data that run outside the area specified.
        
        'main' is the flag which exclusively runs when the ordering points are of the main form. If main is 1, then
        the cropping should be more (This is specific to the form that we have chosen)
    '''
    rect = np.zeros((4, 2), dtype = "float32")
 

    s = pts.sum(axis = 1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
 

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    rect[1][0]+=10
    rect[2][0]+=10

    return rect

def four_point_warp(image, pts):
    ''' 
        The function finds the corner four points of the image and warps the image accordingly.
        
        image : The grayscale image of the scanned form.
        
        pts   : The set of points from which corner points are to be selected to warp.
        
    '''

    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Finding width and height of the image based on corner points
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # The destination where the image needs to be pasted. 
    # This is the size of the image where the warped image will be put.
    dst = np.array([
              [0, 0],
              [maxWidth - 1, 0],
              [maxWidth - 1, maxHeight - 1],
              [0, maxHeight - 1]], dtype = "float32")
    
    # Warping the image
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
    return np.array(rect),warped

area_range = [200,4000]
aspect_ratio_range = [2.4,9]
angle_thresh = 10
edge_density_threshold = 0.35
i=0
n_frames = 1
counts=[]
while(1):
    ret,frame = video.read()
    image = frame
    if((not i) and ret):
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        gray_copy = np.copy(gray)

        gray = cv2.GaussianBlur(gray, (3,3), 0)
        edge_im = cv2.Sobel(gray, -1, 1, 0)
        h,sobel = cv2.threshold(edge_im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11,3))
        gray = cv2.morphologyEx(sobel, cv2.MORPH_CLOSE, kernel)

        cnt_image,cnts, hier =cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = sorted(cnts,key=cv2.contourArea,reverse=True)

        for contour in cnts:
            rect = cv2.minAreaRect(contour)
            img_width = gray.shape[1]
            img_height = gray.shape[0]
            area = img_width*img_height

            box = cv2.boxPoints(rect) 
            box = np.int0(box)

            X = rect[0][0]
            Y = rect[0][1]
            angle = rect[2] 
            width = rect[1][0]
            height = rect[1][1]

            angle = (angle + 180) if width < height else (angle + 90)


            if (width > 0 and height > 0) and \
                ((width < img_width/2.0) and \
                 (height < img_width/2.0)):
                aspect_ratio = float(width)/height if width > height else float(height)/width

                if (aspect_ratio >= aspect_ratio_range[0] and  aspect_ratio <= aspect_ratio_range[1]):
                    if((height*width > area_range[0]) and  (height*width < area_range[1])):
                        box_list = list(box)
                        random_point = box_list[0]
                        del(box_list[0])
                        distances = [((point[0]-random_point[0])**2 + (point[1]-random_point[1])**2) for point in box_list]
                        sorted_distances = sorted(distances)
                        adjacent_far_point = box_list[distances.index(sorted_distances[1])]
                        tmp_angle = 90

                        if abs(random_point[0]-adjacent_far_point[0]) > 0:
                            tmp_angle = abs(float(random_point[1]-adjacent_far_point[1]))/                                         abs(random_point[0]-adjacent_far_point[0])
                            tmp_angle = radian_to_degree(math.atan(tmp_angle))


                        if tmp_angle <= angle_thresh:
                            rect,warped = four_point_warp(image, box)
                            warped_gray = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
                            h,thresh = cv2.threshold(warped_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                            thresh = cv2.bitwise_not(thresh)

#                             black_pixels=0
#                             for i in range(thresh.shape[0]):
#                                     for j in range(thresh.shape[1]):
#                                         if thresh[i][j] == 0:
#                                             black_pixels += 1

#                             edge_density = float(black_pixels)/(thresh.shape[0]*thresh.shape[1])
#         #                     print edge_density
#                             if edge_density > edge_density_threshold :
#                                 count = 0
                            need,contrs,hier=cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                            count = 0
                            for contour in contrs:
#                                     rectangle = cv2.minAreaRect(contour)
#                                     box2 = cv2.boxPoints(rectangle)
#                                     box2 = np.int0(box2)
                                a = cv2.contourArea(contour)
                                t = warped.shape[0]*warped.shape[1]
                                if a > t/30 and a<t/5:
                                    count= count+1
                            #counts.append(count)
                            if count > 0:
                                cv2.drawContours(image, [box], 0, (255,0,0),3)
#                         cv2.drawContours(image, [box], 0, (255,0,0),2)
    if ret:
        cv2.imshow('video',image)
    i=(i+1)%n_frames    
    if cv2.waitKey(1) & 0xff == ord('q'):
            break
video.release()

#The below code is the implementation of other algorithms devised during experimentation. It's for coder reference.

#print sum(counts)/len(counts)


# to_identify = 4
# k=1/2
# min_area_divisor = 75
# max_area_divisor = 10
# min_area_divisor_p = 50
# max_area_divisor_p = 10
# angle_thresh = 45
# i=0
# g=0
# while(1):
#     ret,frame = video.read()
#     if ret:
#         rows, columns = frame.shape[:2]
#         area = rows*columns
#         area_range = [area/min_area_divisor,area/max_area_divisor]
#         input_image = np.copy(frame)

#         fgmask = subtractor.apply(frame)

#         iterations = 2
#         res = cv2.GaussianBlur(fgmask,(3,3),0)
#         for i in range(iterations):
#             res = cv2.morphologyEx(res,cv2.MORPH_OPEN,kernel)
#             res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
#         res = cv2.erode(res,kernel,iterations = 3)
#         res = cv2.dilate(res,kernel,iterations = 5)
#         image, contours, hierarchy = cv2.findContours(res,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#         for i in range(len(contours)):
#             epsilon = 0.1*cv2.arcLength(contours[i],True)
#             approximated = cv2.approxPolyDP(contours[i],epsilon,True)
#             x,y,w,h = cv2.boundingRect(approximated)
#             aspect_ratio = float(w)/h if w > h else float(h)/w
#             area = cv2.contourArea(contours[i])
#             if ((area > area_range[0]) and (area < area_range[1])) and aspect_ratio<1.7 :
#                 #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
#                 required = frame[y:y+h,x:x+w]
#                 gray = cv2.cvtColor(required, cv2.COLOR_BGR2GRAY)
#                 gray = cv2.GaussianBlur(gray, (3,3), 0)
#                 gray = cv2.Sobel(gray, -1, 1, 0)
#                 h,sobel = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#                 se = cv2.getStructuringElement(cv2.MORPH_RECT, (11,3))
#                 gray = cv2.morphologyEx(sobel, cv2.MORPH_CLOSE, se)
#                 ed_img = np.copy(gray)
            
#                 cnt_image,cnts,_=cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
#                 aspect_ratio_range=[2.4,9]
#                 area_range = [200,5000]
#                 for contour in cnts :
#                     rect = cv2.minAreaRect(contour)
#                     img_width = gray.shape[1]
#                     img_height = gray.shape[0]
#                     area = img_width*img_height
                    
#                     box = cv2.boxPoints(rect) 
#                     box = np.int0(box)

#                     X = rect[0][0]
#                     Y = rect[0][1]
#                     angle = rect[2] 
#                     width = rect[1][0]
#                     height = rect[1][1]

#                     angle = (angle + 180) if width < height else (angle + 90)

#                     output=False

#                     if (width > 0 and height > 0) and ((width < img_width/2.0) and (height < img_width/2.0)):
#                         aspect_ratio = float(width)/height if width > height else float(height)/width
#                         if (aspect_ratio >= aspect_ratio_range[0] and aspect_ratio <= aspect_ratio_range[1]):
#                             if((height*width > area_range[0]) and (height*width < area_range[1])):

#                                 box_copy = list(box)
#                                 point = box_copy[0]
#                                 del(box_copy[0])
#                                 dists = [((p[0]-point[0])**2 + (p[1]-point[1])**2) for p in box_copy]
#                                 sorted_dists = sorted(dists)
#                                 opposite_point = box_copy[dists.index(sorted_dists[1])]
#                                 tmp_angle = 90

#                                 if abs(point[0]-opposite_point[0]) > 0:
#                                     tmp_angle = abs(float(point[1]-opposite_point[1]))/abs(point[0]-opposite_point[0])
#                                     tmp_angle = rad_to_deg(math.atan(tmp_angle))

#                                 if tmp_angle <= angle_thresh:
#                                     output = True
#                     if output == True:
#                         Xs = [i[0] for i in box]
#                         Ys = [i[1] for i in box]
#                         x1 = min(Xs)
#                         x2 = max(Xs)
#                         y1 = min(Ys)
#                         y2 = max(Ys)
                        
#                         angle = rect[2]
#                         if angle < -45:
#                             angle += 90 

#                         W = rect[1][0]
#                         H = rect[1][1]
#                         aspect_ratio = float(W)/H if W > H else float(H)/W

#                         center = ((x1+x2)/2,(y1+y2)/2)
#                         size = (x2-x1, y2-y1)
#                         M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0);
#                         tmp = cv2.getRectSubPix(ed_img, size, center)
#                         tmp = cv2.warpAffine(tmp, M, size)
#                         TmpW = H if H > W else W
#                         TmpH = H if H < W else W
#                         tmp = cv2.getRectSubPix(tmp, (int(TmpW),int(TmpH)), (size[0]/2, size[1]/2))
#                         __,tmp = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#                         white_pixels = 0

#                         for x in range(tmp.shape[0]):
#                             for y in range(tmp.shape[1]):
#                                 if tmp[x][y] == 255:
#                                     white_pixels += 1

#                         edge_density = float(white_pixels)/(tmp.shape[0]*tmp.shape[1])

#                         tmp = cv2.getRectSubPix(required, size, center)
#                         tmp = cv2.warpAffine(tmp, M, size)
#                         TmpW = H if H > W else W
#                         TmpH = H if H < W else W
#                         tmp = cv2.getRectSubPix(tmp, (int(TmpW),int(TmpH)), (size[0]/2, size[1]/2))

#                         if edge_density > 0.5:
# #                             for point in box:
# #                                 point[0]+=x
# #                                 point[1]+=y
#                             cv2.drawContours(input_image, [box], 0, (127,0,255),2)
#     #     cnts = sorted(contours, key=cv2.contourArea, reverse=True)
#     #     if len(cnts)<=to_identify :
#     #         chosen_contours = cnts
#     #     else : chosen_contours = cnts[:to_identify]

#     #     for i in range(len(chosen_contours)):
#     #         epsilon = 0.1*cv2.arcLength(chosen_contours[i],True)
#     #         approximated = cv2.approxPolyDP(chosen_contours[i],epsilon,True)
#     #         x,y,w,h = cv2.boundingRect(approximated)
#     #         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

#     #     fg = cv2.bitwise_and(frame,frame,mask=fgmask)

#         cv2.imshow('image',tmp)
#         cv2.imshow('frame',input_image)

#         if cv2.waitKey(1) & 0xff == ord('q'):
#             break
#     else : break
# video.release()

