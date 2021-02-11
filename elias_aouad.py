import os
from tqdm import trange
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
    
def pedestrians(path, w, h, n):
    hog = cv2.HOGDescriptor() 
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgbg = cv2.createBackgroundSubtractorMOG2() 
    
    bboxes = []
    filenames = sorted(os.listdir(path))
    for frame_id in trange(1, len(filenames)+1):
        frame_path = os.path.join(path, filenames[frame_id-1])
        frame = cv2.imread(frame_path)
        gray = cv2.imread(frame_path, 0)
        fgmask = fgbg.apply(gray)
        closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        dilation = cv2.dilate(opening, kernel)
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rects = []
        for i in range(len(contours)):
            if hierarchy[0, i, 3] == -1:
                area = cv2.contourArea(contours[i])
                if area > 100:
                    x0, y0, w0, h0 = cv2.boundingRect(contours[i])
                    add_dim = 115
                    portion = frame[max(y0-add_dim,0):y0+h0+add_dim, max(x0-add_dim,0):x0+w0+add_dim]
                    regions, _ = hog.detectMultiScale(portion)
                    for (x_, y_, w, h) in regions:
                        x = max(x0-add_dim,0) + x_
                        y = max(y0-add_dim,0) + y_
                        rects.append([x, y, x+w, y+h])
        rects = np.array(rects)
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.8)
        for bb_id, (xA, yA, xB, yB) in enumerate(pick):
            bboxes.append([frame_id, bb_id+1, xA, yA, xB-xA, yB-yA])
    return bboxes

