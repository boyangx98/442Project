import numpy as np
import cv2

img = cv2.imread('dataset/val/paris_eval_gt/001_im.png')
mask = cv2.imread('mask.png',0)
img = cv2.resize(img, mask.shape)
dst = cv2.inpaint(img,mask,4,cv2.INPAINT_TELEA)

cv2.imwrite('out.png',dst)