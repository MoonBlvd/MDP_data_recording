from __future__ import division
import cv2
import os
from PIL import Image
import time
import numpy as np
import sys

class simpleCompress():
    def __init__(self):
        self.a = 1
    def run_pillow(self,img,quality,ext,i=0):
        img_name = str(format(i,'04d'))+ext
        image = Image.fromarray(img)
        image.save(img_name, ptimize=True, quality=quality)
        new_size = os.stat(os.path.join(os.getcwd(), img_name)).st_size/1024
        os.remove(os.path.join(os.getcwd(), '000000'+ext))
        return new_size
    def run_opencv(self,img,quality,ext,qual_param,i=0):
        img_name = str(format(i,'04d'))+ext
        cv2.imwrite(img_name, img, [qual_param, quality])
        new_size = os.stat(os.path.join(os.getcwd(), img_name)).st_size/1024
        #os.remove(os.path.join(os.getcwd(), '000000'+ext))
        return new_size
    def run_opencv_encoder(self,img,quality,ext,qual_param):
        _,compressed = cv2.imencode(ext, img, [qual_param, quality])
        return 100*compressed.shape[0]*compressed.shape[1]/(img.shape[0]*img.shape[1])
    # def run_svd(self):
    #
    # def compute_compression_ratio(self):

if __name__ == '__main__':
    compressor = simpleCompress()
    img = cv2.imread('test_image.png')
    quality_list = [10,20,30,40,50,60,70,80,90,100]
    level_list = [9,8,7,6,5,4,3,2,1,0]
    binary_list = [0,1]
    ext = sys.argv[1]

    start_time = time.time()
    print ('Compression is: '+ext)

    if ext == '.jpeg':
        param_list = quality_list
        qual_param = cv2.IMWRITE_JPEG_QUALITY
    elif ext == '.webp':
        param_list = quality_list
        qual_param = cv2.IMWRITE_WEBP_QUALITY
    elif ext == '.png':
        param_list = level_list
        qual_param = cv2.IMWRITE_PNG_COMPRESSION
    elif ext == '.ppm' or '.pgm'or '.pbm':
        param_list = binary_list
        qual_param = cv2.IMWRITE_PXM_BINARY
    else:
        raise NameError('Extension name not recognized')

    for i,quality in enumerate(param_list):
        # start_time = time.time()
        # new_size = compressor.run_opencv(img, quality,ext,qual_param)
        # elapsed_time = time.time() - start_time
        # print ("OpenCV: ", new_size, "KB; Time: ",elapsed_time, 'second')
        
        # start_time = time.time()
        # new_size = compressor.run_opencv(img, quality,ext, qual_param,i)
        # elapsed_time = time.time() - start_time
        # print ("OpenCV: ", new_size, "KB; Time: ",elapsed_time, 'second')

        start_time = time.time()
        new_size = compressor.run_opencv_encoder(img, quality,ext, qual_param)
        elapsed_time = time.time() - start_time
        print ("OpenCV: " + str(format(new_size,'.4f')) + "\%; Time: " + str(format(elapsed_time,'.4f')) + 'second')
