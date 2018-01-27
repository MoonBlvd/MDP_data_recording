from __future__ import division
import cv2
import os
from PIL import Image
import time
import numpy as np

class simpleCompress():
    def __init__(self):
        self.a = 1
    def run_pillow(self,img,quality,ext):
        image = Image.fromarray(img)
        image.save('000000'+ext, ptimize=True, quality=quality)
        new_size = os.stat(os.path.join(os.getcwd(), '000000'+ext)).st_size/1024
        os.remove(os.path.join(os.getcwd(), '000000'+ext))
        return new_size
    def run_opencv(self,img,quality,ext):
        cv2.imwrite('000000'+ext, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        new_size = os.stat(os.path.join(os.getcwd(), '000000'+ext)).st_size/1024
        os.remove(os.path.join(os.getcwd(), '000000'+ext))
        return new_size
    def run_opencv_encoder(self,img,quality,ext):
        _,compressed = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return 100*compressed.shape[0]*compressed.shape[1]/(img.shape[0]*img.shape[1])
    # def run_svd(self):
    #
    # def compute_compression_ratio(self):

if __name__ == '__main__':
    compressor = simpleCompress()
    img = cv2.imread('test_image.png')
    quality_list = [10,20,30,40,50,60,70,80,90,100]
    level_list = [0,1,2,3,4,5,6,7,8,9]
    binary_list = [0,1]
    ext = '.jpeg'
    start_time = time.time()
    print ('Compression is: '+ext)

    if ext == '.jpeg' or '.webp':
        param_list = quality_list
    elif ext == '.png':
        param_list = level_list
    elif ext == '.ppm' or '.PGM'or 'PBM':
        param_list = binary_list
    else:
        raise NameError('Extension name not recognized')

    for quality in param_list:
        new_size = compressor.run_pillow(img, quality,ext)
        elapsed_time = time.time() - start_time
        print ("PIL: " + str(format(new_size,'.4f'))+ "KB; Time: "+ str(format(elapsed_time,'.4f')) + 'second')

        # start_time = time.time()
        # new_size = compressor.run_opencv(img, quality)
        # elapsed_time = time.time() - start_time
        # print ("OpenCV: ", new_size, "KB; Time: ",elapsed_time, 'second')

        start_time = time.time()
        new_size = compressor.run_opencv_encoder(img, quality,ext)
        elapsed_time = time.time() - start_time
        print ("OpenCV: " + str(format(new_size,'.4f')) + "%; Time: " + str(format(elapsed_time,'.4f')) + 'second')
