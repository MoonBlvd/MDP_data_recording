from __future__ import division
import cv2
import os
from PIL import Image
import time
import numpy as np

class simpleCompress():
    def __init__(self):
        self.a = 1
    def run_pillow(self,img,quality):
        image = Image.fromarray(img)
        image.save('000000.jpeg',ptimize=True,quality=quality)
        new_size = os.stat(os.path.join(os.getcwd(), '000000.jpeg')).st_size/1024
        os.remove(os.path.join(os.getcwd(), '000000.jpeg'))
        return new_size
    def run_opencv(self,img,quality):
        cv2.imwrite('000000.jpeg',img, [cv2.IMWRITE_JPEG_QUALITY,quality])
        new_size = os.stat(os.path.join(os.getcwd(), '000000.jpeg')).st_size/1024
        os.remove(os.path.join(os.getcwd(), '000000.jpeg'))
        return new_size
    def run_opencv_encoder(self,img,quality):
        _,compressed = cv2.imencode('.jpeg',img,[cv2.IMWRITE_JPEG_QUALITY,quality])
        return 100*compressed.shape[0]*compressed.shape[1]/(img.shape[0]*img.shape[1])
    # def run_svd(self):
    #
    # def compute_compression_ratio(self):

if __name__ == '__main__':
    compressor = simpleCompress()
    img = cv2.imread('test_image.png')
    quality = 85
    level = 9
    start_time = time.time()
    new_size = compressor.run_pillow(img, quality)
    elapsed_time = time.time() - start_time
    print ("PIL new size: ", new_size, "KB; Time: ",elapsed_time, 'second')

    start_time = time.time()
    new_size = compressor.run_opencv(img, quality)
    elapsed_time = time.time() - start_time
    print ("OpenCV new size: ", new_size, "KB; Time: ",elapsed_time, 'second')

    start_time = time.time()
    new_size = compressor.run_opencv_encoder(img, quality)
    elapsed_time = time.time() - start_time
    print ("OpenCV encoder compression ratio: ", new_size, "%; Time: ",elapsed_time, 'second')