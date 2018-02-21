from __future__ import division
import cv2
import os
from PIL import Image
import time
import numpy as np
import sys
import csv

def write_csv(file_path, data):
    with open(file_path, 'w') as csvfile:
        # field = [field_name]
        writer = csv.writer(csvfile)
        for i in range(data.shape[0]):
            writer.writerow([data[i]])

class simpleCompress():
    def __init__(self, data_path):
        self.data_path = data_path
    def run_pillow(self,img,quality,ext,i=0):
        img_name = str(format(i,'04d'))+ext
        image = Image.fromarray(img)
        image.save(img_name, optimize=True, quality=quality)
        new_size = os.stat(os.path.join(os.getcwd(), img_name)).st_size/1024
        os.remove(os.path.join(os.getcwd(), '000000'+ext))
        return new_size
    def run_opencv(self,img,ext,qual_param,quality=100,i=0,j=0,a=0,persistent_record=False):
        #if a == 1: # strong compress
        #    quality = 20
        #elif a == 2: # weak compress
        #    quality = 80
        #elif a == 3: # no compress
        #    quality = 100
        #else:
        #    quality = 0
        img_name = self.data_path + str(format(i,'06d'))+'_'+ str(format(j,'06d')) +ext
        cv2.imwrite(img_name, img, [qual_param, 100])
        orig_size = os.stat(os.path.join(os.getcwd(), img_name)).st_size/1024 # in KB

        img_name = self.data_path + str(format(i,'06d'))+'_'+ str(format(j,'06d')) +ext
        cv2.imwrite(img_name, img, [qual_param, quality])
        new_size = os.stat(os.path.join(os.getcwd(), img_name)).st_size/1024 # in KB
        if persistent_record is False:
            os.remove(os.path.join(os.getcwd(), img_name))
        return new_size, new_size/orig_size

    def run_opencv_encoder(self,img,ext,qual_param,quality=100,a=0):
        if a == 1: # strong compress
            quality = 20
        elif a == 2: # weak compress
            quality = 80
        elif a == 3: # no compress
            quality = 100
        _,compressed = cv2.imencode(ext, img, [qual_param, quality])
        return 100*compressed.shape[0]*compressed.shape[1]/(img.shape[0]*img.shape[1])
    # def run_svd(self):
    #
    # def compute_compression_ratio(self):

if __name__ == '__main__':
    compressor = simpleCompress('recorded_img/')
    img = cv2.imread('test_image.png')

    video_path = '../Smart_Black_Box/data/videos/'
    video_name = '05182017_video1080p.mp4'
    cap = cv2.VideoCapture(video_path + video_name)

    quality_list = [i for i in range(101)]#[10,20,30,40,50,60,70,80,90,100]
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

    frame_ctr = 0
    frames_to_watch = np.random.permutation(50000)
    #while cap.isOpened():
    all_output = []
    F = open('compression_ratio_quality.csv','w')
    writer = csv.writer(F)
    for img_id in frames_to_watch[:100]:

        cap.set(1, img_id)
        ret,img = cap.read()

        frame_ctr += 1
        all_quality = []
        all_compress_ratio = []
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
        # new_size = compressor.run_opencv_encoder(img, ext, qual_param,quality=quality)
            new_size,compress_ratio = compressor.run_opencv(img,ext,qual_param,quality=quality,i=0,j=0,a=0,persistent_record=False)
            elapsed_time = time.time() - start_time
            print ("OpenCV: " + str(format(new_size,'.4f')) + "\%; Time: " + str(format(elapsed_time,'.4f')) + 'second')
            print ("Ratio: " + str(compress_ratio) + "  Quality: " + str(quality))
            #all_quality.append(quality)
            #all_compress_ratio.append(compress_ratio)
            writer.writerow([compress_ratio,quality])
