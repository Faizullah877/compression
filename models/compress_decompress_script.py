from genericpath import isfile
import os
from posixpath import basename
# from math import log10, sqrt
# import cv2 as cv
import numpy as np
import time
# import sewar 

import tensorflow as tf



def read_png(filename):
    """Loads a PNG image file."""
    string = tf.io.read_file(filename)
    return tf.image.decode_image(string, channels=1)



# Holograms_Path = '/home/user/Documents/Tensorflow_Compression/compression'

Holograms_Path = '/media/image_lab/Holograms/Test_Data'

modelpath = ''
# models = ['bmshj18mse_0p0008', 'bmshj18mse_0p002','bmshj18mse_0p005', 'bmshj18mse_0p008','bmshj18mse_0p01', 'bmshj18mse_0p02','bmshj18mse_0p04', 'bmshj18mse_0p08', 'bmshj18mse_0p1']
models = ['bmshj18mse_mixN_0p002']
# a = {'Horse': 'horse_Hol_v1'}
# a = {'Astronaut': 'Astronaut_Hol_v2'}
# a = {'deepDices2k': 'deepDices2k'}
a = {'Hol_3D_multi': 'Hol_3D_multi'}
# a = {'deepDices2k': 'deepDices2k'}
# a = {'deepDices8k4k-RI': 'deepDices8k4k'}
color = False
# txt_file = os.path.join(Holograms_Path, 'Horse', 'bpp_file.txt')

# bpp_file = open(txt_file, 'a')
# q_file = open('quality_metrics_file.txt')

def create_directories(model, Holograms_Path, a):
    for folder, basename in a.items():
        model_dir = os.path.join(Holograms_Path, folder, model)
        encoded_dir = os.path.join(model_dir, 'encoded')
        decoded_dir = os.path.join(model_dir, 'decoded')
        originals_dir = os.path.join(Holograms_Path, folder, 'Original_PNGs')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
            print("Directory" , model_dir, " created")
        else:
            print("Directory already exists")
        if not os.path.exists(encoded_dir):
            os.mkdir(encoded_dir)
            print("Directory" , encoded_dir, " created")
        else:
            print("Directory already exists")
        if not os.path.exists(decoded_dir):
            os.mkdir(decoded_dir)
            print("Directory" , decoded_dir, " created")
        else:
            print("Directory already exists")
        
    return originals_dir, encoded_dir, decoded_dir
 
def create_files_name_list(folders_Dic):
    all_files_list= list()
    if color == False:
        for folder, baseName in folders_Dic.items():
            # all_files_list.append(baseName+'_c1_real8B.png')
            all_files_list.append(baseName+'_c1_imag8B.png')
    else:
        for folder, baseName in folders_Dic.items():
            # all_files_list.append(baseName+'_c1_real8B.png')
            # all_files_list.append(baseName+'_c1_imag8B.png')
            all_files_list.append(baseName+'_c2_real8B.png')
            all_files_list.append(baseName+'_c2_imag8B.png')
            all_files_list.append(baseName+'_c3_real8B.png')
            all_files_list.append(baseName+'_c3_imag8B.png')        
    return all_files_list
                  

for model in models:

    originals_folder, encoded_folder, decoded_foder = create_directories(model, Holograms_Path, a)

    allFiles = create_files_name_list(a)
    # bpp = 0
    for image in allFiles:
        input_image = os.path.join(originals_folder ,image)
        cmprsFile = image[0:-4]+'_'+ model + '.tfci'
        compressed_file = os.path.join(encoded_folder, cmprsFile)
        decoded_img = cmprsFile[0:-5]+'_rec.png'
        output_image = os.path.join(decoded_foder, decoded_img)
        compress_command = '/bin/python3 models/bmshj18mse_lambda.py --model_path ' + modelpath+model +  ' compress ' + input_image + ' ' + compressed_file

        decompress_command = '/bin/python3 models/bmshj18mse_lambda.py --model_path ' + modelpath+model +  ' decompress ' + compressed_file + ' ' + output_image
        os.system(compress_command)
        # time.sleep(5)
        os.system(decompress_command)
        # time.sleep(5)
        # sizeA = os.path.getsize(compressed_file)
        # bpp = bpp+ sizeA;

        # # # q_file.write(q_line)
        # imge_ref = read_png(input_image)
        # # imge_ref = tf.io.decode_png(input_image, channels=1, dtype=tf.dtypes.uint8, name=None)

        # imge_rec = read_png(output_image)
        # height = imge_rec.shape[0]
        # width = imge_rec.shape[1]
        # # imge_rec = tf.io.decode_png(output_image, channels=1, dtype=tf.dtypes.uint8, name=None)
        # psnr1 = tf.image.psnr(imge_ref, imge_rec, max_val=255)
        # ssim1 = tf.image.ssim(imge_ref, imge_rec, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
        # # # print(psnr1)

        # # q_line = "PSNR of : " + decoded_img + " is : " + str(psnr1) +app.run(main, flags_parser=parse_args) " and SSIM is : " + str(ssim1) + "\n" 
        # q_line = "Deconded Image : " + decoded_img + "   bpp : " + str((sizeA*8)/(width*height)) + "   psnr : " + str(psnr1.numpy()) + "    ssim : " + str(ssim1.numpy()) + "\n"
        # # q_line = "Deconded Image : " + decoded_img + "   bpp : " + str((sizeA*8)/(width*height)) + "\n"
        # bpp_file.write(q_line);
        # # del imge_rec
        # # del imge_ref




    
#     line1 = "bpp of Hologram is : " + cmprsFile + " is  : " + str((bpp*8)/(width*height)) + "\n"
#     bpp_file.write(line1);

# bpp_file.close()
# q_file.close()

        #  time.sleep(5)
        #log_file.write(line)
        
        #  imge1 = tf.image.decode_png(tf.io.read_file(input_image))
        #  imge2 = tf.image.decode_png(tf.io.read_file(output_image))
        #  psnr1 = tf.image.psnr(imge1, imge2, max_val=255)
        #  line2 = "PSNR via TF of " + image + " and " + decoded_img + " is  " + str(psnr1) + "\n"
        #  log_file.write(line2)
        #  image1 = tf.expand_dims(imge1, axis=0)
        #  image2 = tf.expand_dims(imge2, axis=0)
        #  ssim1 = tf.image.ssim(image1, image2, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
        
        #  line3 = "ssim via TF of " + image + " and " + decoded_img + " is  " + str(ssim1) + "\n"
        #  log_file.write(line3)
        #  log_file.write("************************\n")


# log_file.close()