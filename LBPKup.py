# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 13:06:22 2024

@author: cuney
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

powers_of_2 = tf.constant([[[[128, 64, 32,16, 0, 8,4, 2, 1]]]], dtype=tf.uint8)

# @tf.function
def calculate_lbp(patchR, patchG, patchB,durum=0):
    centerR = patchR[:, :, :,4:5]
    centerB = patchB[:, :, :,4:5]
    
    BVR = tf.cast(patchR >= centerR, tf.uint8)
    BVB = tf.cast(patchB >= centerB, tf.uint8)

    lbp1 = tf.reduce_sum(powers_of_2 * BVR, axis=-1)
    lbp2 = tf.reduce_sum(powers_of_2 * BVB, axis=-1)
    
    if durum==0:
       centerG = patchG[:, :, :,4:5]
       BVG = tf.cast(patchG >= centerG, tf.uint8)
       lbp7 = tf.reduce_sum(powers_of_2 * BVG, axis=-1)
       s=tf.cast(tf.stack([lbp1,lbp2,lbp7], axis=-1), dtype=tf.float32)/255.0
        
       d= (1-s)*tf.math.reduce_max(s)/(1 - s)*tf.math.reduce_std(s)
        
       return d

    LBP3 = tf.concat([patchR[:, :,:, :3], patchG[:, :,:, :3], patchB[:, :,:, :3]], axis=-1)
    centerL3 = LBP3[:, :, :,4:5]
    BVB3 = tf.cast(LBP3 >= centerL3, tf.uint8)
    lbp3 = tf.reduce_sum(powers_of_2 * BVB3, axis=-1)

    LBP4 = tf.concat([patchR[:,:, :, 6:9], patchG[:,:, :, 6:9], patchB[:,:, :, 6:9]], axis=-1)
    centerL4 = LBP4[:, :, :,4:5]
    BVB4 = tf.cast(LBP4 >= centerL4, tf.uint8)
    lbp4 = tf.reduce_sum(powers_of_2 * BVB4, axis=-1)

    LBP5 = tf.concat([tf.gather(patchR, [2, 5, 8], axis=-1), tf.gather(patchG, [2, 5, 8], axis=-1), tf.gather(patchB, [2, 5, 8], axis=-1)], axis=-1)
    centerL5 = LBP5[:, :, :,4:5]
    BVB5 = tf.cast(LBP5 >= centerL5, tf.uint8)
    lbp5 = tf.reduce_sum(powers_of_2 * BVB5, axis=-1)

    LBP6 = tf.concat([tf.gather(patchR, [0, 3, 6], axis=-1), tf.gather(patchG, [0, 3, 6], axis=-1), tf.gather(patchB, [0, 3, 6], axis=-1)], axis=-1)
    centerL6 = LBP6[:, :, :,4:5]
    BVB6 = tf.cast(LBP6 >= centerL6, tf.uint8)
    lbp6 = tf.reduce_sum(powers_of_2 * BVB6, axis=-1)
    
    s=tf.cast(tf.stack([lbp1,lbp2, lbp3, lbp4, lbp5, lbp6], axis=-1), dtype=tf.float32)/255.0

    d= (1-s)*tf.math.reduce_max(s)/(1 - s)*tf.math.reduce_std(s)
    
    return d


# @tf.function
def lbp_from_tensor(img_tensor,durum=0):
    r = img_tensor[:, :, 0]
    g = img_tensor[:, :, 1]
    b = img_tensor[:, :, 2]
    
    r_expanded = tf.expand_dims(tf.expand_dims(r, axis=-1), axis=0)
    g_expanded = tf.expand_dims(tf.expand_dims(g, axis=-1), axis=0)
    b_expanded = tf.expand_dims(tf.expand_dims(b, axis=-1), axis=0)

    nnR = tf.image.extract_patches(images=r_expanded, sizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
    nnG = tf.image.extract_patches(images=g_expanded, sizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
    nnB = tf.image.extract_patches(images=b_expanded, sizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')

    lbp_values = calculate_lbp(nnR, nnG, nnB,durum)/255.0
    
    if durum==0:
        lbp_matrix = tf.cast(tf.reshape(lbp_values, [r.shape[0], r.shape[1], 3]), dtype=tf.float32)
    else:
        lbp_matrix = tf.cast(tf.reshape(lbp_values, [r.shape[0], r.shape[1], 6]), dtype=tf.float32)
    lbp_img = tf.concat([img_tensor/255.0, lbp_matrix], axis=-1)
    
    return lbp_img

img_path = r"C:\Users\cuney\Downloads\kedi.jpeg"

# Görüntüyü oku
img = tf.io.read_file(img_path)
img = tf.io.decode_jpeg(img, channels=3)
img = tf.keras.layers.Resizing(256, 256)(img)

result = lbp_from_tensor(img,durum=1)  # Tensor'ı numpy dizisine çeviriyoruz

# plt.imshow(img/255)
# # plt.imsave("orjinal", img/255, cmap='gray')
# plt.show()

for i in range(9):
    plt.imshow(result[:, :, i], cmap='gray')
    filename = f'lbp_{i+1}.png'  # Örneğin: lbp_1.png, lbp_2.png, ...
    plt.imsave(filename, result[:, :, i], cmap='gray')
    plt.show()

