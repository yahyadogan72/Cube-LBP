import numpy as np
import matplotlib.pyplot as plt
import os,cv2
import math
import shutil
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score


ROOT_DIR = 'D:/code/LbpKup/face/105_classes_pins_dataset/'
DEST_DIR = './Data'
n = 105  #int(input("Enter Number of Targets: "))
img_size=96

images = []
labels = []

  # Klasördeki her dosya için
for i,filename in enumerate(os.listdir(ROOT_DIR)):
    for resimler in os.listdir(os.path.join(ROOT_DIR,filename)):
        if resimler.endswith('.jpg') or resimler.endswith('.png'):
            image = cv2.imread(os.path.join(ROOT_DIR, filename,resimler))
            image = cv2.resize(image, (img_size,img_size))
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
            labels.append(tf.keras.utils.to_categorical(i, 105))

images=np.array(images)
labels=np.array(labels)
train_data, val_data, train_label, val_label = train_test_split(images, labels, test_size=0.1, random_state=42, shuffle=True)
train_data, test_data, train_label, test_label = train_test_split(train_data, train_label, test_size=0.1, random_state=42, shuffle=True)



def modelgetir(inputsize):
    input_shape = (img_size, img_size, inputsize)
    tf.keras.backend.clear_session()
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=3,activation="relu",input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(64, kernel_size=3,activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(128, kernel_size=3,activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(n, activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

modelcustom=[]


@tf.function
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
        s=tf.stack([lbp1,lbp2,lbp7], axis=-1)
        return tf.cast(s/255, dtype=tf.float32)

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
    
    s=tf.stack([lbp1,lbp2, lbp3, lbp4, lbp5, lbp6], axis=-1)
    #d= (s)*tf.math.reduce_max(s)/tf.math.reduce_std(s)
    
    return tf.cast(s/255, dtype=tf.float32)


@tf.function
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

    lbp_values = calculate_lbp(nnR, nnG, nnB,durum)
    
    if durum==0:
        lbp_matrix = tf.cast(tf.reshape(lbp_values, [r.shape[0], r.shape[1], 3]), dtype=tf.float32)
    else:
        lbp_matrix = tf.cast(tf.reshape(lbp_values, [r.shape[0], r.shape[1], 6]), dtype=tf.float32)
        
    img_tensor = tf.cast(img_tensor, tf.float32)  # Veri türünü float32'ye dönüştür
    tum = tf.concat([img_tensor / 255.0, lbp_matrix], axis=-1)
    
    # tum = tf.concat([tf.cast(img_tensor/255.0, dtype=tf.float32), lbp_matrix], axis=-1)
    
    tum=(1-tum)*tf.math.reduce_max(tum)-tf.math.reduce_std(tum)
    tek=(1-lbp_matrix)*tf.math.reduce_max(lbp_matrix)-tf.math.reduce_std(lbp_matrix)
    
    
    return tek,tum

powers_of_2 = tf.constant([[[[64, 128, 1,32, 0, 2,16, 8, 4]]]], dtype=tf.uint8)

x_traintek=[]
x_traintum=[]

x_valtek=[]
x_valtum=[]

x_testtek=[]
x_testtum=[]
for i in range(0,4):   
    if i==0:
        for image in train_data:
           tek,tum=lbp_from_tensor(image,durum=0)     
           x_traintek.append(tek)
           x_traintum.append(tum)      
            
        for image in val_data:
            tek,tum=lbp_from_tensor(image,durum=0)
            x_valtek.append(tek)
            x_valtum.append(tum)
            
        for image in test_data:
            tek,tum=lbp_from_tensor(image,durum=0)
            x_testtek.append(tek)
            x_testtum.append(tum)
    
    if i==2:
        x_traintek=[]
        x_traintum=[]
        
        x_valtek=[]
        x_valtum=[]
        
        x_testtek=[]
        x_testtum=[]
        
        for image in train_data:   
            tek,tum=lbp_from_tensor(image,durum=1)
            x_traintek.append(tek)
            x_traintum.append(tum)      
            
        for image in val_data: 
            tek,tum=lbp_from_tensor(image,durum=1)
            x_valtek.append(tek)
            x_valtum.append(tum)
            
        for image in test_data:
            tek,tum=lbp_from_tensor(image,durum=1)
            x_testtek.append(tek)
            x_testtum.append(tum)
        
    
    if i==0:            
        x_train_processed=np.array(x_traintek)
        x_val_processed=np.array(x_valtek)    
        x_test_processed=np.array(x_testtek)
        model=modelgetir(3)
        modelnum='Lbp1003'
    elif i==1:           
        x_train_processed=np.array(x_traintum)
        x_val_processed=np.array(x_valtum) 
        x_test_processed=np.array(x_testtum)    
        model=modelgetir(6)
        modelnum='Lbp1006'
    elif i==2:           
        x_train_processed=np.array(x_traintek)
        x_val_processed=np.array(x_valtek)    
        x_test_processed=np.array(x_testtek)  
        model=modelgetir(6)
        modelnum='LbpKup1006'
    else:
        x_train_processed=np.array(x_traintum)
        x_val_processed=np.array(x_valtum) 
        x_test_processed=np.array(x_testtum) 
        model=modelgetir(9)
        modelnum='LbpKup1009'
    
    
    weight_path = "CifarYuz{}.hdf5".format(modelnum)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(weight_path, monitor='val_loss', verbose=0,  save_best_only=True, mode='auto', save_weights_only=False)
    reduceLROnPlat = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1, mode='min')
    callbacks_list = [reduceLROnPlat,checkpoint]
        
    modelcustom.append(model.fit(x_train_processed, train_label, epochs=30, validation_data=(x_val_processed, val_label)))
    
    y_pred=model.predict(x_test_processed)
    pred=np.argmax(y_pred,axis=1)
    ground = np.argmax(test_label,axis=1)

    accuracy = accuracy_score(ground, pred)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(ground, pred,average='weighted')
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(ground, pred,average='weighted')
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(ground, pred,average='weighted')
    print('F1 score: %f' % f1)
    print('\n')

model_IncRes = modelgetir(3)

model_IncRes.compile(optimizer='adam',loss=tf.keras.losses.categorical_crossentropy,metrics=['accuracy'])
his_IncRes = model_IncRes.fit(x=train_data/255,y=train_label, epochs= 30, validation_data= (val_data/255,val_label))

modelcustom.append(his_IncRes)


y_pred=model_IncRes.predict(test_data)
pred=np.argmax(y_pred,axis=1)
ground = np.argmax(test_label,axis=1)

accuracy = accuracy_score(ground, pred)
print('Accuracy: %f' % accuracy)
precision = precision_score(ground, pred,average='weighted')
print('Precision: %f' % precision)
recall = recall_score(ground, pred,average='weighted')
print('Recall: %f' % recall)
f1 = f1_score(ground, pred,average='weighted')
print('F1 score: %f' % f1)
print('\n')

def plot_trainingvalidasyon(history):
    # epo = range(0, epochs)
    
    plt.figure(figsize=(10, 7))
    plt.plot(history[0].history['val_loss'],'black', label='Orj')
    plt.plot(history[1].history['val_loss'], 'blue', label='Lbp3')
    plt.plot(history[2].history['val_loss'], 'brown', label='Lbp6')
    plt.plot(history[3].history['val_loss'], 'red', label='LbpKup6')
    plt.plot(history[4].history['val_loss'], 'green', label='LbpKup9')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Val Loss')
    plt.legend()
    # plt.savefig('./outputs100/val_loss{}.png'.format(model_num))
    plt.show() 
    
    
    plt.figure(figsize=(10, 7))
    plt.plot(history[0].history['val_accuracy'],'black', label='Orj')
    plt.plot(history[1].history['val_accuracy'], 'blue', label='Lbp3')
    plt.plot(history[2].history['val_accuracy'], 'brown', label='Lbp6')
    plt.plot(history[3].history['val_accuracy'], 'red', label='LbpKup6')
    plt.plot(history[4].history['val_accuracy'], 'green', label='LbpKup9')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Val Accuracy')
    plt.legend()
    # plt.savefig('./outputs100/val_accuracyl{}.png'.format(model_num))
    plt.show() 
    
    
plot_trainingvalidasyon(modelcustom)