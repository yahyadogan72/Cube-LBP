import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix, classification_report
import os 
import seaborn as sns

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


mypath = "./outputs10"
if not os.path.isdir(mypath):
   os.makedirs(mypath)
   
model_num = 'Orj'
log_file = './outputs10/log_{}.txt'.format(model_num)
powers_of_2 = tf.constant([[[[64, 128, 1,32, 0, 2,16, 8, 4]]]], dtype=tf.uint8)

def write_to_file(text, log_file):
    with open(log_file, 'a+') as t_file:
        t_file.write(text+'\n')    
        
def plot_confusion_matrix(modelnum, cm, classes, normalize=False, title='Confusion Matrix'):
    # Özel oluşturulmuş renk paleti
    custom_palette = ["#95a5a6", "#2ecc71","#3498db"]  # Mavi, Gri, Yeşil

    plt.figure(figsize=(10, 8))
    plt.title(title, fontsize=16, fontweight='bold')

    # Renk paletini seç
    selected_palette = custom_palette  # İstediğiniz bir paleti seçebilirsiniz

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm, annot=True, fmt='.2f', cmap=selected_palette, xticklabels=classes, yticklabels=classes,cbar=False)
        print('Normalized Confusion Matrix')
    else:
        sns.heatmap(cm, annot=True, fmt='g', cmap=selected_palette, xticklabels=classes, yticklabels=classes,cbar=False)
        print('Confusion Matrix, Without Normalization')

    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.savefig('./outputs10/{}.png'.format(modelnum), dpi=300)
    # plt.savefig('cm.png', dpi=300, format='png', bbox_inches='tight')
    plt.show()
    
def testsonuclari(model,history,xtest, model_num):
    prob = model.predict(xtest)
    predIdxs = np.argmax(prob, axis=1) 
    y_test1=np.argmax(y_test, axis=1)
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test1, predIdxs)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test1, predIdxs,average='weighted')
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test1, predIdxs,average='weighted')
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test1, predIdxs,average='weighted')
    print('F1 score: %f' % f1)
    print('\n')
    print(classification_report(y_test1, predIdxs))
    
    # Confusion matrix
    write_to_file('Model Numarası: %s ' %(model_num), log_file)
    write_to_file('Accuracy: %f ' %(accuracy), log_file)
    write_to_file('Precision: %f ' %(precision), log_file)
    write_to_file('Recall: %f ' %(recall), log_file)
    write_to_file('F1 score: %f ' %(f1), log_file)
    
    
    cm = confusion_matrix(y_test1, predIdxs, labels=classes)
    plot_confusion_matrix(model_num, cm= cm, classes= classes, title = 'Confusion Matrix')
    # plot_training(history, model_num)
    
def modelgetir(inputsize):
    input_shape = (32, 32, inputsize)
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
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])  
    return model

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
        
        #d= (s)*tf.math.reduce_max(s)/tf.math.reduce_std(s)
        
        return s

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
    #d= (s)*tf.math.reduce_max(s)/tf.math.reduce_std(s)
    
    return s


# @tf.function
def lbp_from_tensor(img_tensor,durum=0):
    r = img_tensor[:, :, 0]
    g = img_tensor[:, :, 1]
    b = img_tensor[:, :, 2]
    durum=1
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
        
    s = tf.concat([img_tensor/255.0, lbp_matrix], axis=-1)
    
    tum=(1-s)*tf.math.reduce_max(s)-tf.math.reduce_std(s)
    tek=(1-lbp_matrix)*tf.math.reduce_max(lbp_matrix)-tf.math.reduce_std(lbp_matrix)
    
    return tek,tum

batch_size = 32
epochs = 20
num_classes = 10
channel=3
img_rows, img_cols = 32, 32
classes = list(range(num_classes)) 
modelcustom=[]
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channel)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channel)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train1 = x_train/255.0
x_test1 = x_test/255.0
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

weight_path = "{}.hdf5".format('lbpKupOrj10')
checkpoint = tf.keras.callbacks.ModelCheckpoint(weight_path, monitor='val_loss', verbose=0,  save_best_only=True, mode='auto', save_weights_only=True)
reduceLROnPlat = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1, mode='min')
callbacks_list = [reduceLROnPlat,checkpoint]

x_traintek=[]
x_traintum=[]
x_testtek=[]
x_testtum=[]
for i in range(0,4):   
    if i==0:
        for image in x_train:
           tek,tum=lbp_from_tensor(image,durum=0)     
           x_traintek.append(tek)
           x_traintum.append(tum)      
            
        for image in x_test:
            tek,tum=lbp_from_tensor(image,durum=0)
            x_testtek.append(tek)
            x_testtum.append(tum)
    
    if i==2:
        x_traintek=[]
        x_traintum=[]
        x_testtek=[]
        x_testtum=[]
        for image in x_train:   
            tek,tum=lbp_from_tensor(image,durum=1)
            x_traintek.append(tek)
            x_traintum.append(tum)      
            
        for image in x_test: 
            tek,tum=lbp_from_tensor(image,durum=1)
            x_testtek.append(tek)
            x_testtum.append(tum)
        
    
    if i==0:            
        x_train_processed=np.array(x_traintek)
        x_test_processed=np.array(x_testtek)    
        model=modelgetir(3)
        modelnum='Lbp103'
    elif i==1:           
        x_train_processed=np.array(x_traintum)
        x_test_processed=np.array(x_testtum)    
        model=modelgetir(6)
        modelnum='Lbp106'
    elif i==2:           
        x_train_processed=np.array(x_traintek)
        x_test_processed=np.array(x_testtek)    
        model=modelgetir(6)
        modelnum='LbpKup106'
    else:
        x_train_processed=np.array(x_traintum)
        x_test_processed=np.array(x_testtum)  
        model=modelgetir(9)
        modelnum='LbpKup109'
        
    weight_path = "CifarOn{}.hdf5".format(modelnum)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(weight_path, monitor='val_loss', verbose=0,  save_best_only=True, mode='auto', save_weights_only=False)
    reduceLROnPlat = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1, mode='min')
    callbacks_list = [reduceLROnPlat,checkpoint]
    modelcustom.append(model.fit(x_train_processed,y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            shuffle=False,
                            validation_split=0.1,
                            callbacks=callbacks_list))
    
    testsonuclari(model,modelcustom,x_test_processed, modelnum)

model=modelgetir(3)
modeln = model.fit(x_train1,y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        shuffle=False,
                        validation_split=0.1,
                        callbacks=callbacks_list)

testsonuclari(model,modeln,x_test1, model_num='Orj')
modelcustom.append(modeln)

def plot_trainingvalidasyon(history):
    # epo = range(0, epochs)
    
    plt.figure(figsize=(10, 7))
    plt.plot(history[0].history['val_loss'],'green', label='Org')
    plt.plot(history[1].history['val_loss'], 'blue', label='Lbp3')
    plt.plot(history[2].history['val_loss'], 'brown', label='Lbp6')
    plt.plot(history[3].history['val_loss'], 'red', label='LbpKup6')
    plt.plot(history[4].history['val_loss'], 'black', label='LbpKup9')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Val Loss')
    plt.legend()
    plt.savefig('./outputs10/val_loss{}.png'.format(model_num))
    plt.show() 
    
    
    plt.figure(figsize=(10, 7))
    plt.plot(history[0].history['val_accuracy'],'green', label='Org')
    plt.plot(history[1].history['val_accuracy'], 'blue', label='Lbp3')
    plt.plot(history[2].history['val_accuracy'], 'brown', label='Lbp6')
    plt.plot(history[3].history['val_accuracy'], 'red', label='LbpKup6')
    plt.plot(history[4].history['val_accuracy'], 'black', label='LbpKup9')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Val Accuracy')
    plt.legend()
    plt.savefig('./outputs10/val_accuracyl{}.png'.format(model_num))
    plt.show() 
    
    
plot_trainingvalidasyon(modelcustom)