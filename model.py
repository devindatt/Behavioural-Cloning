import os
import csv
import cv2
import numpy as np

#path = '/home/workspace/CarND-Behavioral-Cloning-P3/data/driving_log.csv'
#path = '/opt/carnd_p3/data/driving_log.csv'
#path = '/home/workspace/CarND-Behavioral-Cloning-P3/test_data/driving_log.csv'
#path = '/home/workspace/data_sample2/driving_log_test2.csv'
#path = '/home/workspace/data_sample3/driving_log_test3.csv'
path = '/home/workspace/data/driving_log.csv'

#Read in each line of the CSV file to array
lines = []
with open(path) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
 
print('Rows in Excel found :',len(lines))

#For every line in CSV pull out center image and steering angle and save to array
images = []
measurements = []

for line in lines[1:4]:
    for i in range(3):
        excel_path = line[i]
        tokens = excel_path.split('/')
        print(tokens)
#        local_path = '/home/workspace/CarND-Behavioral-Cloning-P3/data/IMG/'+token[-1]
#        local_path = '/opt/carnd_p3/data/IMG/'+tokens[-1]
        local_path = '/home/workspace/data/IMG/'+tokens[-1]
#        local_path = '/home/workspace/data_sample3/IMG/'+tokens[-1]
        print(local_path)    
        image = cv2.imread(local_path)
        images.append(image)
           
    correction = 0.2
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement+correction)
    measurements.append(measurement-correction)
    
print('Total images found :',len(images))    
augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_image = cv2.flip(image, 1)
    flipped_measurement = float(measurement) * -1.0
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)

print('Total images found after flipping:',len(augmented_images))        
print('Total measurements found after flipping:',len(augmented_measurements))        

#Create training set    
#X_train = np.array(images)
X_train = np.array(augmented_images)
#y_train = np.array(measurements)
y_train = np.array(augmented_measurements)    

print('Number of images after flipping :', len(X_train))
print('Number of measurements :', len(y_train))
print('X_train array shape :',X_train.shape)

exit()

#Create the Keras model
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Conv2D(6, kernel_size=(5,5), input_shape=(160,320,3)))
model.add(MaxPooling2D((2,2)))
model.add(Activation('relu'))                 

model.add(Conv2D(16, kernel_size=(5,5)))
model.add(MaxPooling2D((2,2)))
model.add(Activation('relu'))                 

model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs =5)

model.save('model2.h5')
exit()