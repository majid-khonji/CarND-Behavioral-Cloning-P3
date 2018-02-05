import csv
import cv2
import numpy as np
import sklearn


# reads csv data and changes image names to match the same directory of the csv file
def process_csv_data(csv_file_name='data/driving_log.csv', angle_shift = .2) :
    image_names = []
    angles = []
    new_img_directory = ''.join(csv_file_name.split('/')[:-1]) + '/IMG/'
    with open(csv_file_name) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            center_img_name = new_img_directory + line[0].split('/')[-1]
            center_angle = float(line[3])
            image_names.append(center_img_name)
            angles.append(center_angle)

            left_img_name = new_img_directory+line[1].split('/')[-1]
            left_angle = center_angle + angle_shift
            image_names.append(left_img_name)
            angles.append(left_angle)

            right_img_name = new_img_directory+line[2].split('/')[-1]
            right_angle = center_angle - angle_shift
            image_names.append(right_img_name)
            angles.append(right_angle)
    return image_names, np.array(angles)


# generates data
def generator(image_names,angles, batch_size=32):
    num_samples = len(angles)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(image_names, angles)
        for offset in range(0, num_samples, batch_size):
            images = []
            batch_image_names = image_names[offset:offset+batch_size]
            batch_angles = angles[offset:offset+batch_size]

            for img_name in batch_image_names:
                image = cv2.imread(img_name)
                image = image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
            X_train = np.array(images)
            y_train = batch_angles
            yield sklearn.utils.shuffle(X_train, y_train)

#### NN Architectures
from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Lambda, Convolution2D, Dropout
# dummy architecture 
def arch_simple():
    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    model.add(Flatten(input_shape=(160,320,3)))
    model.add(Dense(1))
    return model

# a CNN model, based on Nvidia architecture with a slight modification for  over fitting reduction
def arch_nvidia():
    model = Sequential()

    # pre-processing
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    # normalization
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Dropout(.2))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Dropout(.15))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(.1))
    model.add(Dense(50))
    model.add(Dropout(.05))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


# Trains the CNN
from sklearn.model_selection import train_test_split
def train(csv_file_name='data3/driving_log.csv'):
    print("=======================================")
    print("=========== %s ==============="%csv_file_name)
    print("=======================================")

    # process data
    image_names, angles = process_csv_data(csv_file_name=csv_file_name)
    train_image_names, validation_image_names, train_angles, validation_angles = train_test_split(image_names,angles, test_size=0.2)
    print("# train_X = %d, # train_y = %d, # valid_X = %d, # valid_y = %d"%(len(train_image_names), len(train_angles), len(validation_image_names), len(validation_angles))) 
    train_generator = generator(train_image_names, train_angles, batch_size=32)
    validation_generator = generator(validation_image_names, validation_angles, batch_size=32)

    #model = arch_simple()
    model = arch_nvidia()

    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_angles), validation_data=validation_generator, nb_val_samples=len(validation_angles), nb_epoch=7)
    #model.fit(X_train, y_train, validation_split=.2, shuffle=True, nb_epoch=7)
    #model.fit_generator(train_generator, steps_per_epoch= len(train_angles), validation_data=validation_generator, validation_steps=len(validation_angles), epochs=5, verbose = 1)
    model.save('model.h5')
   
    # save loss data
    np.savez('loss_history', loss=np.array(history_object.history['loss']), val_loss=np.array(history_object.history['val_loss']))

import matplotlib.pyplot as plt
# plots loss data in a jpg file
def plot_fig(file_name='loss_history.npz'):
    f = np.load(file_name)
    plt.plot(f['loss'], label="training set")
    plt.plot(f['val_loss'], label="validation set")
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    plt.savefig("examples/loss.jpg", bbox_inches='tight')
    #plt.show()

from keras.models import load_model
def print_model():
    model = load_model('model.h5')
    model.summary()


if __name__ == "__main__":
	train()


