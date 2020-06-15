# -*- coding: utf-8 -*-
"""
Created on Thu Oct 03 13:04:08 2019

@author: Dr. Mark M. Bailey | National Intelligence University
"""

#Import libraries
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import os



#Helper function
def get_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            file_list.append(os.path.join(root, name))
    return file_list

#Image classifier function (for list of images)
def image_classifier(image_path, model_path):
    classifier = load_model(model_path)
    image_list = get_files(image_path)
    #print(image_list)
    prediction_list = []
    for i in range(len(image_list)):
        test_image = image.load_img(image_list[i], target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        #Generate prediction and reference label
        result = classifier.predict(test_image)
        prediction_list.append(result)
    out_list = dict(zip(image_list, prediction_list))
    return out_list

#standard code to implement functions
if __name__ == '__main__':
    #point code to the directory where the images are 
    directory = '/Users/me/dev/CNN-exercise/test_set/'
    
    #direct the computer to the model path
    model_path = '/Users/me/dev/CNN-exercise/CNN_model.h5'
    
    #code to pull the outlist by running the image classifier on the directory, using the model
    out_list = image_classifier(directory,model_path)
    print(out_list)

    #set up zeroed-out true positive, etc counters so can add to them and determine the total numberss
    number_of_TP = 0
    number_of_FN = 0
    number_of_FP = 0
    number_of_TN = 0

    #out_list is a dictionary which contains the image file as the key and the prediction number as the value.
    #the "for loop" iterates through each item, and looks at whether the key contains "not_tank" or "test_set/tank",
    #then also checks if the value is 0 or anything 1 or greater.  It then adds 1 each time if finds an iterable that
    #matches both those conditions to the appropriate true positive, etc number.  Then it prints the total numbers.
    for image_file, number in list(out_list.items()):
        if 'not_tank' in image_file and number < 1:
            #print(image_file)
            #print(number)
            number_of_TN += 1
        if 'not_tank' in image_file and number >= 1:
            number_of_FP += 1
        if 'test_set/tank' in image_file and number < 1:
            number_of_FN += 1
        if 'test_set/tank' in image_file and number >= 1:
            number_of_TP += 1

    print("There were {} number of True Positives".format(number_of_TP))
    print("There were {} number of False Negatives".format(number_of_FN))
    print("There were {} number of False Positives".format(number_of_FP))
    print("There were {} number of True Negatives".format(number_of_TN))

