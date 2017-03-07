import numpy as np
import pandas as pd

class NLP():
    def __init__(self,train_df,test_df):
        self.train_df = train_df
        self.test_df = test_df


    # IMPLEMENT THIS
    # Out: numpy array for training descriptions, numpy array for test descriptions
    # Default: number of words, number of characters
    def convert_descriptions(self):
        train_descriptions = self.train_df['description']
        test_descriptions = self.test_df['description']

        converted_train = []
        for str in train_descriptions:
            converted = [len(str.split()),len(str)]
            converted_train.append(converted)

        converted_test = []
        for str in test_descriptions:
            converted = [len(str.split()),len(str)]
            converted_test.append(converted)



        converted_train = np.vstack((converted_train))
        converted_test = np.vstack((converted_test))
        print ("converted training descriptions", converted_train)
        print ("converted test descriptions", converted_test)
        return converted_train, converted_test
