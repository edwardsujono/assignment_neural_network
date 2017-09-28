import pandas as pd
import numpy as np


class DataCollector:

    def __init__(self):
        self.df_train = pd.read_csv("./data/sat_train.txt", delimiter=' ')
        self.df_test = pd.read_csv("./data/sat_test.txt", delimiter=' ')

        # change the index
        self.df_train.columns = range(self.df_train.shape[1])
        self.df_test.columns = range(self.df_test.shape[1])

        self.x_train = self.df_train[range(36)]
        self.y_train = self.df_train[36]

        self.x_test = self.df_test[range(36)]
        self.y_test = self.df_test[36]

        self.x_min_train = self.x_train.min()
        self.x_max_train = self.x_train.max()

        self.x_min_test = self.x_test.min()
        self.x_max_test = self.x_test.max()

        return

    def get_train_data(self):

        return self.normalize_data(self.x_max_train, self.x_min_train, self.x_train), \
               self.one_hot_encoding_data(self.y_train)

    def get_test_data(self):

        return self.normalize_data(self.x_max_test, self.x_min_test, self.x_test),\
               self.one_hot_encoding_data(self.y_test)

    def one_hot_encoding_data(self, df, limit_number=6):

        # in this case data 6 is missing so, 7 we assume to be 6
        df[df == 7] = 6
        df_return = np.zeros((df.shape[0], limit_number))
        df_return[np.arange(df.shape[0]), df - 1] = 1
        return df_return

    def normalize_data(self, max_value, min_value, value):
        return (value - min_value)/(max_value-min_value)
