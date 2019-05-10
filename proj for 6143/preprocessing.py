# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class Preprocesser(object):
    def __init__(self, data, categorical_cols, numerical_cols, fill_cols, drop_cols, lower_cols):
        self.data = data
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.fill_cols = fill_cols
        self.drop_cols = drop_cols
        self.lower_cols = lower_cols

    def drop_useless_cols(self):
        print('Drop useless columns')
        self.data = self.data.drop(self.drop_cols, axis=1)

    def to_lower(self):
        print('Convert english to lowercase')
        cols = self.lower_cols

        def str_lower(x):
            if x == np.nan:
                return x
            else:
                return str(x).lower()
        for col in cols:
            self.data[col] = self.data[col].apply(lambda x: str_lower(x))

    def fill_na(self):
        print('Fill in missing values')
        cols = self.fill_cols
        cate_cols = self.categorical_cols
        num_cols = self.numerical_cols
        for col in cols:
            if col in cate_cols:
                col_mode = self.data[col].mode()[0]
                self.data[col] = self.data[col].fillna(col_mode)
            elif col in num_cols:
                col_mean = self.data[col].mean()
                self.data[col] = self.data[col].fillna(col_mean)

    def replace_low_freq_value(self, low_freq=2):
        print('Replace low frequency values in categorical columns')

        def replace_low(low_freq_set, x):
            if x in low_freq_set:
                return 'low_freq'
            else:
                return x
        cols = self.categorical_cols
        for col in cols:
            self.data[col] = self.data[col].map(str)
            value_counts = self.data[col].value_counts()
            low_freq_value = set(list(
                value_counts[value_counts <= low_freq].index))
            self.data[col] = self.data[col].apply(
                lambda x: replace_low(low_freq_value, x))

    def label_encoder(self):
        print('Convert categorical columns to digital')
        cols = self.categorical_cols
        for col in cols:
            label_enc = LabelEncoder()
            self.data[col] = label_enc.fit_transform(self.data[col])


if __name__ == '__main__':
    pass
