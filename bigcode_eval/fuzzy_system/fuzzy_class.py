
from bigcode_eval.fuzzy_system.fuzzy import *
import matplotlib.pyplot as plt
import pandas as pd
import math
import random
from bigcode_eval.fuzzy_system.data_handle import main

class Fuzzyclass:
    def __init__(self, epochs = 400, max_acc = 0.00, last_epoch = 406):
        np.random.seed(last_epoch)
        X_train, Y_train, X_test, Y_test, min_values, max_values = main()
        self.min_values = min_values
        self.max_values = max_values
        self.clf2 = FuzzyMMC(sensitivity=1, exp_bound=0.1, animate=True)
        self.clf2.fit(X_train, Y_train)
        acc = self.clf2.score(X_test, Y_test)
        max_acc = acc
        print("+++++++++++++++++++++++++++++++++++")
        print("+++++++++++++++++++++++++++++++++++")
        print(f"epoch is {last_epoch}, acc is {max_acc}")
        print("+++++++++++++++++++++++++++++++++++")
        print("+++++++++++++++++++++++++++++++++++")

    def fuzzy_prejudge(self, data_X):
        normalized_array = (data_X - self.min_values) / (self.max_values - self.min_values)
        normalized_array = np.round(normalized_array, decimals=2)
        result = self.clf2.Judge(normalized_array)
        return result

if __name__ == "__main__":
    fuzzy_content = Fuzzyclass(epochs = 400, max_acc = 0.00, last_epoch = 406)
    print('+++++++++++++++++++++++++++++++=')
    print(fuzzy_content.fuzzy_prejudge([[0.15, 0.13, 0.55]]))