import numpy as np
import pickle
import os



class Data_Processing():


    def __init__(self, training_data_dir, testing_data_dir):
        with open(training_data_dir, 'rb') as training_data:
            self.train_data =  pickle.load(training_data)

        with open(testing_data_dir, 'rb') as testing_data:
            self.test_data = pickle.load(testing_data)

        self.xtrain = []
        self.ytrain = []

        self.xtest = []
        self.ytest = []



    def split_data(self):

        if self.train_data == None or self.test_data == None:
            ValueError('Both training and testing datasets must exist.')

        if self.train_data is not dict:
            TypeError('training_data argument must be a dictionary.')


        for value in self.train_data.values():
            tag_dict_temp = dict()
            tag_dict_temp = value['tag_dict']
            self.ytrain.append(tag_dict_temp['tags'])

            para_dict_temp = dict()
            para_dict_temp = value['para_dict']
            self.xtrain.append(para_dict_temp['paragraph'])


        if self.test_data is not dict:
            TypeError('testing_data argument must be a dictionary.')

        for value in self.test_data.values():
            tag_dict_temp = dict()
            tag_dict_temp = value['tag_dict']
            self.ytest.append(tag_dict_temp['tags'])

            para_dict_temp = dict()
            para_dict_temp = value['para_dict']
            self.xtest.append(para_dict_temp['paragraph'])




    def tags_to_one_hot(self):
        # [V, I, S, R, C, L, D, A, T, B, f, v, k, M, TO, OM, J, P, Q, IN, FC, FR, q, TE, TC, TR, FL]
        # or do [S, D, A, T] for now???

        one_hot_train = []
        for ii, tags in enumerate(self.ytrain):
            one_hot_train.append([0, 0, 0, 0])
            for tag in tags:

                if 'V' in tag or 'I' in tag or 'f' in tag or 'v' in tag or 'TO' in tag or 'OM' in tag or 'P' in tag or \
                        'Q' in tag or 'q' in tag or 'TE' in tag and one_hot_train[ii][0] == 0:
                    one_hot_train[ii][0] = 1
                if 'D' in tag or 'R' in tag or 'B' in tag or 'FR' in tag or 'TR' in tag and one_hot_train[ii][1] == 0:
                    one_hot_train[ii][1] = 1

                if 'A' in tag or 'C' in tag or 'M' in tag or 'J' in tag or 'FC' in tag or 'TC' in tag and \
                        one_hot_train[ii][2] == 0:
                    one_hot_train[ii][2] = 1

                if 'T' in tag or 'L' in tag or 'k' in tag or 'IN' in tag or 'FL' in tag and one_hot_train[ii][3] == 0:
                    one_hot_train[ii][3] = 1


        one_hot_test = []
        for ii, tags in enumerate(self.ytest):
            one_hot_test.append([0, 0, 0, 0])
            for tag in tags:

                if 'V' in tag or 'I' in tag or 'f' in tag or 'v' in tag or 'TO' in tag or 'OM' in tag or 'P' in tag or \
                        'Q' in tag or 'q' in tag or 'TE' in tag and one_hot_test[ii][0] == 0:
                    one_hot_test[ii][0] = 1

                if 'D' in tag or 'R' in tag or 'B' in tag or 'FR' in tag or 'TR' in tag and one_hot_test[ii][1] == 0:
                    one_hot_test[ii][1] = 1

                if 'A' in tag or 'C' in tag or 'M' in tag or 'J' in tag or 'FC' in tag or 'TC' in tag and \
                        one_hot_test[ii][2] == 0:
                    one_hot_test[ii][2] = 1

                if 'T' in tag or 'L' in tag or 'k' in tag or 'IN' in tag or 'FL' in tag and one_hot_test[ii][3] == 0:
                    one_hot_test[ii][3] = 1

        return one_hot_train, one_hot_test


    def random_idx(self, one_hot_train, one_hot_test, random_state=24):
        # randomize 
        np.random.seed(random_state)

        train_len = len(self.xtrain)
        train_idx = list(range(train_len))
        np.random.shuffle(train_idx)

        # pre-allocate lists
        new_xtrain = [None]*train_len
        new_xtrainhot = [None]*train_len

        for ii, idx in enumerate(train_idx):
            new_xtrain[ii] = self.xtrain[idx]
            new_xtrainhot[ii] = one_hot_train[idx]

        test_len = len(self.xtest)
        test_idx = list(range(test_len))
        np.random.shuffle(test_idx)
        new_xtest = [None]*test_len
        new_xtesthot = [None]*test_len

        for ii, idx in enumerate(test_idx):
            new_xtest[ii] = self.xtest[idx]
            new_xtesthot[ii] = one_hot_test[idx]

        return zip(new_xtrain, new_xtrainhot), zip(new_xtest, new_xtesthot)


    def main(self, random_state=24):
        self.split_data()
        # Lets not make these class variables to
        ytrain, ytest = self.tags_to_one_hot()
        train_data, test_data = self.random_idx(ytrain, ytest, random_state=random_state)



if __name__ == '__main__':
    PCS = Data_Processing('training_dict.pkl', 'testing_dict.pkl ')

