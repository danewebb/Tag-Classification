import numpy as np
import pickle
import os
import re


class Data_Processing():
    """
    This class handles splitting data between
    """

    def __init__(self, training_data_dir, testing_data_dir, vocab_dir):
        with open(training_data_dir, 'rb') as training_data:
            self.train_data =  pickle.load(training_data)

        with open(testing_data_dir, 'rb') as testing_data:
            self.test_data = pickle.load(testing_data)

        with open(vocab_dir, 'rb') as vocab_data:
            self.vocab_data = pickle.load(vocab_data)


        self.xtrain = []
        self.ytrain = []

        self.xtest = []
        self.ytest = []

        self.embedxtrain = []
        self.embedxtest = []

        self.train_lens = []
        self.test_lens = []



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

        # print(f'self.split_data ending lengths\n'
        #       f'xtrain length = {len(self.xtrain)}, ytrain length = {len(self.ytrain)}\n'
        #       f'xtest length = {len(self.xtest)}, ytest length = {len(self.ytest)}\n')




    def tags_to_one_hot(self):
        # [V, I, S, R, C, L, D, A, T, B, f, v, k, M, TO, OM, J, P, Q, IN, FC, FR, q, TE, TC, TR, FL]
        # or do [S, D, A, T] for now???

        one_hot_train = []
        hot_train = []
        for ii, tags in enumerate(self.ytrain):
            one_hot_train.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            hot_train.append([0, 0, 0, 0])
            #
            for tag in tags:

                if 'S' in tag or 'V' in tag or 'I' in tag or 'f' in tag or 'v' in tag or 'TO' in tag or 'OM' in tag or 'P' in tag or \
                        'Q' in tag or 'q' in tag or 'TE' in tag and hot_train[ii][0] == 0:
                    hot_train[ii][0] = 1
                if 'D' in tag or 'R' in tag or 'B' in tag or 'FR' in tag or 'TR' in tag and hot_train[ii][1] == 0:
                    hot_train[ii][1] = 1

                if 'A' in tag or 'C' in tag or 'M' in tag or 'J' in tag or 'FC' in tag or 'TC' in tag and \
                        hot_train[ii][2] == 0:
                    one_hot_train[ii][2] = 1

                if 'T' in tag or 'L' in tag or 'k' in tag or 'IN' in tag or 'FL' in tag and hot_train[ii][3] == 0:
                    hot_train[ii][3] = 1


            if np.sum(hot_train[ii][:]) == 4:
                one_hot_train[ii][0] = 1
            elif np.sum(hot_train[ii][:]) == 3:
                # 0, 1, 2
                if hot_train[ii][0] == 1 and hot_train[ii][1] == 1 and hot_train[ii][2] == 1:
                    one_hot_train[ii][1] = 1
                # 0, 1, 3
                elif hot_train[ii][0] == 1 and hot_train[ii][1] == 1 and hot_train[ii][3] == 3:
                    one_hot_train[ii][2] = 1
                # 0, 2, 3
                elif hot_train[ii][0] == 1 and hot_train[ii][2] == 1 and hot_train[ii][3] == 1:
                    one_hot_train[ii][3] = 1
                # 1, 2, 3
                elif hot_train[ii][1] == 1 and hot_train[ii][2] == 1 and hot_train[ii][3] == 1:
                    one_hot_train[ii][4] = 1
            elif np.sum(hot_train[ii][:]) == 2:
                # 0, 1
                if hot_train[ii][0] == 1 and hot_train[ii][1] == 1:
                    one_hot_train[ii][5] = 1
                # 0, 2
                elif hot_train[ii][0] == 1 and hot_train[ii][2] == 1:
                    one_hot_train[ii][6] = 1
                # 0, 3
                elif hot_train[ii][0] == 1 and hot_train[ii][3] == 1:
                    one_hot_train[ii][7] = 1
                # 1, 2
                elif hot_train[ii][1] == 1 and hot_train[ii][2] == 1:
                    one_hot_train[ii][8] = 1
                # 1, 3
                elif hot_train[ii][1] == 1 and hot_train[ii][3] == 1:
                    one_hot_train[ii][9] = 1
                # 2, 3
                elif hot_train[ii][2] == 1 and hot_train[ii][3] == 1:
                    one_hot_train[ii][10]
            elif np.sum(hot_train[ii][:]) == 1:
                # 0
                if hot_train[ii][0] == 1:
                    one_hot_train[ii][11] = 1
                # 1
                elif hot_train[ii][1] == 1:
                    one_hot_train[ii][12]
                # 2
                elif hot_train[ii][2] == 1:
                    one_hot_train[ii][13]
                # 3
                elif hot_train[ii][3] == 1:
                    one_hot_train[ii][14]
            else:
                one_hot_train[ii][15] = 1




        one_hot_test = []
        hot_test = []
        for ii, tags in enumerate(self.ytest):
            hot_test.append([0, 0, 0, 0])
            one_hot_test.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
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

            if np.sum(hot_test[ii][:]) == 4:
                one_hot_test[ii][0] = 1
            elif np.sum(hot_test[ii][:]) == 3:
                # 0, 1, 2
                if hot_test[ii][0] == 1 and hot_test[ii][1] == 1 and hot_test[ii][2] == 1:
                    one_hot_test[ii][1] = 1
                # 0, 1, 3
                elif hot_test[ii][0] == 1 and hot_test[ii][1] == 1 and hot_test[ii][3] == 3:
                    one_hot_test[ii][2] = 1
                # 0, 2, 3
                elif hot_test[ii][0] == 1 and hot_test[ii][2] == 1 and hot_test[ii][3] == 1:
                    one_hot_test[ii][3] = 1
                # 1, 2, 3
                elif hot_test[ii][1] == 1 and hot_test[ii][2] == 1 and hot_test[ii][3] == 1:
                    one_hot_test[ii][4] = 1
            elif np.sum(hot_test[ii][:]) == 2:
                # 0, 1
                if hot_test[ii][0] == 1 and hot_test[ii][1] == 1:
                    one_hot_test[ii][5] = 1
                # 0, 2
                elif hot_test[ii][0] == 1 and hot_test[ii][2] == 1:
                    one_hot_test[ii][6] = 1
                # 0, 3
                elif hot_test[ii][0] == 1 and hot_test[ii][3] == 1:
                    one_hot_test[ii][7] = 1
                # 1, 2
                elif hot_test[ii][1] == 1 and hot_test[ii][2] == 1:
                    one_hot_test[ii][8] = 1
                # 1, 3
                elif hot_test[ii][1] == 1 and hot_test[ii][3] == 1:
                    one_hot_test[ii][9] = 1
                # 2, 3
                elif hot_test[ii][2] == 1 and hot_test[ii][3] == 1:
                    one_hot_test[ii][10]
            elif np.sum(hot_test[ii][:]) == 1:
                # 0
                if hot_test[ii][0] == 1:
                    one_hot_test[ii][11] = 1
                # 1
                elif hot_test[ii][1] == 1:
                    one_hot_test[ii][12]
                # 2
                elif hot_test[ii][2] == 1:
                    one_hot_test[ii][13]
                # 3
                elif hot_test[ii][3] == 1:
                    one_hot_test[ii][14]
            else:
                one_hot_test[ii][15] = 1


        # print(f'self.tag_to_onehot ending lengths\n'
        #       f'xtrain length = {len(self.xtrain)}, ytrain length = {len(one_hot_train)}\n'
        #       f'xtest length = {len(self.xtest)}, ytest length = {len(one_hot_test)}\n')

        return one_hot_train, one_hot_test


    def random_idx(self, one_hot_train, one_hot_test, random_state=24):
        # randomize indices
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


        # print(f'self.random_idx ending lengths\n'
        #       f'xtrain length = {len(new_xtrain)}, ytrain length = {len(new_xtrainhot)}\n'
        #       f'xtest length = {len(new_xtest)}, ytest length = {len(new_xtesthot)}\n')

        return new_xtrain, new_xtrainhot, new_xtest, new_xtesthot


    def handle_vocab(self):
        # remove uncommon words
        vocab = []
        for word, value in self.vocab_data:
            if value > 1:
                vocab.append(word)
        return vocab

    def word_to_vec(self, vocab):
        """
        Converts words in xtrain and xtest into vector form. This is done by replacing the word with its place in a ranked
        vocab list.
        :return:
        """

        word_store = []

        para_vec = []

        # vocab = []

        # most common first
        vocab.reverse()

        for paras in self.xtrain: # loops each item in the list

            for para in paras: # takes the paragraph from the item
                for let in para: # each letter in paragraph
                    # if letter is a space, everything previous makes up a word
                    if let == ' ':
                        word = ''.join(word_store)
                        clean_word = self.word_cleaner(word)
                        word_store = []

                        for idx, w in enumerate(vocab):
                            if w == clean_word:
                                para_vec.append(idx)
                                break
                            elif w == word:
                                para_vec.append(idx)
                                break
                            elif idx == len(vocab) - 1:
                                para_vec.append(0)
                                break


                    else:
                        word_store.append(let)

            self.embedxtrain.append(para_vec)
            para_vec = []




        word_store = []
        for paras in self.xtest:
            for para in paras:
                for let in para:
                    if let == ' ':
                        word = ''.join(word_store)
                        clean_word = self.word_cleaner(word)
                        word_store = []

                        for idx, w in enumerate(vocab):
                            if w == clean_word:
                                para_vec.append(idx)
                                break
                            elif w == word:
                                para_vec.append(idx)
                                break
                            elif idx == len(vocab) - 1:
                                para_vec.append(0)
                                break

                    else:
                        word_store.append(let)

            self.embedxtest.append(para_vec)
            para_vec = []


        # print(f'self.word_to_vec lengths\n'
        #       f'xtrain length = {len(self.embedxtrain)}, ytrain length = {len(self.ytrain)}\n'
        #       f'xtest length = {len(self.embedxtest)}, ytest length = {len(self.ytest)}\n')

    # def list_to_ndarray(self, ytrain, ytest):
    #
    #     xtrain = np.asarray(self.embedxtrain)
    #     ytrain = np.asarray(ytrain)
        # for ii, para in enumerate(self.embedxtrain):
        #     xarrtrain = np.asarray(para)





    def vec_lengths(self):
        ele_len = 0
        for item in self.embedxtrain:
            for ele in item:
                ele_len += 1
            self.train_lens.append(ele_len)
            ele_len = 0

        for item in self.embedxtest:
            for ele in item:
                ele_len += 1
            self.test_lens.append(ele_len)
            ele_len = 0



    def word_cleaner(self, word):
        pattern = re.compile(r'\w')
        word_store = []
        for let in word:
            if pattern.match(let):
                word_store.append(let)


        return ''.join(word_store)




    def main(self, random_state=24):
        output_dict = dict()
        vocab = self.handle_vocab()
        self.split_data()
        # Lets not make these class variables to
        ytrain, ytest = self.tags_to_one_hot()
        train_para, train_lab, test_para, test_lab = self.random_idx(ytrain, ytest, random_state=random_state)


        self.word_to_vec(vocab)

        self.vec_lengths()

        # xtrain, ytrain, xtest, ytest = self.list_to_ndarray(ytrain, ytest)


        output_dict = {'trainx': self.embedxtrain, 'trainy': ytrain, 'testx': self.embedxtest, 'testy': ytest,
                       'voc': vocab, 'trainlen': self.train_lens, 'testlen': self.test_lens}

        return output_dict







if __name__ == '__main__':
    PCS = Data_Processing('training_dict.pkl', 'testing_dict.pkl', 'rank_vocab.pkl')
    PCS.main()

