import tensorflow as tf
import tensorboard as tb
import numpy as np
import pickle
import os
import warnings
import datetime
from tensorflow.python.framework import ops
# from sklearn.metrics import precision_recall_fscore_support
# from sklearn.metrics.classification import accuracy_score
from Processing import Data_Processing as DP
from Classifier_RNN import LSTM_RNN


def prep_batch(x, y, seq_len, batch_size, idx):
    xlen = len(x)
    ylen = len(y)
    assert xlen == ylen

    if idx + batch_size <= xlen:
        newx = x[idx:idx+batch_size]
        newy = y[idx:idx+batch_size]
        newlen = seq_len[idx:idx+batch_size]

    else:
        carryover = batch_size - (xlen - idx)

        newx = np.concatenate((x[idx:], x[:carryover]))
        newy = np.concatenate((y[idx:], y[:carryover]))
        newlen = np.concatenate((seq_len[idx:], seq_len[:carryover]))

    # starting point for next batch
    idx += batch_size

    return newx, newy, newlen, idx



# ignore all of the tensorflow warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()


# n_classes?? how many different tags?

n_classes = 16

#
# train_data_dir = 'training_dict.pkl'
# test_data_dir = 'testing_dict.pkl'

# with open('training_dict.pkl', 'rb') as f1:
#     train_dict = pickle.load(f1)
#
# with open('testing_dict.pkl', 'rb') as f2:
#     test_dict = pickle.load(f2)

with open('rank_vocab.pkl', 'rb') as f3:
    vocab_dict = pickle.load(f3)
# make sure vocab isn't unnecesarrily copied

DP = DP('training_dict.pkl', 'testing_dict.pkl', 'rank_vocab.pkl')
# xtrain, ytrain, xtest, ytest, pro_vocab = DP.main(random_state=24)

processed_data = DP.main(random_state=24)
pro_vocab = processed_data['voc']
xxtrain = processed_data['trainx']
yytrain= processed_data['trainy']
xxtest = processed_data['testx']
yytest = processed_data['testy']
train_len = processed_data['trainlen']
test_len = processed_data['testlen']

old_num = 0

# need longest paragraph length for padding
for num in train_len:
    if num > old_num:
        old_num = num

# train paras
maxtrain_len = old_num

# padding to get an even matrix
xtrain = tf.keras.preprocessing.sequence.pad_sequences(xxtrain, value=0, padding='post', maxlen=old_num)
ytrain = np.asarray(yytrain)

old_num = 0
# need longest paragraph length for padding
for num in test_len:
    if num > old_num:
        old_num = num

# test paras
xtest = tf.keras.preprocessing.sequence.pad_sequences(xxtest, value=0, padding='post', maxlen=old_num)
ytest = np.asarray(yytest)

maxtest_len = old_num

# pad based on longest length
if maxtrain_len > maxtest_len:
    maxseq_len = maxtrain_len
else:
    maxseq_len = maxtest_len

vocab_size = len(pro_vocab)


n_samples= None # Set n_samples=None to use the whole dataset

summaries_dir= 'logs/'# Directory where TensorFlow summaries will be stored'
batch_size = 100 #Batch size
train_steps = 1000 #Number of training steps
hidden_size= [75] # Hidden size of LSTM layer
embedding_size = 75 # Size of embeddings layer

random_state = 24 # Random state used for data splitting. Default is 0
epoch = 0
idx = 0

learning_rate = 0.01
test_size = 0.2
dropout_keep_prob = 0.5 # 0<dropout_keep_prob<=1. Dropout keep-probability
sequence_len = None # Maximum sequence length
validate_every = 100 # Step frequency in order to evaluate the model using a validation set'

# Prepare summaries
summaries_dir = '{0}/{1}'.format(summaries_dir, datetime.datetime.now().strftime('%d_%b_%Y-%H_%M_%S'))
train_writer = tf.summary.FileWriter(summaries_dir + '/train')
validation_writer = tf.summary.FileWriter(summaries_dir + '/validation')

# Prepare model directory
model_name = str(datetime.time())
model_dir = '{0}/{1}'.format('checkpoints', model_name)

# if not os.path.exists(model_dir):
#     os.makedirs(model_dir)



lstm_model = LSTM_RNN(hidden_size, vocab_size, embedding_size, n_classes, max_len=maxseq_len,
                      learning_rate=learning_rate, rand_state=random_state)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
train_writer.add_graph(lstm_model.input.graph)


trainloss = []
steps = []
step = 0

for ii in range(0, train_steps):
    x, y, sseq_len, idx = prep_batch(xtrain, ytrain, train_len, batch_size, idx)
    seq_len = np.asarray(sseq_len)
    loss, _, summary = sess.run([lstm_model.loss, lstm_model.train_step, lstm_model.merged],
                                feed_dict={lstm_model.input: x, lstm_model.target: y,
                                           lstm_model.seq_len: seq_len,
                                           lstm_model.dropout_keep_prob: dropout_keep_prob})

    train_writer.add_summary(summary, ii)
    trainloss.append(loss)
    steps.append(ii)

    print(f'Batch {ii+1} of {train_steps}, LOSS: {loss}')



# validation

# plot


# Save model
# Invalid argument: Failed to create a directory: checkpoints/00:00:00; Invalid argument
# checkpoint_file = '{}/model.ckpt'.format(model_dir)
# save_path = saver.save(sess, checkpoint_file)
# print('Model saved in: {0}'.format(model_dir))


# tensorboard



