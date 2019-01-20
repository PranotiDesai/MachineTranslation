import time
import numpy as np
import tensorflow as tf
from HindiNLP import HindiNLP
from Word2Vec import HindiWord2Vec
from tensorflow.python.framework import ops
from functions import preprocessing_text_from_source, get_batch, read_vocab_and_count

np.random.seed(11)
root_dir = "/users/imishra/workspace/Word2Vec"
text_file = root_dir+"/data/monolingual_extra_small2.hi"
vocab_file = root_dir+"/data/vocab.pickle"
nlp = HindiNLP()
vocab, word_counts, index_word, word_index = preprocessing_text_from_source(nlp, text_file, vocab_file)

vocab, word_counts, word_index, index_word, word_counts = read_vocab_and_count(vocab_file)

vocab_size = len(vocab)
########################################
# Create Training and Validation Data
########################################
validation_set_indices = np.random.choice(50, 3, replace=False)
train_data = []
start_time = time.time()
for word in vocab:
    if word in word_index:
        train_data.append(word_index[word])
    else:
        train_data.append(0)

end_time = time.time()
time_taken = end_time - start_time
print("Building Training Set Took: %.2f" % time_taken)

start_time = time.time()

batch_size = 16
embedding_size = 300
context_word_count = 1
window_size = 3
ops.reset_default_graph()
g = tf.get_default_graph()
X = tf.placeholder(shape=[batch_size], dtype=tf.int32, name="x")
Y = tf.placeholder(shape=[batch_size, 1], dtype=tf.int32, name="y")
validation_indices = tf.constant(validation_set_indices, dtype=tf.int32)

embd_vec = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
train_X = tf.nn.embedding_lookup(embd_vec, X)

# Initialize weights using Xavier Initialization
weights = tf.Variable(tf.truncated_normal(shape=[vocab_size, embedding_size], dtype=tf.float32,
                                          stddev=1.0/np.sqrt(embedding_size)))
# Initialize bias
bias = tf.Variable(tf.random_normal(shape=[vocab_size]))
# Compute the hidden layer output
h_out = tf.add(tf.matmul(train_X, tf.transpose(weights)), bias)
train_Y = tf.one_hot(Y, vocab_size)

## Negative samples to test
num_sampled = 20
## Use the tensorflow nce loss function
loss = tf.reduce_mean(tf.nn.nce_loss(weights=weights, biases=bias, labels=Y, inputs=train_X, num_sampled=num_sampled,
                                     num_classes=vocab_size))
optimizer = tf.train.RMSPropOptimizer(0.1).minimize(loss)

##############################
#  Compute Cosine Similarity
##############################
emb_vec = embd_vec/tf.sqrt(tf.reduce_sum(tf.square(embd_vec), 1, keep_dims=True))
validation_X = tf.nn.embedding_lookup(embd_vec, validation_indices)
cos_simi = tf.matmul(validation_X, tf.transpose(embd_vec))

end_time = time.time()
time_taken = end_time - start_time
print("Writing vocab to file took: %.2f" % time_taken)

hm_epochs = 200000
batch_index = 0
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    epoch_loss = 0
    for epoch in range(1, hm_epochs+1):
        log = ""
        batch_index, batch = get_batch(batch_index, train_data, batch_size, context_word_count, window_size)
        _, batch_loss = session.run([optimizer, loss], feed_dict={X:batch[:,0], Y:batch[:,1].reshape(-1,1)})
        epoch_loss += batch_loss
        if epoch == 1000 or epoch == 500:
            log += "Loss at epoch %d  is %.3f\n" %(epoch, epoch_loss / epoch)
        elif epoch % 5000 == 0:
            log += "Loss at epoch %d  is %.3f\n" %(epoch, epoch_loss/epoch)
            simi = cos_simi.eval()
            for i in range(int(validation_X.shape[0])):
                word = index_word[validation_set_indices[i]]
                nearest = (-simi[i, :]).argsort()[1:2]
                close_word = index_word[nearest[0]]
                log+="%s is close to %s \n" % (word, close_word) 
        print(log)
        with open(root_dir+"/logs.txt", "a") as myfile:
            myfile.write(log)
    print("Final Loss at epoch is"+str(epoch_loss/(epoch-1)))
    embd_vec = embd_vec.eval()
    tf.train.Saver().save(session, root_dir+"model/word2vec", global_step=1000)

word2Vec = HindiWord2Vec()
word2Vec.create(word_index, index_word, embd_vec)
word2Vec.save(root_dir+"/model/HindiWord2Vec.pickle")
