import pickle
import collections
import numpy as np
import time

def get_word_counts(words):
    """
    This function build a word count dictionary for the given vocabulary
    """
    word_count = {}
    for word in words:
        if word not in word_count.keys():
            word_count[word] = 1
        else:
            word_count[word] += 1

    return sorted(word_count.items(), key=lambda kv: kv[1], reverse=True)


def create_index(words):
    """
    This function index the most common word using an unique integer where the word is the key
    and the unique integer is the value. It creates another dictionary where the key value pair
    is swapped and integer serves as key and word serves as value
    """
    index_word_dictionary = {}
    word_index_dictionary = {}
    for word in words:
        if word not in index_word_dictionary.keys():
            index_word_dictionary[len(word_index_dictionary)] = word
            word_index_dictionary[word] = len(word_index_dictionary)

    return index_word_dictionary, word_index_dictionary


def get_batch(index, word_vectors, batch_size, context_words_count, window_size):
    """
    THis method creates a batch for the training the tensorflow model
    :param index: start index of the batch
    :param word_vectors: vector representation of the words
    :param batch_size: size of each batch
    :param context_words_count: Numbe rof context word to consider
    :param window_size: size the sliding window
    :return:
    """
    assert batch_size % context_words_count == 0
    assert context_words_count <= window_size
    batch = np.zeros((batch_size, 2), dtype=np.int32)
    window = collections.deque(maxlen=window_size)
    for i in range(window_size):
        window.append(word_vectors[index])
        index = (index+1) % len(word_vectors)
    window_slide_count = np.int32(batch_size/context_words_count)
    for i in range(window_slide_count):
        word_index = np.int32(window_size/2)
        context_words_used = []
        for j in range(context_words_count):
            while True:
                # Iterate till a index is generated which is not in alreaded generated list
                idx = np.random.randint(0, window_size)
                if idx!=word_index and idx not in context_words_used:
                    break
            context_words_used.append(idx)
            batch[i*context_words_count+j,0] = window[word_index]
            batch[i*context_words_count+j,1] = window[idx]
        window.append(word_vectors[index])
        index = (index+1)%len(word_vectors)
    index = (index-window_size)%len(word_vectors)
    return index, batch


def preprocessing_text_from_source(nlp, text_file, vocab_target_file):
    """
    This method does all the loading and pre-processing of data
    :param nlp: HindiNLP object
    :return: vocab, word_counts, index_word, word_index
    """
    start_time = time.time()
    text = nlp.read_file(text_file)
    end_time = time.time()
    time_taken = end_time - start_time
    print("Time Taken to read File: %.2f" % time_taken)

    start_time = time.time()
    vocab = nlp.tokenize(text)
    end_time = time.time()
    time_taken = end_time - start_time
    print("Time Taken to Tokenize: %.2f" % time_taken)

    start_time = time.time()
    file = open(vocab_target_file, "wb")
    pickle.dump(vocab, file)
    file.close()
    end_time = time.time()
    time_taken = end_time - start_time
    print("Writing vocab to file took: %.2f" % time_taken)

    start_time = time.time()
    word_counts = get_word_counts(vocab)
    end_time = time.time()
    time_taken = end_time - start_time
    print("Finding word counts Took: %.2f" % time_taken)

    start_time = time.time()
    index_word, word_index = create_index(vocab)
    end_time = time.time()
    time_taken = end_time - start_time
    print("Indexing Took: %.2f" % time_taken)

    return vocab, word_counts, index_word, word_index

def read_vocab_and_count(filename):
    """
    This method reads the saved vocabulary
    :param filename:
    :return:
    """
    start_time = time.time()

    file = open(filename, "rb")
    vocab = pickle.load(file)
    word_counts = get_word_counts(vocab)

    #reduce tehe vocab size
    vocab_size = 10000
    word_counts = word_counts[0:vocab_size]
    vocab = [word_count[0] for word_count in word_counts]
    new_word_counts = {}
    for k, v in word_counts:
        new_word_counts[k] = len(new_word_counts)
    word_counts = new_word_counts

    index_word, word_index = create_index(vocab)
    file.close()

    end_time = time.time()
    time_taken = end_time - start_time
    print("Reading Vocabs from file took: %.2f" % time_taken)
    return vocab, word_counts, word_index, index_word, word_counts
