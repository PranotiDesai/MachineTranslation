import pickle

class HindiWord2Vec:
    """
    This class represents the trained word2vec model
    """

    def __init__(self):
        self.word_index = None
        self.index_word = None
        self.embeddings = None

    def create(self, word_index, index_word, embeddings):
        """
        :param word_index:
        :param index_word:
        :param embeddings:
        :return:
        """
        self.word_index = word_index
        self.index_word = index_word
        self.embeddings = embeddings

    def get_word_embedding(self, word):
        """
        This is a method which returns the
        """
        if type(word) == int:
            return self.embeddings[word]
        else:
            return self.embeddings[self.word_index[word]]

    def save(self, filename):
        """
        This is a method to serialize the object to file
        """
        file = open(filename, "wb")
        pickle.dump(self, file)
        file.close()

    @staticmethod
    def load(filename):
        """
        This is a static method to deserialize the object from the file
        """
        file = open(filename, "rb")
        data = pickle.load(file)
        file.close()
        return data
