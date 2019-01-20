import re


class HindiNLP:
    """
        HindiNLP: A NLP library for Devnagri language.
    """

    def __init__(self):
        # Read the punctuation marks from the file
        self.punctuations = self.read_file("/users/imishra/workspace/Word2Vec/data/punctuations.txt")
        self.stopwords = self.read_file("/users/imishra/workspace/Word2Vec/data/stopwords.txt", str_type=False)

    def read_file(self, filename, encoding="utf8", str_type=True, tokenize=False, normalize_text=False):
        """
        This method reads teh given hindi text file in the desired format
        :param filename: Name of the text file
        :param encoding: type of encodings used
        :param str_type: A boolean to denote if need to read the file as whole string
        :param tokenize: A boolean to denote if the text also needs to be tokenize
        :param normalize_text: A boolean to denote if the text also needs to be normlaize
        :return:
        """
        file = open(filename, 'r', encoding=encoding)
        if str_type:
            data = file.read()
            if normalize_text:
                data = self.normalize(data)
            if tokenize:
                data = self.tokenize(data)
        else:
            if not normalize_text or not tokenize:
                data = [line.rstrip() for line in file.readlines()]
            elif normalize_text and tokenize:
                data = [self.tokenize(self.normalize(line.rstrip())) for line in file.readlines()]
            elif normalize_text:
                data = [self.normalize(line.rstrip()) for line in file.readlines()]
            else:
                data = data = [self.tokenize(line.rstrip()) for line in file.readlines()]

        file.close()
        return data

    def tokenize(self, text, sentence=False):
        """
        A method to create tokens from teh text
        :param text:
        :param sentence:
        :return:
        """
        # First Split by the space
        if sentence:
            return [sentence + u"|" for sentence in text.split(u"|")]

        else:
            text = re.sub('[a-zA-z]', '', text)
            raw_tokens = re.split("[" + re.escape(self.punctuations) + "\s]", text)
            tokens = [token for token in raw_tokens if token.strip() != ""]
            return tokens

    def normalize(self, text):
        """
        In order to normalize the unicode representation of the text we need to switch to one representatation
        of the characters which has multiple representation in unicode for example the characters with nukta
        :param text: Text to normalize
        :return:
        """
        #################################################################
        # Remove non-significant marker charcaters such as Word Joiner,
        # soft-hypehn(to denote broken words) and other marker characters
        #################################################################
        marker_characters = [u'\u2060', u'\u00AD', u'\u200B', u'\uFEFF', u'\u00A0', u'\uFFFE', u'\u200C', u'\u200D']
        for char in marker_characters:
            text = text.replace(char, '')

        # Derived characters using nukta
        nukta = u'\u093C'
        original_characters = [u'\u0928', u'\u0915', u'\u0916', u'\u0917', u'\u091C', u'\u0921', u'\u0922', u'\u092B',
                               u'\u092F']
        nukta_characters = [char + nukta for char in original_characters]
        nukta_one_char_representation = [u'\u0929', u'\u0958', u'\u0959', u'\u095A', u'\u095B', u'\u095C', u'\u095D',
                                         u'\u095E',
                                         u'\u095F']

        # Now take other derived characters
        # and then combine nukta character and other derived characters
        derivied_characters = [u'\u0905' + u'\u0945', u'\u0905' + u'\u093A', u'\u0906' + u'\u093A',
                               u'\u0905' + u'\u094F',
                               u'\u0905' + u'\u0956', u'\u0905' + u'\u0957', u'\u0917' + u'\u0952',
                               u'\u091C' + u'\u0952',
                               u'\u0921' + u'\u0952', u'\u092C' + u'\u0952']
        one_char_representation = [u'\u0972', u'\u0973', u'\u0974', u'\u0975', u'\u0976', u'\u0977', u'\u097B',
                                   u'\u097C',
                                   u'\u097E', u'\u097F']

        derivied_characters.extend(nukta_characters)
        one_char_representation.extend(nukta_one_char_representation)

        for i in range(len(derivied_characters)):
            text = text.replace(derivied_characters[i], one_char_representation[i])

        return text

    def remove_stop_words(self, words):
        """
        A method to remove the stopwords from the list of token
        :param words:
        :return:
        """
        return [word for word in words if word not in self.stopwords]
