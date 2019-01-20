# MachineTranslation

Machine Translation is one the most important areas in Natural language Processing because we have several languages across the world and it becomes difficult to communicate in a language that we don’t understand. For instance, suppose someone posted a very good article on neural network  in a foreign language. If we see this in conventional way, we need to learn that language first before we can read that article. But suppose, how helpful it would be if we have an intelligent learning algorithm at our disposal which can convert text from any language to the language we want. 

Developed a sequence to sequence machine translation model using LSTM Neural Network for predicting corresponding target text sequence for a given source sequence. This work includes following components that we have created.

1.	Hindi NLP: Developed a framework for processing Hindi Text which includes tokenization, stop words processing and Unicode normalization.  
2.	Word2Vec: Created a word2vec model for Hindi which is used to create word embeddings for Hindi words using the context as well. 
3.	Learning Model: Developed a Character2Character Sequence model uses LSTM which was easier to train then the Word2Word sequence model for predicting the source language(English) to target Language(Hindi) as well as with the same source language to French as target language.

We tried working on different dataset for the Hindi processing referenced in [3] and we also tried our model on the French dataset[5] and it seems that the model performs a bit better on the French dataset .

## Technology Stack

Language: Python 
Packages Used:
1.	NLTK 
2.	pandas
3.	numpy
4.	scikit-learn
5.	tensorflow
6.	keras
7.	matplotlib

## Results

**Hindi Translation:**
1.	Avg bleu score on 10 random text(weights(1.0, 0, 0, 0)): (Train - 0.46, Val - 0.31)
2.	Avg bleu score on 10 random text(weights(0.5, 0.5, 0,  0)): (Train - 0.16, Val - 0.24)
3.	Avg bleu score on 10 random text(weights(0.33, 0.33, 0.33, 0)): (Train-0.11, Val - 0.14)

**French Translation:**
1.	Avg bleu score on 10 random text(weights(1.0, 0, 0, 0)): (Train - 0.48, Val - 0.32)
2.	Avg bleu score on 10 random text(weights(0.5, 0.5, 0, 0)): (Train - 0.30, Val - 0.24)
3.	Avg bleu score on 10 random text(weights(0.33, 0.33, 0.33, 0)): (Train - 0.19, Val -0.07)

## References
[1]	O. Dhariya, S. Malviya, and U. S. Tiwary, “A hybrid approach forhindi-english machine translation,”CoRR, vol. abs/1702.01587,  2017.[Online]. Available: http://arxiv.org/abs/1702.01587

[2] S. P. Singh, A. Kumar, H. Darbari, L. Singh, A. Rastogi, and S. Jain, “Ma-chine translation using deep learning: An overview,” in2017 InternationalConference on Computer, Communications and Electronics (Comptelix),July 2017, pp. 162–167

[3]	IIT Bombay English-Hindi Corpus
 http://www.cfilt.iitb.ac.in/iitb_parallel/

[4]	Tab-delimited Bilingual Sentence Pairs
 http://www.manythings.org/anki/ 

[5]	Understanding LSTM Networks
http://colah.github.io/posts/2015-08-Understanding-LSTMs/

[5]	Sequence to Sequence modelling 
https://www.analyticsvidhya.com/blog/2018/03/essentials-of-deep-learning-sequence-to-sequence-modelling-with-attention-part-i/ 

7.	Introduction to Sequence 2 Sequence Learning 
https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html 

