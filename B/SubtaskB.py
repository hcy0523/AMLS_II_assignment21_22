import os
import re
import shutil
from tqdm import tqdm

import numpy  as np
import pandas as pd
import joblib
import nltk
import ekphrasis
from collections import Counter

from gensim.models import Word2Vec, KeyedVectors
from nltk import word_tokenize
import multiprocessing

import keras
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SpatialDropout1D, Bidirectional

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


import seaborn as sns
import matplotlib.pyplot as plt
from IPython import display
import tensorflow_hub as hub

import warnings
warnings.filterwarnings("ignore")

class B:
    
# Step1: Data Pre-processing    
    def folderdir(self):
        
        Path =os.getcwd()
        
        return Path
    def read_data(self,Path,address):
        '''
        Read data from given address
        '''
        data_path=os.path.join(Path,address)
        data = pd.read_table(data_path,sep='\t',header=None)
        
        return data
        

    def add_label(self,sentiment):
        if sentiment == 'negative':
            return 0
        elif sentiment == 'positive':
            return 1
    
    def label_distrib(self,dataB):
        distrib=dataB.loc[:,['Sentiment','label']].value_counts().to_dict()
        
        return distrib
 
    def topic_counts(self,dataB):
        topic_counts=Counter(dataB.Topic)
        
        return len(topic_counts)
    
    def add_topic(self,topic,text):
        New_text=[]
        for i in range(len(text)):
            New_text.append(topic[i]+' '+text[i])

        return New_text
    
    #import ekphrasis library
    from ekphrasis.classes.preprocessor import TextPreProcessor
    from ekphrasis.classes.tokenizer import SocialTokenizer
    from ekphrasis.dicts.emoticons import emoticons
    
    text_processor = TextPreProcessor(
        # terms that will be normalized
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
            'time', 'url', 'date', 'number'],
        # terms that will be annotated
        annotate={"hashtag", "allcaps", "elongated", "repeated",
            'emphasis', 'censored'},
        fix_html=True,  # fix HTML tokens

        # corpus from which the word statistics are going to be used 
        # for word segmentation 
        segmenter="twitter", 

        # corpus from which the word statistics are going to be used 
        # for spell correction
        corrector="twitter", 

        unpack_hashtags=True,  # perform word segmentation on hashtags
        unpack_contractions=True,  # Unpack contractions (can't -> can not)
        spell_correct_elong=False,  # spell correction for elongated words

        # select a tokenizer. You can use SocialTokenizer, or pass your own
        # the tokenizer, should take as input a string and return a list of tokens
        tokenizer=SocialTokenizer(lowercase=True).tokenize,

        # list of dictionaries, for replacing tokens extracted from the text,
        # with other expressions. You can pass more than one dictionaries.
        dicts=[emoticons]
    )
    
    def Tokenize(self,Texts):
        token=[]
        for Text in Texts:
            words = [sentence for sentence in self.text_processor.pre_process_doc(Text) if (sentence!='s' and sentence!='\'')]
            token.append(words)
        words=[word for words in token for word in words]

        print("All words: {}".format(len(words)))
        # Create Counter
        counts = Counter(words)
        print("Unique words: {}".format(len(counts)))

        Most_common= counts.most_common()[:30]
        print("Top 30 most common words: {}\n".format(Most_common))

        vocab = {word: num for num, word in enumerate(counts, 1)}
        id2vocab = {v: k for k, v in vocab.items()}
        return token,vocab
    
# Step2: Word2Vec Pretraining    
    def word2vec(self,token,window,min_count,epochs):
        word2vec_model=Word2Vec(token,window=window, min_count=min_count,workers = multiprocessing.cpu_count())
        word2vec_model.train(token, total_examples = len(token), epochs = epochs)
        print('This is summary of Word2Vec: {}\n'.format(word2vec_model))
        
        index=word2vec_model.wv.key_to_index
        word2vec_model.wv.save_word2vec_format('./B/Word2Vec.vector')
        embed_matrix = np.zeros((len(index), 100))
        
        embed_dict={}
        for word, i in index.items():
            if word in word2vec_model.wv:
                embed_matrix[i] = word2vec_model.wv[word]
                embed_dict[word] = word2vec_model.wv[word]
                
        del word2vec_model # del model to Save RAM
        
        return embed_dict
    
# Step3: split train and test sets
    def tokenizer_lstm(self,X, vocab, seq_len,embed_dict):
        '''
        Returns tokenized tensor with left padding
        '''
        X_tmp = np.zeros((len(X), seq_len), dtype=np.int64)
        for i, text in enumerate(X):
            tokens = [word for word in self.text_processor.pre_process_doc(text) if (word!='s' and word!='\'')]
    #         tokens = [word for word in tokens if (word not in stop)]
            token_ids = [vocab[word] for word in tokens if word in embed_dict.keys()]###
            end_idx = min(len(token_ids), seq_len)
            start_idx = max(seq_len - len(token_ids), 0)
            X_tmp[i,start_idx:] = token_ids[:end_idx]

        return X_tmp
    
    def split(self,Text,vocab,label,seq_len,embed_dict):
        
        X = self.tokenizer_lstm(Text, vocab, seq_len,embed_dict)
        Y = tf.one_hot(label, depth=2)
        Y = np.array(Y)
        X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size=0.1, random_state=1000) 
        
        # sentiment distribution of train set
        train_dis={}
        train_dis['negative - 0']=np.sum(Y_train[:,0]==1)
        train_dis['positive - 1']=np.sum(Y_train[:,1]==1)
        
        # sentiment distribution of test set
        test_dis={}
        test_dis['negative - 0']=np.sum(Y_test[:,0]==1)
        test_dis['positive - 1']=np.sum(Y_test[:,1]==1)
        
        print('Train label Distribution:',train_dis,'\n')
        print('Test label Distribution:',test_dis,'\n')
        
        return X_train, X_test, Y_train, Y_test      
        
    
    def Final_PreProcess(self,address,window,min_count,epochs,seq_len):    
        PathB=self.folderdir()
        dataB=self.read_data(PathB,address)
        dataB.columns = ['ID','Topic','Sentiment','Text','label']
        dataB['label'] = dataB.Sentiment.apply(self.add_label)
        distrib=self.label_distrib(dataB)
        print('label Distribution for SubtaskB:',distrib,'\n')
        topic_counts=self.topic_counts(dataB)
        print("Unique Topics: {}\n".format(topic_counts))
        Text_Topic = self.add_topic(dataB.Topic,dataB.Text)
        dataB['Text_Topic']=Text_Topic
        
        token,vocab=self.Tokenize(dataB.Text_Topic)
        
        embed_dict=self.word2vec(token,window,min_count,epochs)
        X_train, X_test, Y_train, Y_test=self.split(dataB.Text_Topic,vocab,dataB.label,seq_len,embed_dict)
        return X_train, X_test, Y_train, Y_test,vocab,embed_dict
        

# Step4: Training
        
    def build_embedding_layer(self, vocab, embed_dict):
        """
        Build embedding matrix and embedding layer
        :param vocab_size: vocabulary size
        :param tok: tokenizer
        :param embeddings_index: embedding index
        :return: embedding matrix and embedding layer
        """
        #Build embedding matrix
        vocab_size=len(vocab)+1
        embedding_matrix = np.zeros((vocab_size, 100))
        for word, i in vocab.items():
            # Vector corresponds to word
            embedding_vector = embed_dict.get(word)###,embed_dict['<unk>']

            if embedding_vector is not None:
                # Ensure vector of embedding_matrix row matches word index
                embedding_matrix[i] = embedding_vector

        # Build embedding layer
        embedding_layer = Embedding(input_dim = vocab_size, output_dim = 100, weights = [embedding_matrix], input_length = 100, trainable=False)

        return embedding_layer

    def model_build(self,vocab,embed_dict):
        """
        Train, validate and test BiLSTM model, calculate accuracy of training and validation set
        :param X_train: tweet train data
        :param y_train: sentiment label train data
        :param embedding_layer: embedding layer
        :param X_test: tweet test data
        :param y_test: sentiment label test data
        :return: accuracy, recall, precision, F1 score and history
        """
        tf.debugging.set_log_device_placement(True)
        model = Sequential()
        embedding_layer=self.build_embedding_layer(vocab,embed_dict)
        model.add(embedding_layer)
        model.add(SpatialDropout1D(0.2))
        
        model.add(Bidirectional(LSTM(128,dropout = 0.5,return_sequences=True)))
        model.add(Bidirectional(LSTM(64,dropout = 0.5)))  #2 loss: 0.6735 - accuracy: 0.7028 - val_loss: 0.7640 - val_accuracy: 0.6669


    
        model.add(Dense(2, activation = 'softmax'))
        model.summary()
        return model
    
    def model_train(self,X_train, y_train,model,validation_split,batch_size,epochs,epochOfModel):
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', 
                      metrics = ['Recall','Accuracy','Precision'])
        
        history = model.fit(X_train, y_train, validation_split=validation_split, epochs = epochs, batch_size = batch_size )
        
        model.save('./B/taskB.h5'.format(epochOfModel))
        
        train_acc = history.history['Accuracy'][-1]
        val_acc = history.history['val_Accuracy'][-1]
        
        train_rec = history.history['recall'][-1]
        val_rec = history.history['val_recall'][-1]
        
        train_pre = history.history['precision'][-1]
        val_pre = history.history['val_precision'][-1]
        
        return history      
    
    def sketch_model():
    
        class_names = ['0: Negative','1: Positive']
        
        return
    
# Step5: prediction and evaluation
    def pred_eval(self, model,X_test, Y_test):
        class_names = ['0: Negative','1: Neutra;', '2: Positive']
        class_names = ['0: Negative','1: Positive']
        Y_pred = model.predict(X_test)
        Metrics = model.evaluate(X_test, Y_test, return_dict=True)
        print('Predict Metrcs for subtaskB:',Metrics,'\n')
        
        return Metrics