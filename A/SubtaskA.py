import os
import numpy  as np
import pandas as pd
import ekphrasis
from collections import Counter

from gensim.models import Word2Vec, KeyedVectors
import multiprocessing

import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, SpatialDropout1D, Bidirectional, Embedding
from tensorflow.keras.models import Sequential,load_model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

import warnings
warnings.filterwarnings("ignore")

class A:
    
# Step1: Data Pre-processing    
    def folderdir(self):
        
        Path =os.getcwd()
        
        return Path
    def read_data(self,Path,address):
        '''
        param: Path: path of the filefolder
        param: address: figure address
        return: data from given address
        '''
        data_path=os.path.join(Path,address)
        data = pd.read_table(data_path,sep='\t',header=None)
        
        return data
        

    def add_label(self,sentiment):
        '''
        param: sentiment:text labels
        return: figure labels
        '''    
        if sentiment == 'negative':
            return 0
        elif sentiment == 'neutral':
            return 1
        elif sentiment == 'positive':
            return 2
    
    def label_distrib(self,dataA):
        '''
        param: dataA
        return distrib distribution of labels
        '''
        distrib=dataA.loc[:,['Sentiment','label']].value_counts().to_dict()
        
        return distrib
    

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
        '''
        param: Texts: sentences raw data
        return: token: tokenized words
        return: vocab: vocabulary of word--words to id
        '''
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
        '''
        param: token: words be tokenized from sentences.
        param: window: context length
        param: min_count: times a word appear to be consider
        param: epochs: times of iteration
        return: embed_dict: embeded dictionary
        '''
        word2vec_model=Word2Vec(token,window=window, min_count=min_count,workers = multiprocessing.cpu_count())
        word2vec_model.train(token, total_examples = len(token), epochs = epochs)
        print('This is summary of Word2Vec: {}\n'.format(word2vec_model))
        
        index=word2vec_model.wv.key_to_index
        word2vec_model.wv.save_word2vec_format('./A/Word2Vec.vector')
        embed_matrix = np.zeros((len(index), 100))
        
        embed_dict={}
        for word, i in index.items():
            if word in word2vec_model.wv:
                embed_matrix[i] = word2vec_model.wv[word]
                embed_dict[word] = word2vec_model.wv[word]
                
        del word2vec_model # del model to Save RAM
        
        return embed_dict
    
# Step3: split train and test sets
    def tokenizer_lstm(self,X, vocab, maxlen,embed_dict):
        '''
        param: vocab:vocabulary of words
        param: maxlen: The maximum words of a sentences contained
        param: embed_dict: embeded dictionary
        Return: tokenized tensor with left padding
        '''
        X_tmp = np.zeros((len(X), maxlen), dtype=np.int64)
        for i, text in enumerate(X):
            tokens = [word for word in self.text_processor.pre_process_doc(text) if (word!='s' and word!='\'')]
    #         tokens = [word for word in tokens if (word not in stop)]
            token_ids = [vocab[word] for word in tokens if word in embed_dict.keys()]###
            end_idx = min(len(token_ids), maxlen)
            start_idx = max(maxlen - len(token_ids), 0)
            X_tmp[i,start_idx:] = token_ids[:end_idx]

        return X_tmp
    
    def split(self,Text,vocab,label,test_size,maxlen,embed_dict):
        '''
        param: Text: the sentences raw data
        param: vocab:vocabulary of words
        param: label the figure labels
        param: test_size:proportion to split test set
        param: maxlen: The maximum words of a sentences contained
        param:embed_dict: embeded dictionary
        return: train set and text set Y in one-hot form.
        '''
        X = self.tokenizer_lstm(Text, vocab, maxlen,embed_dict)
        Y = tf.one_hot(label, depth=3)
        Y = np.array(Y)
        X, Y = shuffle(X,Y)
        X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size=test_size, random_state=0) 
        
        # sentiment distribution of train set
        train_dis={}
        train_dis['negative - 0']=np.sum(Y_train[:,0]==1)
        train_dis['neutral - 1']=np.sum(Y_train[:,1]==1)
        train_dis['positive - 2']=np.sum(Y_train[:,2]==1)
        
        # sentiment distribution of test set
        test_dis={}
        test_dis['negative - 0']=np.sum(Y_test[:,0]==1)
        test_dis['neutral - 1']=np.sum(Y_test[:,1]==1)
        test_dis['positive - 2']=np.sum(Y_test[:,2]==1)
        
        print('Train label Distribution:',train_dis,'\n')
        print('Test label Distribution:',test_dis,'\n')
        
        return X_train, X_test, Y_train, Y_test
    
    def Final_PreProcess(self,address,window,min_count,epochs,maxlen,test_size):    
        '''
        param: address: figure address
        param: window: context length
        param: min_count: times a word appear to be consider
        param: epochs: times of iteration
        param: test_size:proportion to split test set
        param: maxlen: The maximum words of a sentences contained
        return: train set and text set Y in one-hot form. 
        return: vocab:vocabulary of words
        return: embed_dict: embeded dictionary
        '''
        PathA=self.folderdir()
        dataA=self.read_data(PathA,address)
        dataA.columns = ['ID','Sentiment','Text','Nan']
        dataA['label'] = dataA.Sentiment.apply(self.add_label)
        distrib=self.label_distrib(dataA)
        print('label Distribution for SubtaskA:',distrib,'\n')
        token,vocab=self.Tokenize(dataA.Text)
        
        embed_dict=self.word2vec(token,window,min_count,epochs)
        X_train, X_test, Y_train, Y_test=self.split(dataA.Text,vocab,dataA.label,test_size,maxlen,embed_dict)
        return X_train, X_test, Y_train, Y_test,vocab,embed_dict
        

# Step4: Training
        
    def build_embedding_layer(self, vocab, embed_dict,maxlen):
        """
        Build embedding matrix and embedding layer
        :param vocab: word2vec vocabulary
        :param embed_dict: embedded dictionary
        :param maxlen: The maximum words of a sentences contained
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
        embedding_layer = Embedding(input_dim = vocab_size, output_dim = 100, weights = [embedding_matrix], input_length = maxlen, trainable=False)

        return embedding_layer

    def model_build(self,vocab,embed_dict,maxlen):
        """
        Bulid BiLSTM model.
        param vocab: word2vec vocabulary
        param embed_dict: embedded dictionary
        param maxlen: The maximum words of a sentences contained
        return；model
        """
        tf.debugging.set_log_device_placement(True)
        model = Sequential()
        embedding_layer=self.build_embedding_layer(vocab,embed_dict,maxlen)
        model.add(embedding_layer)
        model.add(SpatialDropout1D(0.2))
        
        #model.add(Bidirectional(LSTM(128,dropout = 0.5,return_sequences=True)))
        model.add(Bidirectional(LSTM(128,dropout = 0.5)))  


    
        model.add(Dense(3, activation = 'softmax'))
        model.summary()
        return model
    
    def model_train(self,X_train, y_train,model,validation_split,batch_size,epochs,epochOfModel):
        '''
        Train BiLSTM model
        param: X_train: text train data
        param: y_train: sentiment label train data
        param: model
        param: validation_split: proportion to split valid set
        param: batch_size: the amount of data to train together
        param: epochs: times of iteration
        param: epochOfModel: choose which epoch of model to save
        return；model
        '''
        model.compile(optimizer='adam', loss='categorical_crossentropy', 
                      metrics = ['Recall','Accuracy','Precision',tf.keras.metrics.AUC(from_logits=True,name='auc')])
        
        history = model.fit(X_train, y_train, validation_split=validation_split, epochs = epochs, batch_size = batch_size )
        
        model.save('./A/taskA.h5'.format(epochOfModel))
        
        train_acc = history.history['Accuracy'][epochOfModel]
        val_acc = history.history['val_Accuracy'][epochOfModel]
        
        train_rec = history.history['recall'][epochOfModel]
        val_rec = history.history['val_recall'][epochOfModel]
        
        train_pre = history.history['precision'][epochOfModel]
        val_pre = history.history['val_precision'][epochOfModel]
        
        train_los = history.history['loss'][epochOfModel]
        val_los = history.history['val_loss'][epochOfModel]
        
        train_auc = history.history['auc'][epochOfModel]
        val_auc = history.history['val_auc'][epochOfModel]
        return train_acc, val_acc,train_los,val_los, history      
    
    
# Step5: prediction and evaluation
    def pred_eval(self, model,X_test, Y_test):
        '''
        Test BiLSTM model, calculate accuracy of test set
        param: model
        param: X_test: text test data
        param: y_test: sentiment label test data
        return；model
        return: Y_pred,Metric,Confusion_Matrix,report
        '''
        Metric=pd.DataFrame(index=['Y_Predict'])
        Y_pred = model.predict(X_test)
        y_test = np.argmax(Y_test, axis = 1)
        y_pred = np.argmax(Y_pred, axis = 1)
        Metric['Recall'] = recall_score(y_test, y_pred,average='macro')
        Metric['Accuracy'] = accuracy_score(y_test, y_pred)
        Metric['precision'] = precision_score(y_test, y_pred,average='macro')
        Metric['F1_Score'] = f1_score(y_test, y_pred,average='macro')
        Confusion_Matrix=confusion_matrix(y_test, y_pred)
        report=classification_report(y_test,y_pred, target_names=['negative:0','neutral:1','positive:2'])
        return Y_pred,Metric,Confusion_Matrix,report
        
    def Accuracy(self,train_acc, val_acc,Metrics,which):
        '''
        tidy accuracy
        '''
        Accuracy=[train_acc, val_acc,Metrics.loc['Y_Predict','Accuracy']]
        Accuracy=pd.DataFrame(Accuracy).T
        Accuracy.columns=['Training','Validation','Test']
        Accuracy.index=['Accuracy of '+which]
        
        return Accuracy   
        
    def sketch_model(self,history, Confusion_Matrix):
        '''
        draw the lineplot and heatmap
        '''
        Train_acc=history.history['Accuracy']
        Valid_acc=history.history['val_Accuracy']

        Train_rec=history.history['recall']
        Valid_rec=history.history['val_recall']

        Train_pre=history.history['precision']
        Valid_pre=history.history['val_precision']


        Train_los=history.history['loss']
        Valid_los=history.history['val_loss']
        
        Train_auc=history.history['auc']
        Valid_auc=history.history['val_auc']
        
        df=pd.DataFrame([Train_acc,Valid_acc,Train_rec,Valid_rec,Train_pre,Valid_pre,Train_los,Valid_los,Train_auc,Valid_auc],columns=range(1,51),
                index=['Train_acc','Valid_acc','Train_rec','Valid_rec','Train_pre','Valid_pre','Train_los','Valid_los','Train_auc','Valid_auc']).T

        df['Train_F1score']=2*np.array(Train_pre)*np.array(Train_rec)/(np.array(Train_pre)+np.array(Train_rec))
        df['Valid_F1score']=2*np.array(Valid_pre)*np.array(Valid_rec)/(np.array(Valid_pre)+np.array(Valid_rec))

        sns.set_theme(style="whitegrid")
        '''
        Train_acc    Valid_acc 
        Train_rec    Valid_rec
        Train_pre    Valid_pre
        Train_los    Valid_los
        '''
        Acc=sns.lineplot(data=df[['Train_acc','Valid_acc']], palette="tab10", linewidth=2.5)
        Acc.set(xlabel = 'Epochs', ylabel = 'Accuracy')
        fig = Acc.get_figure()
        fig.savefig("./Figure/A/Accuracy.png")
        plt.close()
        
        Rec=sns.lineplot(data=df[['Train_rec','Valid_rec']], palette="tab10", linewidth=2.5)
        Rec.set(xlabel = 'Epochs', ylabel = 'Recall')
        fig = Rec.get_figure()
        fig.savefig("./Figure/A/Recall.png")
        plt.close()
        
        
        Pre=sns.lineplot(data=df[['Train_pre','Valid_pre']], palette="tab10", linewidth=2.5)
        Pre.set(xlabel = 'Epochs', ylabel = 'Precision')
        fig = Pre.get_figure()
        fig.savefig("./Figure/A/Precision.png")
        plt.close()
        
        F1=sns.lineplot(data=df[['Train_F1score','Valid_F1score']], palette="tab10", linewidth=2.5)
        F1.set(xlabel = 'Epochs', ylabel = 'F1score')
        fig = F1.get_figure()
        fig.savefig("./Figure/A/F1score.png")
        plt.close()
        
        los=sns.lineplot(data=df[['Train_los','Valid_los']], palette="tab10", linewidth=2.5)
        los.set(xlabel = 'Epochs', ylabel = 'Loss')
        fig = los.get_figure()
        fig.savefig("./Figure/A/loss.png")
        plt.close()
        
        auc=sns.lineplot(data=df[['Train_auc','Valid_auc']], palette="tab10", linewidth=2.5)
        auc.set(xlabel = 'Epochs', ylabel = 'ROC')
        fig = auc.get_figure()
        fig.savefig("./Figure/A/ROC.png")
        plt.close()        
        
        lea=sns.lineplot(data=df[['Train_acc','Valid_acc','Train_los','Valid_los']], palette="tab10", linewidth=2.5)
        lea.set(xlabel = 'Epochs', ylabel = 'Learning')
        fig = lea.get_figure()
        fig.savefig("./Figure/A/learning.png")
        plt.close()      
        
        CM=sns.heatmap(Confusion_Matrix, annot=True,center=True, cmap="mako",xticklabels=['negative','neutral','positive'],yticklabels=['negative','neutral','positive'])
        CM.set_xlabel('Predict')
        CM.set_ylabel('True')
        fig = CM.get_figure()
        fig.savefig("./Figure/A/Confusion_Matrix.png")
        plt.close()
        del fig

        return 