# AMLS_II_assignment21_22

#### This project contains subtask A and subtask B of   [SemEval-2017 Task 4](https://alt.qcri.org/semeval2017/task4/#) 

All the data are collected from Twitter based on different topics.

**Subtask A:** Message Polarity Classification: Given a message, classify whether the message is of positive, negative, or neutral sentiment

Data for subtask A: it contains 20632 pieces of text data and each data includes 3 parts, which is id of each piece of data, sentiment that the text containing and the content of text.



**Subtask B**: Topic-Based Message Polarity Classification: Given a message and a topic, classify the message on two-point scale: positive or negative sentiment towards that topic

Data for subtask B: it contains 10551 pieces of text data and each data includes 4 parts, which is id of each piece of data, the topic of the text, sentiment that the text containing and the content of text. There are 100 unique topics in total.



**Structure:**

- A: Folder-contain the code file for subtaskA

  - SubtaskA.py : contain all the def function about subtaskA

    

- B: Folder-contain the code file for subtaskB

  

  - SubtaskB.py: contain all the def function about subtaskB

- Datasets

  - A: Folder
    - twitter-2016A.txt: text data, split it for train,valid and test set of subtaskA 
  - B: Folder
    - twitter-2016B.txt: text data, split it for train,valid and test set of subtaskB 

- Figure: folder to save the figures,default to comment the function. Enable to produce figures, please uncomment the ClassA/B.sketch_model ()

- main.py: main python file to run the programme by calling the def function of SubtaskA.py  and SubtaskB.py 

- main.ipynb: notebook version of main.py.

- requirements.txt: contains the name of the dependencies need to run the code 

  

This project running under python 3.8.13:

| External libraries | Version |
| ------------------ | ------- |
| numpy              | 1.21.5  |
| pandas             | 1.4.2   |
| keras              | 2.6.0   |
| tensorflow         | 2.6.0   |
| ekphrasis          | 0.5.1   |
| scikit-learn       | 1.0.2   |
| gensim             | 4.1.2   |
| seaborn            | 0.11.2  |
| matplotlib         | 3.5.1   |

