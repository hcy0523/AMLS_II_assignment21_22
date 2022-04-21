import A.SubtaskA as A
import B.SubtaskB as B

ClassA=A.A()
ClassB=B.B()

# ======================================================================================================================
# Task A
address='Datasets/A/twitter-2016A.txt'
window=5 
min_count=1 
Word_epochs=100 
maxlen=64 
Train_epoch=50 
batch_size=128
test_size=0.1
validation_split=0.2
epochOfModel=27 # save model in 27th epoch.

# Data preprocessing
X_train_A, X_test_A, Y_train_A, Y_test_A,vocab,embed_dict = ClassA.Final_PreProcess(address,window,min_count,Word_epochs,maxlen,test_size)
# Build model object.
model_A = ClassA.model_build(vocab,embed_dict,maxlen)
# Train model based on the training set
train_acc_A, val_acc_A,train_los_A,val_los_A, history_A = ClassA.model_train(X_train_A, Y_train_A,model_A,validation_split,batch_size,Train_epoch,epochOfModel)
# Test model based on the test set.
Y_pred_A,Metrics_A,Confusion_Matrix_A,report_A = ClassA.pred_eval(model_A,X_test_A, Y_test_A)

#ClassA.sketch_model(history_A,Confusion_Matrix_A)
Accuracy_A=ClassA.Accuracy(train_acc_A, val_acc_A,Metrics_A,'A')

test_acc_A=Metrics_A.loc['Y_Predict','Accuracy']
# ======================================================================================================================
# Task B
address='Datasets/B/twitter-2016B.txt'
window=5 
min_count=1 
Word_epochs=100 
maxlen=100 
Train_epoch=50 
batch_size=128
test_size=0.1
validation_split=0.2
epochOfModel=23 # save model in 23th epoch.
# Data preprocessing
X_train_B, X_test_B, Y_train_B, Y_test_B,vocab,embed_dict = ClassB.Final_PreProcess(address,window,min_count,Word_epochs,maxlen,test_size)
# Build model object.
model_B = ClassB.model_build(vocab,embed_dict,maxlen)
# Train model based on the training set
train_acc_B, val_acc_B,train_los_B,val_los_B, history_B = ClassB.model_train(X_train_B, Y_train_B,model_B,validation_split,batch_size,Train_epoch,epochOfModel)

Y_pred_B,Metrics_B,Confusion_Matrix_B,report_B = ClassB.pred_eval(model_B,X_test_B, Y_test_B)

#ClassB.sketch_model(history_B,Confusion_Matrix_B)

Accuracy_B=ClassB.Accuracy(train_acc_B, val_acc_B,Metrics_B,'B')
test_acc_B=Metrics_B.loc['Y_Predict','Accuracy']
# ======================================================================================================================
## Print out your results with following format:
print('TA:{},{};TB:{},{};'.format(train_acc_A, test_acc_A,
                                                        train_acc_B, test_acc_B))