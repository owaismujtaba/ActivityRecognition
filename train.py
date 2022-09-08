from  utils import load_processed_data
from sklearn.model_selection import train_test_split
from model import cnn_model
from sklearn.preprocessing import OneHotEncoder
import pdb
import os
from vis_utils import plot_train_accuracy_loss 


def trainer():
    x , y = load_processed_data()
    #x=x[:100]
    #y = y[:100]
    #pdb.set_trace()
    #x = x.todense()
    #y = y.to_tedense()
    # converting y to 4 column vector
    encoder = OneHotEncoder()
    y = encoder.fit_transform(y.reshape(-1,1))
    y = y.todense()
    #pdb.set_trace()
    x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.33, shuffle=True)
    
    model = cnn_model()
    
    model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ["accuracy"]) 
    
    
    model_training_history = model.fit(x = x_train, y = y_train, epochs = 2, batch_size = 8 , shuffle = True)
    
    model.save(os.getcwd()+"/Models/cnnmodel.h5")
    plot_train_accuracy_loss(model_training_history)
    #pdb.set_trace()

    
    