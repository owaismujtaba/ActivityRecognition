from  utils import load_processed_data
from sklearn.model_selection import train_test_split
from model import cnn_model
from sklearn.preprocessing import OneHotEncoder
import pdb
import os
import numpy as np
from vis_utils import plot_train_accuracy_loss 


def trainer():
    
    x , y = load_processed_data()
    # converting y to 4 column vector
    encoder = OneHotEncoder()
    y = encoder.fit_transform(y.reshape(-1,1))
    y = y.todense()
    #pdb.set_trace()
    x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.33, shuffle=True)
    
    np.save(os.getcwd()+'/Data/x_test.npy', x_test)
    np.save(os.getcwd()+'/Data/y_test.npy', y_test)
    
    
    model = cnn_model()
    
    model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ["accuracy"]) 
    
    
    model_training_history = model.fit(x = x_train, y = y_train, validation_data=(x_test, y_test), epochs = 50, batch_size = 256 , shuffle = True)
    
    model.save(os.getcwd()+"/Models/cnnmodel.h5")
    plot_train_accuracy_loss(model_training_history)
    #pdb.set_trace()

    
    