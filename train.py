from  utils import load_processed_data
from sklearn.model_selection import train_test_split
from model import cnn_model
from sklearn.preprocessing import OneHotEncoder
import pdb


def trainer():
    x , y = load_processed_data()
    
    # converting y to 4 column vector
    encoder = OneHotEncoder()
    y = encoder.fit_transform(y.reshape(-1,1))
    
    #pdb.set_trace()
    x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.33, shuffle=True)
    
    model = cnn_model()
    
    model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ["accuracy"])
    #pdb.set_trace()
    
    
    
    
    
    
    model_training_history = model.fit(x = x_train, y = y_train, epochs = 50, batch_size = 8 , shuffle = True)
    
    
    
    
    