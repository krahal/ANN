# Artificial Neural Network (classification problem)

# Installing Theano (numerical computations)
# Installing Tensorflow (numerical computations)
# Installing Keras (wraps Theano and Tensorflow -> few lines of code) 

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np # most of mathematics and to work with arrays
import matplotlib.pyplot as plt # plotting graphs
import pandas as pd # import and manage datasets

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values # matrix of features
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] # avoid dummy variable trap (take 1 less)

# Splitting the data into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling (so some terms don't dominate)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part 2 - Make the ANN

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential # initialize the ANN
from keras.layers import Dense # build layers of ANN (will also initialize weights)
from keras.layers import Dropout

# Initializing the ANN (sequence of layers)
classifier = Sequential() # this is a classification problem

# Adding the input layer and the first hidden layer (recitifier activation fn for hidden layers) with dropout
# How many nodes in this layer? easy choice: avg of input + output nodes
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(rate = 0.1)) # usually start with rate = 0.1

# Adding the second hidden layer with dropout
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.1))

# Adding the output layer (sigmoid activation fn -> get probabilites for binary outcomes)
# If dv with >2 categories, change units and activation = 'softmax'
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN (applying stochastic gradient descent to whole ANN)
# optimizer: what algo you wanna use to find optimal weights -> SGD
# loss: loss fn within adam SGD that you wanna optimize // categorical_crossentropy for >2 dv categories
# metrics: criterion used to evaluate model -> after weights updates, used to improve model performance
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
# No rule of thumb for batch_size and epochs
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test) # vector that gives prediction of each test set observation
y_pred = (y_pred > 0.5) # change predictions to 1 or 0 using threshold

# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
# numpy horizontal vector // need to use dummy variable encodings // need feature scaling for this prediction
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

# Making the Confusion Matrix (evaluate our model)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Calculating Model Accuracy
from sklearn.metrics import accuracy_score
acs = accuracy_score(y_test, y_pred)
print("\nAccuracy Score: %.2f%%" % (acs * 100))

# Save the model
classifier.save('models/bankChurnPrediction1.h5')

# Save the scaler
import pickle
scalerfile = 'scaler.sav'
pickle.dump(sc, open(scalerfile, 'wb'))

# Part 4 - Evaluating, Improving, and Tuning the ANN

# Evaluating the ANN (bias and variance)
from keras.wrappers.scikit_learn import KerasClassifier # wrapper for k-fold into keras model
from sklearn.model_selection import cross_val_score
from keras.models import Sequential 
from keras.layers import Dense 
def build_classifier():
    classifier = Sequential() 
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100) # k-fold cross validation classifier
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
# ANN will be trained 10 times for each fold and 10 iterations -> 10 accuracies returned
mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN
# Dropout Regularization to reduce overfitting if needed

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV # will return best combo of params for accuracy
from keras.models import Sequential 
from keras.layers import Dense 
def build_classifier(optimizer):
    classifier = Sequential() 
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32], 
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters, 
                           scoring = 'accuracy',
                           cv = 10) # implements k-fold cross validation
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_