import pandas as pd
import numpy as np
np.random.seed(2)
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential ,Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt

# Load the data
train = pd.read_csv("C:/Users/risha/Downloads/train.csv")
test = pd.read_csv("C:/Users/risha/Downloads/test.csv")

Y_train = train["label"]    #store result in answer

# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1)

del train

#normalization
X_train = X_train / 255.0
test = test / 255.0

# Reshape image in 3 dimensions
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

# Encode labels to one hot vectors
Y_train = to_categorical(Y_train, num_classes = 10)

random_seed = 2

# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)

# Set the CNN model

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (4,4),padding = 'Same',
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (4,4),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


#neural network layer
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

optimizer = RMSprop(lr=0.001)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

epochs = 15
batch_size = 86

# Fit the model
history = model.fit(X_train,Y_train, batch_size=batch_size,
                    epochs = epochs, validation_data = (X_val,Y_val),
                    verbose = 2)

Y_pred = model.predict(test)
Y_pred_classes = np.argmax(Y_pred,axis = 1)

ids = [i for i in range(1,len(Y_pred_classes)+1)]

output = pd.DataFrame({ 'ImageId' : ids, 'Label': Y_pred_classes })
output.to_csv('predictions.csv', index = False)
print(output.head())

