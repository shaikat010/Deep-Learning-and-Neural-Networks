import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np

(X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()

# print(len(X_train))
# print(len(y_test))
# print(len(y_train))
#
# print(X_train[0].shape)
# print(X_train[0])
#
# #the first image
# plt.matshow(X_train[0])
# plt.show()
#
# print(y_train[2])


#scaling the X_train and test datasets, accuracy improves after this
X_train = X_train/255
X_test  =X_test/255

print(X_train.shape)

#flatten out training dataset
X_train_flattened = X_train.reshape(len(X_train),28*28)
print(X_train_flattened.shape)

X_test_flattened = X_test.reshape(len(X_test),28*28)
print(X_test_flattened.shape)

print(X_test_flattened[0])

model = keras.Sequential([keras.layers.Dense(10,input_shape=(784,),activation='sigmoid')])

model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(X_train_flattened,y_train,epochs=5)

print(model.evaluate(X_test_flattened,y_test))
#accuracy comes more than 92 percent.

# plt.matshow(X_test[0])
# plt.show()
# the first imahe is 7

y_predicted = model.predict(X_test_flattened)
print(y_predicted[0])

print(np.argmax(y_predicted[0]))
#the value mathces and it is 7 here too, prefect!


y_predicted_labels = [np.argmax(i) for i in y_predicted]

#now we build a confusion matrix
cm  = tf.math.confusion_matrix(labels = y_test,predictions=y_predicted_labels)
print(cm)

import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt = 'd')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()


#copy and pasting the same model but adding a hidden layer here!
model = keras.Sequential([
    # hidden layer added here!
    keras.layers.Dense(100,input_shape=(784,),activation='relu'),
    keras.layers.Dense(10 , activation='sigmoid')

])

model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(X_train_flattened,y_train,epochs=5)
# accurpacy increase much more here after adding the hidden layer
print(model.evaluate(X_test_flattened,y_test))

#again build the confusion matrix and then check the truth values will improve!

cm  = tf.math.confusion_matrix(labels = y_test,predictions=y_predicted_labels)
import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt = 'd')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

#keras has a special layer called flatten, so we will now model without flattening the arrays
# this makes it very simple
model = keras.Sequential([
    keras.layers.flatten(input_shape = (28,28)),
    # hidden layer added here!
    keras.layers.Dense(100,activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')

])

model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(X_train_flattened,y_train,epochs=5)

#there are different optimizers and loss funciton and metrics and epochs adn activation function in tensorflow
