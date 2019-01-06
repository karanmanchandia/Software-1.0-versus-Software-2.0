
# coding: utf-8

# In[314]:


# standard imports
import numpy as np
import tensorflow as tf
import tqdm
from tqdm import tqdm_notebook
import pandas as pd
from keras.utils import np_utils
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Logic Based FizzBuzz Function [Software 1.0]

# In[315]:


def fizzbuzz(n):
    
    # Logic Explanation
    # here n is the input integer. If n is divisible by 3 and also divisible by 5, the the intiger should fall in the fizzbuzz bucket, If n is divisible by 3 then integer should fall in the fizz bucket, If n is divisible by 5 then integer should fall in the buzz bucket,
    # if no above condition is satisfied then integer should fall in the other bucket.
    if n % 3 == 0 and n % 5 == 0:
        return 'FizzBuzz'
    elif n % 3 == 0:
        return 'Fizz'
    elif n % 5 == 0:
        return 'Buzz'
    else:
        return 'Other'


# ## Create Training and Testing Datasets in CSV Format

# In[316]:


def createInputCSV(start,end,filename):
    
    # Why list in Python?
    # List is a collection in python which is ordered and changeable and it allows duplicate members. This is also the reason why list are required in python.
    # It can be thought that tuples in python can be used in place of List, but there is no append(), remove() etc. method for tuples.
    inputData   = []
    outputData  = []
    
    # Why do we need training Data?
    # Initially, we start with randomly initialized weights in our model. These randomly initialized weights will not give us a accurate result. So, we need to adjust the weights to get the most accurate result.
    # For adjusting the weights we need to train the neural network with the training data set. That is why we require the training data set.
    # The quality of the training dat set matters a lot. Consider an example, A face detection system shuld be able to correctly recognize a human and a alien, that is to land on Earth in 2050. The correctness of the system in these type of conditions depends on the quality of training data set.  
    for i in range(start,end):
        inputData.append(i)
        outputData.append(fizzbuzz(i))
    
    # Why Dataframe?
    # python has been great for data preparation, but not so great for data analysis, modelling. Pandas library provide python with this power.
    # Pandas has a fast and efficient dataframe object for data manupulation and integrated indexing.
    dataset = {}
    dataset["input"]  = inputData
    dataset["label"] = outputData
    
    # Writing to csv
    pd.DataFrame(dataset).to_csv(filename)
    
    print(filename, "Created!")


# ## Processing Input and Label Data

# In[317]:


def processData(dataset):
    
    # Why do we have to process?
    # We have to process to encode Fizz, Buzz, Fizzbuzz and other as 1,2,3 and 0.
    data   = dataset['input'].values
    labels = dataset['label'].values
    
    processedData  = encodeData(data)
    processedLabel = encodeLabel(labels)
    
    return processedData, processedLabel


# In[318]:


import numpy as np

def encodeData(data):
    
    processedData = []
    
    for dataInstance in data:
        
        # Why do we have number 10?
        # Inputs to our neural Network model will be integers from 1-1000. As per decimal to binary conversion, for taking 1000 integers as input we require input with 10 binary bits.
        # Also number of input neurons in our model is 10.
        processedData.append([dataInstance >> d & 1 for d in range(10)])
    
    return np.array(processedData)


# In[319]:


def encodeLabel(labels):
    
    processedLabel = []
    
    for labelInstance in labels:
        if(labelInstance == "FizzBuzz"):
            # Fizzbuzz
            processedLabel.append([3])
        elif(labelInstance == "Fizz"):
            # Fizz
            processedLabel.append([1])
        elif(labelInstance == "Buzz"):
            # Buzz
            processedLabel.append([2])
        else:
            # Other
            processedLabel.append([0])

    return np_utils.to_categorical(np.array(processedLabel),4)


# In[320]:


# Create datafiles
# Here we are creating training and testing data files.
createInputCSV(101,1001,'training.csv')
createInputCSV(1,101,'testing.csv')


# In[321]:


# Read Dataset
trainingData = pd.read_csv('training.csv')
testingData  = pd.read_csv('testing.csv')

# Process Dataset
processedTrainingData, processedTrainingLabel = processData(trainingData)
processedTestingData, processedTestingLabel   = processData(testingData)


# ## Tensorflow Model Definition

# In[322]:


# Defining Placeholder
# Inputs: training and the testing file 
# Output_buckets: fizz,buzz,fizzbuzz,other
# We have to specify the input dimension as 10 and output dimension as 4 as there are ten input neurons and 4 output buckets.
inputTensor  = tf.placeholder(tf.float32, [None, 10])
outputTensor = tf.placeholder(tf.float32, [None, 4])


# In[323]:


NUM_HIDDEN_NEURONS_LAYER_1 = 100
LEARNING_RATE = 0.05

# Initializing the weights to Normal Distribution
# The weights should be initialized to random values
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01))

# Initializing the input to hidden layer weights
# Here, the number of input hidden weights depend on the number of neurons in first layer (10) and the number of neurons in hidden neuron layer 1
input_hidden_weights  = init_weights([10, NUM_HIDDEN_NEURONS_LAYER_1])
# Initializing the hidden to output layer weights
#The number of output hidden weights depend on the number of neurons in hidden neuron layer 1 and the number of neurons in the output layer (4)
hidden_output_weights = init_weights([NUM_HIDDEN_NEURONS_LAYER_1, 4])

# Computing values at the hidden layer
# Activation of input and the hidden layers
# Rectified Linear Unit Activation function is used
hidden_layer = tf.nn.relu(tf.matmul(inputTensor, input_hidden_weights))
# Computing values at the output layer
output_layer = tf.matmul(hidden_layer, hidden_output_weights)

# Defining Error Function
# The error function at the output layer checks the predicted value and compares it with the true value, subtracts these values, and finally the error value is stored in the error function
error_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=outputTensor))

# Defining Learning Algorithm and Training Parameters
# Here we are minimizing the error function and the optimizer is gradient decent optimizer
training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(error_function)

# Prediction Function
prediction = tf.argmax(output_layer, 1)


# # Training the Model

# In[324]:


NUM_OF_EPOCHS = 5000             # Epoch is one iteration of all the training exmaples 
BATCH_SIZE = 128               # Batch is a subset of the training data
                             # We need batches because at the end of each batch weights are updated
training_accuracy = []

with tf.Session() as sess:
    
    # Set Global Variables ?
    # All the variables inside the session are global variables
    tf.global_variables_initializer().run()
    
    for epoch in tqdm_notebook(range(NUM_OF_EPOCHS)):
        
        #Shuffle the Training Dataset at each epoch
        # After every epoch we shuffle the data so as to have a general pattern in the model
        p = np.random.permutation(range(len(processedTrainingData)))
        processedTrainingData  = processedTrainingData[p]
        processedTrainingLabel = processedTrainingLabel[p]
        
        # Start batch training
        for start in range(0, len(processedTrainingData), BATCH_SIZE):
            end = start + BATCH_SIZE
            sess.run(training, feed_dict={inputTensor: processedTrainingData[start:end], 
                                          outputTensor: processedTrainingLabel[start:end]})
        # Training accuracy for an epoch
        # after every epoch we find the accuracy
        training_accuracy.append(np.mean(np.argmax(processedTrainingLabel, axis=1) ==
                             sess.run(prediction, feed_dict={inputTensor: processedTrainingData,
                                                             outputTensor: processedTrainingLabel})))
    # Testing
    predictedTestLabel = sess.run(prediction, feed_dict={inputTensor: processedTestingData})


# In[325]:


df = pd.DataFrame()
df['acc'] = training_accuracy
df.plot(grid=True)


# In[326]:


def decodeLabel(encodedLabel):
    if encodedLabel == 0:
        return "Other"
    elif encodedLabel == 1:
        return "Fizz"
    elif encodedLabel == 2:
        return "Buzz"
    elif encodedLabel == 3:
        return "FizzBuzz"


# # Testing the Model [Software 2.0]

# In[327]:


wrong   = 0
right   = 0

predictedTestLabelList = []
""
for i,j in zip(processedTestingLabel,predictedTestLabel):
    predictedTestLabelList.append(decodeLabel(j))
    
    if np.argmax(i) == j:
        right = right + 1
    else:
        wrong = wrong + 1

print("Errors: " + str(wrong), " Correct :" + str(right))

print("Testing Accuracy: " + str(right/(right+wrong)*100))

# Please input your UBID and personNumber 
testDataInput = testingData['input'].tolist()
testDataLabel = testingData['label'].tolist()

testDataInput.insert(0, "UBID")
testDataLabel.insert(0, "karanman")

testDataInput.insert(1, "personNumber")
testDataLabel.insert(1, "50290755")

predictedTestLabelList.insert(0, "")
predictedTestLabelList.insert(1, "")

output = {}
output["input"] = testDataInput
output["label"] = testDataLabel

output["predicted_label"] = predictedTestLabelList

opdf = pd.DataFrame(output)
opdf.to_csv('output.csv')

