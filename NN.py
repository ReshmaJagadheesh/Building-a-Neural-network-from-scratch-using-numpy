import numpy as np
import h5py  
import copy 
from random import randint

#load MNIST data 
MNIST_data = h5py.File('MNISTdata.hdf5', 'r') 
x_train = np.float32(MNIST_data['x_train'][:]) 
y_train = np.int32(np.array(MNIST_data['y_train'][:,0])) 
x_test = np.float32(MNIST_data['x_test'][:]) 
y_test = np.int32(np.array(MNIST_data['y_test'][:,0]))
MNIST_data.close()

#number of inputs 
num_inputs = 28*28
#number of outputs 
num_outputs = 10
#number of hidden units
num_hidden = 120

model = {} 
model["W1"] = np.random.randn(num_hidden,num_inputs) / np.sqrt(num_inputs)
model["b1"] = np.zeros((num_hidden,1))
model["W2"] = np.random.randn(num_outputs,num_hidden) / np.sqrt(num_hidden)
model["b2"] = np.zeros((num_outputs,1))

model_grads = copy.deepcopy(model)

def softmax_function(z): 
    ZZ = np.exp(z)/np.sum(np.exp(z),axis=0) 
    return ZZ

def relu(z):
    return z*(z>0)

def drelu(z):
    return 1*(z>0)

def forward_propagation(x,model):
    x = np.reshape(x,(-1,1))
    Z1 = np.dot(model["W1"],x)+model["b1"]
    A1 = relu(Z1)
    Z2 = np.dot(model["W2"],A1)+model["b2"]
    A2 = softmax_function(Z2)
    
    cache = {"Z1":Z1,"A1":A1,"Z2":Z2,"A2":A2}
    return A2,cache

def backward_propagation(x,y,A2, model, model_grads,cache):
    y_t = np.zeros((1,10))
    y_t[np.arange(1), y] = 1
    y_t = np.reshape(y_t,(-1,1))
    
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    W2 = model["W2"]
    
    delta1 = y_t-A2
    db2 = np.sum(delta1,axis=1,keepdims=True)
    dW2 = np.dot(delta1,A1.T)
    
    delta2 = np.dot(W2.T,delta1)
    db1 = np.sum(np.multiply(delta2,drelu(Z1)),axis=1,keepdims=True)   
    
    delta3 = np.multiply(delta2,drelu(Z1))
    x_T = np.reshape(x,(-1,1))
    dW1 = np.dot(delta3,x_T.T)
    
    
    model_grads["W1"] = dW1
    model_grads["b1"] = db1
    model_grads["W2"] = dW2
    model_grads["b2"] = db2
    
    return model_grads

learning_rate = 0.01
num_epochs = 20

for epochs in range(num_epochs):
    if (epochs > 5): 
        learning_rate = 0.001 
    if (epochs > 10):
        learning_rate = 0.0001 
    if (epochs > 15): 
        learning_rate = 0.00001
        
    total_correct = 0 
    
    for n in range(len(x_train)): 
        n_random = randint(0,len(x_train)-1) 
        y = y_train[n_random] 
        x = x_train[n_random][:] 
        A2,cache = forward_propagation(x, model) 
        prediction = np.argmax(A2) 
        if (prediction == y): 
            total_correct += 1 
        model_grads = backward_propagation(x,y,A2, model, model_grads,cache) 
        model["W1"] = model["W1"] + learning_rate*model_grads["W1"] 
        model["b1"] = model["b1"] + learning_rate*model_grads["b1"]
        model["W2"] = model["W2"] + learning_rate*model_grads["W2"]
        model["b2"] = model["b2"] + learning_rate*model_grads["b2"]
    print(total_correct/np.float(len(x_train)))

#test data 
total_correct = 0 
for n in range(len(x_test)): 
    y = y_test[n] 
    x = x_test[n][:] 
    x = np.reshape(x,(1,-1))
    A2,cache = forward_propagation(x, model) 
    prediction = np.argmax(A2) 
    if (prediction == y): 
        total_correct += 1

print(total_correct/np.float(len(x_test)))
