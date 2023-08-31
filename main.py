from mnist import MNIST
import numpy as np;
def initWeight():
    w = [np.empty((784,16),dtype=np.float64),np.empty((16,16),dtype=np.float64),np.empty((16,10),dtype=np.float64)]
    for i in range(0,len(w)):
        for j in range(0,w[i].shape[0]):
            for k in range(0,w[i].shape[1]):
                w[i][j][k] = np.random.uniform(-2.4/784,2.4/784);
    return w;

def sigmoid(x):
    try:
     return 1/(1 + np.exp(-x));
    except:
        print(x);

def fowardRun(layer,w):
    sum = 0;
    for i in range(0, len(layer)):
        sum += layer[i] * w[i];
    return sum;

def errorOuput(expected, predicted):
    error = np.empty(expected.shape,dtype=np.float64);
    for i in range(0,len(expected)):
        error[i] = (expected[i] - predicted[i]);
    return error;

def errorLayer(currLayer, error_af, w):
    error = np.empty(currLayer.shape,dtype=np.float64);
    for i in range(0, currLayer.shape[0]):
        error_gradient_single = 0;
        for j in range(0, len(error_af)):
            error_gradient_single += w[i][j] * error_af[j];
        error[i] = error_gradient_single * currLayer[i] * (1-currLayer[i]);
    return error;

def updateW(w, error, alpha, layer):
    newW = np.empty(shape=w.shape,dtype=np.float64);
    for i in range(0, len(w)):
        for j in range(0, len(w[i])):
            newW[i][j] = w[i][j] + alpha * error[j] * layer[i];
    return newW;

def softmax(z):
    sum = 0;
    result = np.empty(z.shape,dtype=np.float64);
    for i in range(0,len(z)):
        sum += np.exp(z[i]);
    for i in range(0,z.shape[0]):
        result[i] = np.exp(z[i])/sum;
    return result;

def normalize(arr):
    result = np.array(arr,dtype=np.float64);
    for i in range(0,len(arr)):
        result[i] = result[i]/255.0;
    return result;
def train():
    mndata = MNIST('samples');
    np.random.seed();
    counter = 0;
    train_images, train_labels = mndata.load_training();
    test_images, test_labels = mndata.load_testing();
    train_images = np.array(train_images);
    train_labels = np.array(train_labels);
    test_images = np.array(test_images);
    test_labels = np.array(test_labels);
    w = initWeight();
    alpha = 0.05
    iteration = 0;
    size = 0;
    while(iteration < 5 ):
        for i in range(0, len(train_images)):
            layer_input = train_images[i].flatten();
            layer_input = normalize(layer_input);
            layer_1 = layer_input @ w[0];
            for j in range(0, layer_1.shape[0]):
                layer_1[j] = sigmoid(layer_1[j]);
            layer_2 = layer_1 @ w[1];
            for j in range(0, layer_2.shape[0]):
                layer_2[j] = sigmoid(layer_2[j]);
            layer_output = layer_2 @ w[2];
            layer_output = softmax(layer_output);
            expected = np.zeros((10),dtype=np.float64);
            expected[train_labels[i]] = 1;
            max_num = 0;
            for j in range(0,len(layer_output)):
                if(layer_output[max_num] < layer_output[j]):
                    max_num = j;
            error_output = np.empty(expected.shape,dtype=np.float64);
            for j in range(0,error_output.shape[0]):
                error_output[j] = expected[j] - layer_output[j]; 
            error_2 = errorLayer(layer_2,error_output,w[2]);
            error_1 = errorLayer(layer_1,error_2,w[1]);
            w2 = updateW(w[2],error_output,alpha,layer_2);
            w1 = updateW(w[1],error_2,alpha,layer_1);
            w0 = updateW(w[0],error_1,alpha,layer_input);
            w = [w0,w1,w2];
            if(np.fabs(train_labels[i] - max_num) == 0):
                counter +=1;
            print(str(size) + ' : ' + str(counter/(size+1)));
            size += 1;
        iteration += 1;
    with open("filename.txt", "w") as file:
        for i in range(0,len(w)):
            for j in range(0,len(w[i])):
                for k in w[i][j]:
                    file.write(str(k) + '\n');
     

    

#train();



def test(w):
    mndata = MNIST('samples');
    np.random.seed();
    counter = 0;
    test_images, test_labels = mndata.load_testing();
    test_images = np.array(test_images);
    test_labels = np.array(test_labels);
    size = 0;
    for i in range(0, len(test_images)):
        layer_input = test_images[i].flatten();
        layer_input = normalize(layer_input);
        layer_1 = layer_input @ w[0];
        for j in range(0, layer_1.shape[0]):
            layer_1[j] = sigmoid(layer_1[j]);
        layer_2 = layer_1 @ w[1];
        for j in range(0, layer_2.shape[0]):
            layer_2[j] = sigmoid(layer_2[j]);
        layer_output = layer_2 @ w[2];
        layer_output = softmax(layer_output);
        max_num = 0;
        for j in range(0,len(layer_output)):
            if(layer_output[max_num] < layer_output[j]):
                max_num = j;
        if(np.fabs(test_labels[i] - max_num) == 0):
            counter +=1;
        print(str(size) + ' : ' + str(counter/(size+1)));
        size += 1;


w = [np.empty((784,16),dtype=np.float64),np.empty((16,16),dtype=np.float64),np.empty((16,10),dtype=np.float64)];

with open('filename.txt','r') as f:
    for i in range(0,len(w)):
        for j in range(0,len(w[i])):
            for k in range(0,len(w[i][j])):
                w[i][j][k] = np.float64(f.readline());

test(w);