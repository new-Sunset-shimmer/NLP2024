import numpy as np
import re
from tqdm import tqdm
import pickle
import argparse
from copy import deepcopy

class layer:
    def __init__(self,onehot = None ,dict = None):
        self.dict = dict
        self.onehot = onehot
    def linear(self,x,y):
        return x@y
    def forward(self):
        pass
    def backward(self):
        pass
    def _get_onehot(self,x):
        temp = self.onehot[x].T
        return temp

class embedding(layer):
    def __init__(self,onehot,dict):
        super().__init__(onehot,dict)
    def forward(self,x,E): 
        x = self._get_onehot(x)
        y = self.linear(E,x)
        return y, x
    def backward(self, x):
        return x
class hiddenstate(layer):
    def forward(self,x,h,U,W):
        temp_1 = self.linear(U,x)
        temp_2 = self.linear(W,h)
        return temp_1 + temp_2, (x, h, U,  W)
    def backward(self, x):
        x, h, U, W = x 
        dU = x
        dW = h
        dx = U
        dh = W
        return dU,dW,dx,dh
class relu(layer):
    def forward(self,x):
        mask = x < 0
        x[mask] = 0
        return x, mask
    def backward(self,x, mask):
        x[mask] = 0
        return x
class output_layer(layer):
    def __init__(self,onehot,dict):
        super().__init__(onehot,dict)
    def softmax(self,x,V): 
        self.x = x
        self.V = V
        x = self.linear(V,x)
        self.o_gamma = x
        exp_x = cp.exp(x - cp.max(x))
        exp_x_sum = cp.sum(exp_x,-1).reshape(-1,1) + 1e-12
        o = exp_x / exp_x_sum
        return o
    def forward(self,x,V,y):
        y = self._get_onehot(y)
        o = self.softmax(x,V)
        self.y = y
        return self.linear(-y.T,cp.log(o+1e-12)).sum(), (self.o_gamma, self.y, self.x, self.V)
    def backward(self,x):
        o_gamma,y,x,V = x
        return o_gamma - y, x, V

class RNN:
    def __init__(self,dataset = None,epochs = 10, batchs = 4096, lr = 0.01,dim = 300):
        self.dim = dim
        self.weight = {}
        if dataset != None:
            words, labels, self.sentences = self._word_label(dataset)
            words.add("<UKN>")
            words.add("<END>")
            words.add("<PAD>")
            labels.add("<END>")
            labels.add("<UKN>")
            labels.add("<PAD>")
            self.dictlen = len(words)
            self.dictlen_y = len(labels)
            self.dict = {state: idx for idx, state in enumerate(words)}
            self.dict_y = {state: idx for idx, state in enumerate(labels)}
            self.onehot = cp.eye(self.dictlen,dtype=cp.int8)
            self.onehoy_y = cp.eye(self.dictlen_y,dtype=cp.int8)
            self.weight['U'] = cp.random.randn(self.dim,self.dim,dtype=cp.float32)
            self.weight['V'] = cp.random.randn(self.dictlen_y,self.dim,dtype=cp.float32)
            self.weight['W'] = cp.random.randn(self.dim,self.dim,dtype=cp.float32)
            self.weight['E'] = cp.random.randn(self.dim,self.dictlen,dtype=cp.float32)
            self.embed = embedding(self.onehot,self.dict)
            self.hiddenstate = hiddenstate()
            self.activate = relu()
            self.output = output_layer(self.onehoy_y, self.dict_y)
        self.epochs = epochs
        self.batchs = batchs
        self.lr = lr
    def _word_label(self,dataset):
        tag_parser = r'([^\s/]+)\s*/\s*([^\s/]+)'
        line_parser = r'[\n]'
        tags = np.array(re.findall(tag_parser, dataset), dtype = object)
        sentences = np.array(re.split(line_parser, dataset), dtype = object)
        words = set(tags[:,0])
        labels = set(tags[:,1])
        return words, labels, sentences
    def forward(self,x_ground,y_ground):
        self.tracker = []
        loss = 0
        h = cp.zeros((self.dim,self.batchs))
        for i in range(self.max_len-1):
            x = x_ground[:,i]
            self.layer = []
            x, forbackward= self.embed.forward(x,self.weight['E'])
            self.layer.append(forbackward)
            x, forbackward = self.hiddenstate.forward(x,h,self.weight['U'],self.weight['W'])
            self.layer.append(forbackward)
            x,mask = self.activate.forward(x)
            self.layer.append(mask)
            h = x
            y = y_ground[:,i]
            x,forbackward = self.output.forward(x,self.weight['V'],y)    
            loss += x
            self.layer.append(forbackward)
            self.tracker.append(self.layer)
    def backward(self):
        self.tracker.reverse()
        h = cp.zeros((self.dim,self.batchs))
        temp_V = np.zeros_like(self.weight['V'])
        temp_E = np.zeros_like(self.weight['E'])
        temp_U = np.zeros_like(self.weight['U'])
        temp_W = np.zeros_like(self.weight['W'])
        for step in self.tracker:
            dL,dV,dO = self.output.backward(step[-1])
            V = dL@dV.T
            x = dL.T@dO
            x = x.T + h
            dZ = self.activate.backward(x,step[-2])
            dU,dW,dx,dh = self.hiddenstate.backward(step[-3])
            h = (dZ.T@dh).T
            dx = dZ.T@dx
            dE = self.embed.backward(step[-4])
            temp_V += self._clipping(V)
            temp_E += self._clipping(dx.T@dE.T)
            temp_U += self._clipping(dZ@dU.T)
            temp_W += self._clipping(dZ@dW.T)
        self.weight['E'] -= self.lr*temp_E
        self.weight['U'] -= self.lr*temp_U
        self.weight['W'] -= self.lr*temp_W
        self.weight['V'] -= self.lr*temp_V
        del temp_E
        del temp_U
        del temp_W
        del temp_V
        del x
        del dW
        del dU
    def _clipping(self,x):
        return np.clip(x, -1, 1, out=x)
    def _test_run(self,x_ground,y_ground):
        h = cp.zeros((self.dim,self.batchs))
        valid_count = 0
        total = 0
        for i in tqdm(range(self.max_len-1)):
            x = x_ground[:,i]
            x, forbackward= self.embed.forward(x,self.weight['E'])
            x, forbackward = self.hiddenstate.forward(x,h,self.weight['U'],self.weight['W'])
            x,mask = self.activate.forward(x)
            h = x
            y = y_ground[:,i]
            x = self.output.softmax(x,self.weight['V'])  
            x = x.argmax(0).get()
            y = y.get()
            valid_count += np.sum(x[y == x])
            total += np.sum(y[y > -1])
        return valid_count, total
    def train(self):
        for i in range(self.epochs):
            before_idx = 0
            for sentences_idx in tqdm(range(self.batchs,int(len(self.sentences)/6),self.batchs)):
                sentences = self.sentences[before_idx:sentences_idx]
                if len(sentences) > 0:
                    self.max_len = len(max(sentences, key=len).split())
                    y_list = cp.array(list(map(self._splitter_y, sentences)),dtype=int)
                    x_list = cp.array(list(map(self._splitter_x, sentences)),dtype=int)
                    x = x_list
                    y = y_list
                    before_idx = sentences_idx
                    self.forward(x,y)
                    self.backward()
    def test(self,test_set):
        before_idx = 0
        valid_count = 0
        total = 0
        _,_,self.sentences = self._word_label(test_set)
        for sentences_idx in tqdm(range(self.batchs,int(len(self.sentences)),self.batchs)):
            sentences = self.sentences[before_idx:sentences_idx]
            self.max_len = len(max(sentences, key=len).split())
            y_list = cp.array(list(map(self._splitter_y, sentences)),dtype=int)
            x_list = cp.array(list(map(self._splitter_x, sentences)),dtype=int)
            x = x_list
            y = y_list
            before_idx = sentences_idx
            temp_valid_count,temp_total = self._test_run(x,y)
            valid_count += temp_valid_count
            total += temp_total
        print(f"ACCURACY : {self.check_valid(valid_count,total)}")
    def _splitter_x(self, x):
        x = x.split()
        result = []
        for i in range(1,self.max_len):
            if i < len(x):
                try:
                    result.append(self.dict[x[i].split('/')[0]])
                except:
                    result.append(self.dict["<UKN>"])
            else:
                result.append(self.dict["<PAD>"])
        return result
    def _splitter_y(self, x):
        x = x.split()
        result = []
        for i in range(1,self.max_len):
            if i < len(x):
                try:
                    result.append(self.dict_y[x[i].split('/')[1]])
                except:
                    result.append(self.dict_y["<UKN>"])
            else:
                result.append(self.dict_y["<PAD>"])
        return result
    def save_model(self):
        with open("model_RNN.pt","wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
    def load_model(self,name):
        with open(name,"rb") as f:
            model = pickle.load(f)
        return model
    def check_valid(self,predict,label):
        return predict/label 
def load_file(name):
    with open(name,"r") as f:
        file = f.read()
    return file
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", "--train",  action="store_true")
    parser.add_argument("-test", "--test", action="store_true")
    parser.add_argument("-run", "--run", action="store_true")
    parser.add_argument("-gpu", "--gpu", action="store_true")
    parser.add_argument("-dim", "--dim", type=int, default=200, action="store")
    parser.add_argument("-epochs", "--epochs", type=int, default=7, action="store")
    parser.add_argument("-batchs", "--batchs", type=int, default=16, action="store")
    parser.add_argument("-lr", "--lr", type=float, default=0.1, action="store")
    parser.add_argument("-train_set", "--train_set", default="tagged_train.txt", action="store")
    parser.add_argument("-test_set", "--test_set", default="tagged_test.txt", action="store")
    parser.add_argument("-model", "--model", default="model_RNN.pt", action="store")
    args = parser.parse_args()
    print(args)
    if args.gpu == True:
        print("GPU")
        import cupy as cp
    else:
        print("CPU")
        import numpy as cp
    if args.train == True:
        rnn = RNN(load_file(args.train_set),args.epochs,args.batchs,args.lr,args.dim)
        rnn.train()
        rnn.save_model()
    elif args.test == True:
        rnn = RNN()
        rnn = rnn.load_model(args.model)
        rnn.test(load_file(args.test_set))
    elif args.run == True:
        rnn = RNN()
