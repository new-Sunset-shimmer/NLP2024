import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import re
from tqdm import tqdm
import pickle
import argparse
from copy import deepcopy
from w2v import word2vec
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
class LSTMCell(layer):
    def forward(self, x, h_prev, c_prev, weight):
        z = cp.concatenate((h_prev, x), axis=0)
        i = cp.concatenate((weight['Wi'], weight['Ui']), axis=1)
        f = cp.concatenate((weight['Wf'], weight['Uf']), axis=1)
        o = cp.concatenate((weight['Wo'], weight['Uo']), axis=1)
        g = cp.concatenate((weight['Wg'], weight['Ug']), axis=1)
        i = self.sigmoid(self.linear(i,z))
        f = self.sigmoid(self.linear(f,z))
        o = self.sigmoid(self.linear(o,z))
        g = cp.tanh(self.linear(g,z))
        c = f * c_prev + i * g
        tanhc = cp.tanh(c)
        h = o * tanhc
        
        return h, c,(h,c,h_prev,c_prev,f,i,g,o,x,tanhc)
    
    def backward(self, dh_next, dc_next, x, weight):
        # Derivative of tanh activation for c
        h_next, c_next, h_prev, c_prev, ft, it, gt, ot, xt, tanhc = x
        
        n_a, m = h_next.shape
        
        dot = dh_next * tanhc * ot * (1 - ot)
        dcct = (dc_next * it + ot * (1 - cp.square(tanhc)) * it * dh_next) * (1 - np.square(gt))
        dit = (dc_next * gt + ot * (1 - cp.square(tanhc)) * gt * dh_next) * it * (1 - it)
        dft = (dc_next * c_prev + ot *(1 - cp.square(tanhc)) * c_prev * dh_next) * ft * (1 - ft)
        
        concat = cp.concatenate((h_prev, xt), axis=0)
        dW = {}
        dU = {}
        m = dh_next.shape[0]
        temp_i = cp.dot(dit, concat.T)
        temp_f = cp.dot(dft, concat.T)
        temp_g = cp.dot(dcct, concat.T)
        temp_o = cp.dot(dot, concat.T)
        dW['Wf'] = temp_f[:,:m]
        dW['Wi'] = temp_i[:,:m]
        dW['Wg'] = temp_g[:,:m]
        dW['Wo'] = temp_o[:,:m]
        
        dU['Uf'] = temp_f[:,m:]
        dU['Ui'] = temp_i[:,m:]
        dU['Ug'] = temp_g[:,m:]
        dU['Uo'] = temp_o[:,m:]

        dh_prev = cp.dot(weight['Wf'][:, :n_a].T, dft) + cp.dot(weight['Wi'][:, :n_a].T, dit) + cp.dot(weight['Wg'][:, :n_a].T, dcct) + cp.dot(weight['Wo'][:, :n_a].T, dot)
        dc_prev = dc_next * ft + ot * (1 - cp.square(cp.tanh(tanhc))) * ft * dh_next
        dxt = cp.dot(weight['Wf'][:, :n_a].T, dft) + cp.dot(weight['Wi'][:, :n_a].T, dit) + cp.dot(weight['Wg'][:, :n_a].T, dcct) + cp.dot(weight['Wo'][:, :n_a].T, dot)
        return dh_prev, dc_prev, dW, dU,dxt
    
    def sigmoid(self, x):
        return 1 / (1 + cp.exp(-x))
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
        exp_x_sum = cp.sum(exp_x,-1).reshape(-1,1) + 1e-19
        o = exp_x / exp_x_sum
        return o
    def forward(self,x,V,y):
        y = self._get_onehot(y)
        o = self.softmax(x,V)
        self.y = y
        return self.linear(-y.T,cp.log(o+1e-9)).sum(), (self.o_gamma, self.y, self.x, self.V)
    def backward(self,x):
        o_gamma,y,x,V = x
        return o_gamma - y, x, V

class LSTM:
    def __init__(self,dataset = None,w2v = None,epochs = 10, batchs = 4096, lr = 0.01, dim = 300):
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
            self.weight['Uo'] = cp.random.randn(self.dim,self.dim,dtype=cp.float32)
            self.weight['Wo'] = cp.random.randn(self.dim,self.dim,dtype=cp.float32)
            self.weight['Ui'] = cp.random.randn(self.dim,self.dim,dtype=cp.float32)
            self.weight['Wi'] = cp.random.randn(self.dim,self.dim,dtype=cp.float32)
            self.weight['Uf'] = cp.random.randn(self.dim,self.dim,dtype=cp.float32)
            self.weight['Wf'] = cp.random.randn(self.dim,self.dim,dtype=cp.float32)
            self.weight['Ug'] = cp.random.randn(self.dim,self.dim,dtype=cp.float32)
            self.weight['Wg'] = cp.random.randn(self.dim,self.dim,dtype=cp.float32)
            self.weight['V'] = cp.random.randn(self.dictlen_y,self.dim,dtype=cp.float32)
            self.embed = w2v
            self.hiddenstate = LSTMCell()
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
    def forward(self,x_ground,y_ground,x_word):
        self.tracker = []
        h = cp.zeros((self.dim,self.batchs))
        c = cp.zeros((self.dim,self.batchs))
        for i in range(self.max_len-1):
            x = x_ground[:,i]
            self.layer = []
            x = self.embed.embed(x_word[:,i])
            h, c, forbackward = self.hiddenstate.forward(x,h,c,self.weight)
            self.layer.append(forbackward)
            y = y_ground[:,i]
            x,forbackward = self.output.forward(h,self.weight['V'],y)    
            self.layer.append(forbackward)
            self.tracker.append(self.layer)
    def backward(self):
        self.tracker.reverse()
        h = cp.zeros((self.dim,self.batchs))
        c = cp.zeros((self.dim,self.batchs))
        temp_Uo = cp.zeros_like(self.weight['Uo'])
        temp_Wo = cp.zeros_like(self.weight['Wo'])
        temp_Uf = cp.zeros_like(self.weight['Uf'])
        temp_Wf = cp.zeros_like(self.weight['Wf'])
        temp_Ug = cp.zeros_like(self.weight['Ug'])
        temp_Wg = cp.zeros_like(self.weight['Wg'])
        temp_Ui = cp.zeros_like(self.weight['Ui'])
        temp_Wi = cp.zeros_like(self.weight['Wi'])
        temp_V = cp.zeros_like(self.weight['V'])
        for step in self.tracker:
            dL,dV,dO = self.output.backward(step[-1])
            V = dL@dV.T
            x = dL.T@dO
            h = x.T + h
            h, c, dW,dU,dx = self.hiddenstate.backward(h, c, step[-2], self.weight)
            temp_Uo += self._clipping(dU['Uo'])
            temp_Wo += self._clipping(dW['Wo'])
            temp_Uf += self._clipping(dU['Uf'])
            temp_Wf += self._clipping(dW['Wf'])
            temp_Ug += self._clipping(dU['Ug'])
            temp_Wg += self._clipping(dW['Wg'])
            temp_Ui += self._clipping(dU['Ui'])
            temp_Wi += self._clipping(dW['Wi'])
            temp_V += self._clipping(V)
        self.weight['Uo'] -= self.lr*temp_Uo
        self.weight['Wo'] -= self.lr*temp_Wo
        self.weight['Uf'] -= self.lr*temp_Uf
        self.weight['Wf'] -= self.lr*temp_Wf
        self.weight['Ug'] -= self.lr*temp_Ug
        self.weight['Wg'] -= self.lr*temp_Wg
        self.weight['Ui'] -= self.lr*temp_Ui
        self.weight['Wi'] -= self.lr*temp_Wi
        self.weight['V'] -= self.lr*temp_V
        del temp_Uo, temp_Wo, temp_Uf, temp_Wf, temp_Ug, temp_Wg, temp_Ui, temp_Wi, temp_V
    def _clipping(self,x):
        return np.clip(x, -1, 1, out=x)
    def _test_run(self,x_ground,y_ground,x_word):
        h = cp.zeros((self.dim,self.batchs))
        c = cp.zeros((self.dim,self.batchs))
        valid_count = 0
        total = 0
        for i in tqdm(range(self.max_len-1)):
            x = x_ground[:,i]
            x = self.embed.embed(x_word[:,i])
            h, c, forbackward = self.hiddenstate.forward(x,h,c,self.weight)
            y = y_ground[:,i]
            x = self.output.softmax(h,self.weight['V'])    
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
                    x_list = cp.array(list(map(self._splitter_x, sentences)),dtype=int)
                    x_word = np.array(list(map(self._word_splitter_x, sentences)),dtype=object)
                    y_list = cp.array(list(map(self._splitter_y, sentences)),dtype=int)
                    x = x_list
                    y = y_list
                    before_idx = sentences_idx
                    self.forward(x,y,x_word)
                    self.backward()
    def test(self,test_set):
        before_idx = 0
        valid_count = 0
        total = 0
        _,_,self.sentences = self._word_label(test_set)
        for sentences_idx in tqdm(range(self.batchs,int(len(self.sentences)/6),self.batchs)):
            sentences = self.sentences[before_idx:sentences_idx]
            self.max_len = len(max(sentences, key=len).split())
            x_list = cp.array(list(map(self._splitter_x, sentences)),dtype=int)
            x_word = np.array(list(map(self._word_splitter_x, sentences)),dtype=object)
            y_list = cp.array(list(map(self._splitter_y, sentences)),dtype=int)
            x = x_list
            y = y_list
            before_idx = sentences_idx
            temp_valid_count,temp_total = self._test_run(x,y,x_word)
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
    def _word_splitter_x(self, x):
        x = x.split()
        result = []
        for i in range(1,self.max_len):
            if i < len(x):
                try:
                    result.append(x[i].split('/')[0])
                except:
                    result.append("<UKN>")
            else:
                result.append("<PAD>")
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
        with open("model_LSTMv2w.pt","wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
    def load_model(self,name):
        with open(name,"rb") as f:
            model = pickle.load(f)
        return model
    def check_valid(self,predict,label):
        return predict/label 
def load_file(name):
    with open(name,"r", encoding="utf-8") as f:
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
    parser.add_argument("-model", "--model", default="model_LSTMv2w.pt", action="store")
    args = parser.parse_args()
    print(args)
    if args.gpu == True:
        print("GPU")
        import cupy as cp
    else:
        print("CPU")
        import numpy as cp
    with open('./w2v_model.pt','rb') as f:
        w2v = pickle.load(f)
    if args.train == True:
        lstm = LSTM(load_file(args.train_set), w2v,args.epochs,args.batchs,args.lr,args.dim)
        lstm.train()
        lstm.save_model()
    elif args.test == True:
        lstm = LSTM()
        lstm = lstm.load_model(args.model)
        lstm.test(load_file(args.test_set))
    elif args.run == True:
        lstm = LSTM()
