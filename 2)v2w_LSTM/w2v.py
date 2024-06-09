import cupy as cp
import numpy as np
import string
import os
from tqdm import tqdm
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = cp.exp(x - cp.max(x))
    return e_x / e_x.sum()

class word2vec(object):
    def __init__(self):
        self.N = 200
        self.X_train = []
        self.y_train = []
        self.window_size = 2
        self.alpha = 0.0001
        self.words = []
        self.word_index = {}
    
    def initialize(self,V,data):
        self.V = V
        self.W = cp.random.uniform(-0.8, 0.8, (self.V, self.N),dtype=cp.float32)
        self.W1 = cp.random.uniform(-0.8, 0.8, (self.N, self.V),dtype=cp.float32)
        self.words = data
        self.word_index[0] = "<UKN>"
        self.word_index[1] = "<END>"
        self.word_index[2] = "<PAD>"
        for i in range(len(data)):
            self.word_index[data[i]] = i
#        self.W = cp.random.randn(self.V,self.N)
#        self.W1 = cp.random.randn(self.N,self.V)
    def _clipping(self,x):
        return cp.clip(x, -0.8, 0.8, out=x)
    def feed_forward(self,X):
        self.h = cp.dot(self.W.T,X).reshape(self.N,1)
        self.u = cp.dot(self.W1.T,self.h)
        #print(self.u)
        self.y = softmax(self.u)  
        return self.y
        
    def backpropagate(self,x,t):
        e = self.y - cp.asarray(t).reshape(self.V,1)
        # e.shape is V x 1
      
        dLdW1 = cp.dot(self.h,e.T)
        X = cp.array(x).reshape(self.V,1)
        dLdW = cp.dot(X, cp.dot(self.W1,e).T)
        self.W1 = self.W1 - self.alpha*self._clipping(dLdW1)
        self.W = self.W - self.alpha*self._clipping(dLdW)
        
    def train(self,epochs):
        for x in tqdm(range(0,epochs)):  
            self.loss = 0
            for j in range(len(self.X_train)):
                x = cp.asarray(self.X_train[j])
                y = cp.asarray(self.y_train[j])
                self.feed_forward(x)
                self.backpropagate(x,y)
                C = 0
                for m in range(self.V):
                    if(self.y_train[j][m]):
                        self.loss += -1*self.u[m][0]
                        C += 1
                self.loss += C*cp.log(cp.sum(cp.exp(self.u)))
            print("loss = ",self.loss)
            
    def predict(self,word,number_of_predictions):
        if word in self.words:
            index = self.word_index[word]
            X = [0 for i in range(self.V)]
            X[index] = 1
            prediction = self.feed_forward(X)
            output = {}
            for i in range(self.V):
                output[prediction[i][0]] = i
            
            top_context_words = []
            for k in sorted(output,reverse=True):
                top_context_words.append(self.words[output[k]])
                if(len(top_context_words)>=number_of_predictions):
                    break
    
            return top_context_words
        else:
            print("Word not found in dicitonary")
    def embed(self,word):
        k = []
        for i in range(len(word)):
            try:
                index = self.word_index[word[i]]
            except:
                index = 0
            X = np.zeros(self.V,dtype=int)
            X[index] = 1
            k.append(X)
        h = cp.dot(self.W.T,cp.array(k).T)
        return h

def preprocessing(corpus):
    training_data = []
    sentences = corpus.split(".")
    for i in tqdm(range(len(sentences))):
        sentences[i] = sentences[i].strip()
        sentence = sentences[i].split()
        x = [word.strip(string.punctuation) for word in sentence if "::" not in word]
        x = [word.lower() for word in x]
        training_data.append(x)
    return training_data
    
def prepare_data_for_training(sentences,w2v):
    data = {}
    for sentence in tqdm(sentences):
        for word in sentence:
            if word not in data:
                data[word] = 1
            else:
                data[word] += 1
    V = len(data)
    data = sorted(list(data.keys()))
    vocab = {}
    for i in range(len(data)):
        vocab[data[i]] = i
    pattern = r'[\t]'
    #for i in range(len(words)):
    bi = 0
    w2v.initialize(V,data)
    for i in range(2):
        for i in tqdm(range(10,int(len(sentences)/1000),10)):
            for sentence in sentences[bi:i]:
                bi = i
                for i in range(len(sentence)):
                    center_word = np.zeros(V,dtype=int)
                    try:
                        center_word[vocab[sentence[i]]] = 1
                    except:
                        continue
                    context = np.zeros(V,dtype=int)
                    if i>0:
                        if i>1:
                            try:
                                context[vocab[sentence[i-2]]] += 1
                            except:
                                pass
                        try:
                            context[vocab[sentence[i-1]]] += 1
                        except:
                                pass
                    max = len(sentence)
                    if i<max-1:
                        if i<max-2:
                            try:
                                context[vocab[sentence[i+2]]] += 1
                            except:
                                pass
                        try:
                            context[vocab[sentence[i+1]]] += 1
                        except:
                                pass
                    w2v.X_train.append(center_word)
                    w2v.y_train.append(context)
            w2v.train(1) 
            w2v.X_train = []
            w2v.y_train = []
    return w2v.X_train,w2v.y_train    
if __name__ == "__main__":
    with open('raw_train.txt', 'r') as myfile:
        data=myfile.read().replace('\n', '')
    w2v = word2vec()
    training_data = preprocessing(data)
    X,y = prepare_data_for_training(training_data,w2v)
    with open('./w2v_model.pt','wb') as f:
        pickle.dump(w2v, f, protocol=pickle.HIGHEST_PROTOCOL)
    