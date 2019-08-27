import numpy as np

class TwoLayerNet():
    # input - Linear - ReLU - Linear - Softmax (Softmax 결과는 입력 N개의 데이터에 대해 각 클래스에 대한 확률)

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
         self.params = {}
         self.params["W1"] = std * np.random.randn(input_size, hidden_size) # i -> h
         # std를 곱해주는 이유: 초기값 설정을 할 때 표준편차가 크면 시그모이드에서 발산하니까 0으로 모아줌.
         self.params["b1"] = np.zeros(hidden_size)
         self.params["W2"] = std * np.random.randn(hidden_size, output_size) # h -> o
         self.params["b2"] = np.zeros(output_size)

    def forward(self, X, y=None): 
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        N, D = X.shape

        h = np.dot(X, W1) + b1
        a = np.maximum(0, h) #relu

        o = np.dot(a, W2) + b2
        exps = np.exp(o)
        p = exps / np.sum(exps, axis=1).reshape(-1, 1) #softmax

        if y is None:
            return p, a

        LL = -np.log(p[np.arange(len(y)), y])  # get LogLikelihood
        Loss = LL.sum() / len(y)  # get p loss

        return Loss


    def backward(self, X, y, learning_rate=1e-5):
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]

        N = X.shape[0] # 데이터 개수
        grads = {}

        p, a = self.forward(X)
        
        dp = p.copy()
        dp[np.arange(N), y] -= 1
        dp /= N # p-y

        da = np.dot(np.dot(dp, W2.T),np.maximum(0,1))

        # shape 명시
        grads["W2"] = np.zeros([W2.shape[0], W2.shape[1]])
        grads["b2"] = np.zeros(len(b2))
        grads["W1"] = np.zeros([W1.shape[0], W1.shape[1]])
        grads["b1"] = np.zeros(len(b1))
        
        # 연산
        grads["W2"] = np.dot(a.T, dp)
        grads["b2"] = np.sum(dp, axis = 0)
        grads["W1"] = np.dot(X.T, da)
        grads["b1"] = np.sum(da, axis = 0)

        # 업데이트
        self.params["W2"] = self.params["W2"] - learning_rate * grads["W2"]
        self.params["b2"] = self.params["b2"] - learning_rate * grads["b2"]
        self.params["W1"] = self.params["W1"] - learning_rate * grads["W1"]
        self.params["b1"] = self.params["b1"] - learning_rate * grads["b1"]

    def accuracy(self, X, y):
        p, _ = self.forward(X)
        y_pred = np.argmax(p, axis=1) 
        return np.sum(y_pred==y)/len(X)