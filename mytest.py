import read_file as r
import softmax_regression as reg
import numpy as np
import gradient_descent as gr

data = r.read("MSRC/MSRC.mat")
label = data[:, -1]
data = data[:, :-1]
reg.load_data(data, label)
classes = len(set(label))
features = data.shape[1]
# model = reg.gradient_log_likelihood(np.random.rand(classes, features))
model = reg.gradient_log_likelihood
coef = gr.gradient_descent(model, 6, 240)
predicted = reg.predict(coef)

for i in range(len(label)):
    print(predicted[i],label[i])
