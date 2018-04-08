# Report

## 1. Introduction

This assignment is to train and test a one-layer network with multi outputs to classify images from CIFAR-10 dataset.

In the assignment, mini-batch gradient descent was implemented and applied to the dataset. Cross-entropy loss of the classifier was computed for training.

## 2. Methods

#### 2.1 Function ```EvaluateClassifier```

The equations used:

<img src="http://chart.googleapis.com/chart?cht=tx&chl=%20s%20%3D%20Wx%2Bb" style="border:none;">

<img src="http://chart.googleapis.com/chart?cht=tx&chl=%20p%20%3D%20SOFTMAX%28s%29%20%3D%20%5Cfrac%7Bexp%28s%29%7D%7B1%5ET%20exp%28s%29%7D" style="border:none;">

The corresponding code in Matlab is as following:

```Matlab
s = W * X;
s = bsxfun(@plus, s, b);
P = bsxfun(@rdivide, exp(s), sum(exp(s), 1));
```

I used ```bsxfun``` here to apply element to element plus on ```W * X``` and ```b```, since **W** is _K x d_ and **X** is _d x N_ but **b** is _K x 1_.

Then, to implement softmax, I used ```bsxfun``` to apply element to element divide on **exp(s)** and **sum of the first dimension of exp(s)**.

#### 2.2 Function ```ComputeCost```

The equations used:

<img src="http://chart.googleapis.com/chart?cht=tx&chl=%20J%28D%2C%5Clambda%2CW%2Cb%29%20%3D%20%5Cfrac%7B1%7D%7B%7CD%7C%7D%20%5Csum_%7Bx%2Cy%5Cin%20D%7D%20l_%7Bcross%7D%20%28x%2Cy%2CW%2Cb%29%20%2B%20%5Clambda%5Csum_%7Bi%2Cj%7DW%5E2_%7Bij%7D" style="border:none;">

<img src="http://chart.googleapis.com/chart?cht=tx&chl=%20l_%7Bcross%7D%20%28x%2Cy%2CW%2Cb%29%20%3D%20-log%28y%5ETp%29" style="border:none;">

The corresponding code in Matlab is as following:

```Matlab
P = EvaluateClassifier(X, W, b);
J = -sum(log(sum(Y.*P, 1)))/n + lambda*sumsqr(W);
```

Here I used ```EvaluateClassifier``` to calculate ```p``` first and then apply p to the first equation. **Y** here is already the transpose because it is initialized as _K x n_. P is _K x N_, so it needs dot product.

#### 2.3 Function ```ComputeGradients ```

The gradients are computed according to the following equations:

<img src="http://chart.googleapis.com/chart?cht=tx&chl=%20%5Cfrac%7B%5Cpartial%20J%28B%5E%7Bt%2B1%7D%2C%5Clambda%2CW%2Cb%29%7D%7B%5Cpartial%20W%7D%20%3D%20%5Cfrac%7B1%7D%7B%7CB%5E%7Bt%2B1%7D%7C%7D%20%5Csum_%7B%28x%2Cy%29%5Cin%20B%5E%7Bt%2B1%7D%7D%20%5Cfrac%7B%5Cpartial%20l_%7Bcross%7D%28x%2Cy%2CW%2Cb%29%7D%7B%5Cpartial%20W%7D%20%2B%202%5Clambda%20W" style="border:none;">

<img src="http://chart.googleapis.com/chart?cht=tx&chl=%20%5Cfrac%7B%5Cpartial%20J%28B%5E%7Bt%2B1%7D%2C%5Clambda%2CW%2Cb%29%7D%7B%5Cpartial%20b%7D%20%3D%20%5Cfrac%7B1%7D%7B%7CB%5E%7Bt%2B1%7D%7C%7D%20%5Csum_%7B%28x%2Cy%29%5Cin%20B%5E%7Bt%2B1%7D%7D%20%5Cfrac%7B%5Cpartial%20l_%7Bcross%7D%28x%2Cy%2CW%2Cb%29%7D%7B%5Cpartial%20b%7D" style="border:none;">

According to the lecture, the main steps are:

For each (x,y):

1. Evaluate p = Softmax(Wx+b)
2. Let g = -(y-p)T
3. <img src="http://chart.googleapis.com/chart?cht=tx&chl=%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20b%7D%20%2B%3D%20g" style="border:none;">
4. <img src="http://chart.googleapis.com/chart?cht=tx&chl=%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%7D%20%2B%3D%20g%5ETx%5ET" style="border:none;">

Then, divide them by the number of entries and add gradient for regularization term.

The corresponding code in Matlab is as following:

```Matlab
for i = 1:n
    ......
    g = -(y-p)';
    grad_b = grad_b + g';
    grad_W = grad_W + g' * x';
end

grad_b = grad_b/n;
grad_W = grad_W/n + 2 * lambda * W;
```

The logic is very straight forward here.

#### 2.4 Function ```MiniBatchGD```

Here is how we update ```W,b``` in ```MiniBatchGD```:

<img src="http://chart.googleapis.com/chart?cht=tx&chl=%20W%5E%7B%28t%2B1%29%7D%20%3D%20W%5E%7B%28t%29%7D%20-%20%5Ceta%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20W%7D%0D%0A%0D%0A" style="border:none;">

<img src="http://chart.googleapis.com/chart?cht=tx&chl=%20b%5E%7B%28t%2B1%29%7D%20%3D%20b%5E%7B%28t%29%7D%20-%20%5Ceta%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20b%7D%0D%0A" style="border:none;">

## 3. Results

#### 3.1 Parameter setting1

Parameters:

```
lambda=0; n_epochs=40; n_batch=100; eta=.1;
```

Accurary: 25.61%

![Representing images](https://github.com/MandyXue/DeepLearning-Assignment/blob/master/A1/Result_Pics/image1.png)

<img src="https://github.com/MandyXue/DeepLearning-Assignment/blob/master/A1/Result_Pics/loss1.png" width=500/>

#### 3.2 Parameter setting2

Parameters:

```
lambda=0; n_epochs=40; n_batch=100; eta=.01;
```

Accuracy: 36.87%

![](https://github.com/MandyXue/DeepLearning-Assignment/blob/master/A1/Result_Pics/image2.png)

<img src="https://github.com/MandyXue/DeepLearning-Assignment/blob/master/A1/Result_Pics/loss2.png" width=500/>

#### 3.3 Parameter setting3

Parameters:

```
lambda=.1; n_epochs=40; n_batch=100; eta=.01;
```

Accuracy: 33.36%

![](https://github.com/MandyXue/DeepLearning-Assignment/blob/master/A1/Result_Pics/image3.png)

<img src="https://github.com/MandyXue/DeepLearning-Assignment/blob/master/A1/Result_Pics/loss3.png" width=500/>

#### 3.4 Parameter setting4

Parameters:

```
lambda=1; n_epochs=40; n_batch=100; eta=.01;
```

Accuracy: 21.93%

![](https://github.com/MandyXue/DeepLearning-Assignment/blob/master/A1/Result_Pics/image4.png)

<img src="https://github.com/MandyXue/DeepLearning-Assignment/blob/master/A1/Result_Pics/loss4.png" width=500/>

