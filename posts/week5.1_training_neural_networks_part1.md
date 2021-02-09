# DAVIAN Lab. Deep Learning Winter Study (2021)

- **Writer:** Yuri Kim

## Information

- **Title:** (cs231n) Lecture 7 : Training Neural Networks, part I
- **Link:** http://cs231n.stanford.edu/slides/2020/lecture_7.pdf
- **Keywords:** Activation functions, data processing, Batch Normalization, Transfer learning
-------------------------------------------------------

## Details of training Neural Networks
- How do we set up our neural networks at the beginning?
- Which activation functions should we choose?
- How to preprocess the data
- Weight initialization, regularization and gradient checking
- Training dynamics (which include spcecific parameter update rules & hyperparemter optimization)
- Evaluation & model ensembles

## Activation Functions : 7 types

### 1. Sigmoid
<img src="https://latex.codecogs.com/gif.latex?\sigma(x)&space;=&space;{1&space;\over&space;1&space;&plus;&space;e^{-x}}" title="\sigma(x) = {1 \over 1 + e^{-x}}" />

![](images/sigmoid.png)
- The **sigmoid** function takes each element and squashes it into the range of [0,1]
    Thus, if you get very high values as input, the output would be very near to 1. On the other hand for negative values, it will be near zero
- Saturation at 'firing rates' of a neuron

#### Problem
1. Saturated neurons can kill off the gradients : In the regions where the function is flat, the gradient will be 0. Hence when we multiply it with the upstream gradient, it will become very small and "kill" the gradient flow
2. Sigmoid outputs are not zero-centered : If the inputs are all positive, the gradients on the weights will all be in the same direction which gives very inefficient gradient updates

**Example)** When w(weights) is 2-dimensional, only two gradient update directions will be possible which will result in a **zig zag path** which is less efficient than if the optimal direction was directly followed

3. Calculating the exponential part of the sigmoid function is expensive

### 2. Tanh
![](images/tanh.png)
- Tanh squashes each element into a range of [-1,1] : **zero-centered**

#### Problem
- However it still kills the gradient flow when saturated(flat)
&#8594; a bit better than sigmoid

### 3. ReLU
f(x)=max(0,x)
![](images/relu.png)
- It doesn't saturate in the positive region
- Computationally efficient
- Coverges about 6 times faster than sigmoid/tanh
- Biologically more plausible : closer approximation in neuroscience experiments

#### Problem
- Still not zero-centered
- The negative half is still saturated and kills the gradient
- **Dead ReLU** : will never activate and update
```
Reason 1. Bad initialization : weights can be off the data cloud and will never get input to activate
Reason 2. High learning rate : start off with a good ReLU but the updates are too huge and the weights jump around
→ ReLU units get knocked off the data manifold during training
```
> Initializing with slightly positive biases may increase the likelihood of it being active and more **firing** ReLUs

### 4. Leaky ReLU
f(x)=max(0.01x,x)
![](images/leaky_relu.png)
- No saturation at the negative space(No gradient dying problem)
- Computationally efficient
- Converges much faster than sigmoid/tanh

### 5. PReLU
f(x)=max(ax,x)
![](images/prelu.png)
- Because the slope in the negative regime is trained as a parameter, it gives the model more flexibility

### 6. ELU
<img src="https://latex.codecogs.com/gif.latex?f(x)=\begin{cases}&space;&x&space;\text{&space;if&space;}&space;x>0&space;\\&space;&\alpha(exp(x)-1)&space;\text{&space;if&space;}&space;x\leq&space;0&space;\end{cases}" title="f(x)=\begin{cases} &x \text{ if } x>0 \\ &\alpha(exp(x)-1) \text{ if } x\leq 0 \end{cases}" />

![](images/elu.png)
- Closer to zero mean outputs
- Compared to leaky ReLU, it builds back to the negative saturation regime which adds more robustness to noise

### 7. Maxout Neuron
<img src="https://latex.codecogs.com/gif.latex?max(w{_1}^{T}&plus;b{_1},w{_2}^{T}&plus;b{_2})" title="max(w{_1}^{T}+b{_1},w{_2}^{T}+b{_2})" />

![](images/maxout.png)
- Generalizes the ReLU and the Leaky ReLU
- Operating in a linear regime, it doesn’t saturate or die
</br> &#8594; However this doubles the number of parameters and neurons

&#8756; In conclusion, **use ReLU** with caution in adjusting learning rates. Also, try out other variants of it or tanh but don't use sigmoid.

## Data preprocessing


### Conclusion


#### references
- 
