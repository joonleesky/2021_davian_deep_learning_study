
# DAVIAN Lab. Deep Learning Winter Study (2021)

- **Writer:** Min-Jung Kim

## Information

- **Title:** (cs229) Lecture 4 : Perceptron. Exponential Family. Generalized Linear Models.
- **Link:** http://cs229.stanford.edu/notes2020fall/notes2020fall/cs229-notes1.pdf   
http://cs229.stanford.edu/livenotes2020spring/cs229-livenotes-lecture4.pdf
- **Keywords:** Perceptron, Exponential Family, Generalized Linear Model, Softmax Regression (Multi-class classification)

## Perceptron
Perceptron is somewhat similar to sigmoid function but different.   
It is hard version of sigmoid function.   

- **Logistic Regression with Sigmoid function**   
<img src="images/sigmoid_function.png"></img>    
   
- **Perceptron**   
<img src="images/perceptron.png"></img>    
   
- **Geometrical Interpretation of Perceptron theta update.**   
<img src="images/perceptron_update.png"></img>    

> Perceptron is not something that's widely used in practice.   
> We study it mostly for historical reasons.   
> It is not used because it does not have a probabilistic interpretation of what 's happening.      
> Also it could never classify xor   

## Exponential Family
It is class of probability distributions, whos PDF can be written in the form   
   
- <img src="http://latex.codecogs.com/gif.latex?\dpi{110}&space;p(y;\eta)=b(y)exp[\eta^{T}T(y)-a(\eta)]\textbf{}" title="http://latex.codecogs.com/gif.latex?\dpi{110} p(y;\eta)=b(y)exp[\eta^{T}T(y)-a(\eta)]\textbf{}" /> => integrates to 1
      
> y : data (output)   
> <img src="http://latex.codecogs.com/gif.latex?\dpi{110}&space;\eta" title="http://latex.codecogs.com/gif.latex?\dpi{110} \eta" /> : natural parameter (parameter of distribution)   
> b(y) : base measure   
> T(y) : sufficient statistic. In this lecture, T(y) = y    
> <img src="http://latex.codecogs.com/gif.latex?\dpi{110}&space;a(\eta)" title="http://latex.codecogs.com/gif.latex?\dpi{110} a(\eta)" /> : log partition, normalizing constant

### Ex1 : Bernoulli Distribution
**Bernoulli Distribution**
<img src="http://latex.codecogs.com/gif.latex?\dpi{110}&space;p(y;\phi)=\phi^y(1-\phi)^{1-y}" title="http://latex.codecogs.com/gif.latex?\dpi{110} p(y;\phi)=\phi^y(1-\phi)^{1-y}" />   
<img src="http://latex.codecogs.com/gif.latex?\dpi{110}&space;\phi" title="http://latex.codecogs.com/gif.latex?\dpi{110} \phi" /> : probability of event   

**Matching with E.F.**   
<img src="http://latex.codecogs.com/gif.latex?\dpi{110}&space;p(y;\phi)&space;=&space;\phi^{y}(1-\phi)^{1-y}&space;=&space;e^{log(\phi^{y}(1-\phi)^{1-y})}=e^{ylog\phi&plus;(1-y)log(1-\phi)}=e^{ylog\frac{\phi}{1-\phi}-log\frac{1}{1-\phi}" title="http://latex.codecogs.com/gif.latex?\dpi{110} p(y;\phi) = \phi^{y}(1-\phi)^{1-y} = e^{log(\phi^{y}(1-\phi)^{1-y})}=e^{ylog\phi+(1-y)log(1-\phi)}=e^{ylog\frac{\phi}{1-\phi}-log\frac{1}{1-\phi}" />   
<img src="http://latex.codecogs.com/gif.latex?\dpi{110}&space;\therefore&space;T(y)=y,&space;\eta^{T}=log(\frac{\phi}{1-\phi}),&space;\phi=\frac{1}{1&plus;e^{-\eta}},&space;a(\eta)=-log(1-\phi)=log(1&plus;e^{\eta})" title="http://latex.codecogs.com/gif.latex?\dpi{110} \therefore T(y)=y, \eta^{T}=log(\frac{\phi}{1-\phi}), \phi=\frac{1}{1+e^{-\eta}}, a(\eta)=-log(1-\phi)=log(1+e^{\eta})" />

### Ex2 : Gaussian Distribution (with fixed variance)
assume    
<img src="http://latex.codecogs.com/gif.latex?\dpi{110}&space;\sigma^{2}=fixed=1" title="http://latex.codecogs.com/gif.latex?\dpi{110} \sigma^{2}=fixed=1" />    
Gaussian Dist.   
<img src="http://latex.codecogs.com/gif.latex?\dpi{110}&space;\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}" title="http://latex.codecogs.com/gif.latex?\dpi{110} \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}" /> => <img src="http://latex.codecogs.com/gif.latex?\dpi{110}&space;\frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}(x-\mu)^2}" title="http://latex.codecogs.com/gif.latex?\dpi{110} \frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}(x-\mu)^2}" />   
   
**Matching with E.F.**   
<img src="http://latex.codecogs.com/gif.latex?\dpi{110}&space;\therefore&space;b(y)=\frac{1}{\sqrt{2\pi}}e^{-\frac{y^{2}}{2}},&space;T(y)=y,&space;\mu=\eta,&space;a(\eta)=\frac{-\eta^{2}}{2}" title="http://latex.codecogs.com/gif.latex?\dpi{110} \therefore b(y)=\frac{1}{\sqrt{2\pi}}e^{-\frac{y^{2}}{2}}, T(y)=y, \mu=\eta, a(\eta)=\frac{-\eta^{2}}{2}" />   
   
### Exponential Family Properties
a) MLE w.r.t. <img src="http://latex.codecogs.com/gif.latex?\dpi{110}&space;\eta" title="http://latex.codecogs.com/gif.latex?\dpi{110} \eta" /> --> concave
(Which means.. If we perform maximum likelihood on E.F, and when E.F. is parameterized in the natural parameters, then the optimization problem is concave.   
 Therefore, Negative Log Likelihood is convex)   
b) <img src="http://latex.codecogs.com/gif.latex?\dpi{110}&space;E[y;\eta]&space;=&space;\frac{\partial}{\partial&space;\eta}a(\eta)" title="http://latex.codecogs.com/gif.latex?\dpi{110} E[y;\eta] = \frac{\partial}{\partial \eta}a(\eta)" />   
c) <img src="http://latex.codecogs.com/gif.latex?\dpi{110}&space;Var[y;\eta]=\frac{\partial^{2}}{\partial\eta^{2}}a(\eta)" title="http://latex.codecogs.com/gif.latex?\dpi{110} Var[y;\eta]=\frac{\partial^{2}}{\partial\eta^{2}}a(\eta)" />   
   
## GLM (Generalized Linear Model)   
We can build a lot of powerful models by choosing nappropriate E.F and plugging it onto a linear model.   

### Assumptions / Design Choices   
a) <img src="http://latex.codecogs.com/gif.latex?\dpi{110}&space;y|x,&space;\theta" title="http://latex.codecogs.com/gif.latex?\dpi{110} y|x, \theta" /> ~ Exponential Family   
Depending on the problem that you have, you can choose any member of E.F as parameterized by <img src="http://latex.codecogs.com/gif.latex?\dpi{110}&space;\eta" title="http://latex.codecogs.com/gif.latex?\dpi{110} \eta" />   
b) <img src="http://latex.codecogs.com/gif.latex?\dpi{110}&space;\eta&space;=&space;\theta^{T}x" title="http://latex.codecogs.com/gif.latex?\dpi{110} \eta = \theta^{T}x" />   
c) Test Time Output = <img src="http://latex.codecogs.com/gif.latex?\dpi{110}&space;E[y|x;\theta]" title="http://latex.codecogs.com/gif.latex?\dpi{110} E[y|x;\theta]" />   
   
<img src="images/GLM.png"></img>    

### GLM Training   
No matter what kind of GLM you are doing, no matter which choice of distribution that you make,    
the learning update rule is the same.
   
**Learning Update Rule**   
<img src="http://latex.codecogs.com/gif.latex?\dpi{110}&space;\theta_{j}:=\theta_{j}&plus;\alpha(y^{(i)}-h_{\theta}(x^{(i)}))x_{j}^{(i)}" title="http://latex.codecogs.com/gif.latex?\dpi{110} \theta_{j}:=\theta_{j}+\alpha(y^{(i)}-h_{\theta}(x^{(i)}))x_{j}^{(i)}" />   

### Terminology   
<img src="http://latex.codecogs.com/gif.latex?\dpi{110}&space;\mu=E[y;\eta]=g(\eta)=\frac{\partial}{\partial&space;\eta}a(\eta)" title="http://latex.codecogs.com/gif.latex?\dpi{110} \mu=E[y;\eta]=g(\eta)=\frac{\partial}{\partial \eta}a(\eta)" /> = canonical response function   
<img src="http://latex.codecogs.com/gif.latex?\dpi{110}&space;\eta=g^{-1}(\mu)" title="http://latex.codecogs.com/gif.latex?\dpi{110} \eta=g^{-1}(\mu)" /> = canonical link function   
   
### 3 Parameterization   
<img src="images/GLM_params.png"></img>   
   
## Softmax Regression 
Yet another member of GLM family.
Usually, hypothesis equals probability or scalar, while softmax outputs prob. distribution.   
<img src="http://latex.codecogs.com/gif.latex?\dpi{110}&space;\frac{e^{\theta_{i}^{T}x}}{\sum_{i\in&space;{class1,&space;class2,&space;...}}^{}e^{\theta_{i}^{T}x}}" title="http://latex.codecogs.com/gif.latex?\dpi{110} \frac{e^{\theta_{i}^{T}x}}{\sum_{i\in {class1, class2, ...}}^{}e^{\theta_{i}^{T}x}}" />
