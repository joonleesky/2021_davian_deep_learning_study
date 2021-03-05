# DAVIAN Lab. Deep Learning Winter Study (2021)

---

- Writer: Jaeun Jeong (VAE), Haneol Lee(Pixel CNN, Pixel RNN, GANs)

## Information

---

- **Title:** (cs231n) Lecture 13 : Visualizing and Understanding

- **Link:** http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture13.pdf

- **Keywords:** VAE, Explicit density model, PixelRNN, PixelCNN, Generative adversarial networks, KL-divergence, GANs problems, mode collapse

---



## Introduction

- Supervised Learning
  - Data: (x, y), x is data, y is label
  - Goal: Learn a function to map x -> y
  - Examples: Classification, regression, object detection, semantic segmentation, image captioning, etc.

- Unsupervised Learning
  - Data: x, x is data
  - Goal: Learn some underlying hidden structure of the data
  - Examples: Clustering, dimensionality reduction, feature learning, density estimation, etc

- Generative Models

  - Given training data, generate new samples from same distribution
  - Want to learn $p_{model}(x)$ similar to $p_{data}(x)$

  - **Taxonomy of Generative models:**![01_taxonomy of generative models](.\images\lecture13-01_taxonomy.png)image reference: [[1](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture13.pdf)]



## PixelRNN and PixelCNN

Explicit density model, optimizes exact likelihood, good samples. But inefficient sequential generation.

### Explicit density model

- Use chain rule to decompose likelihood of an image x into product of 1-d distributions:

  $p_{\theta}(x)=\prod_{i=1}^{n} p_{\theta}(x_i|x_1, ..., x_{i-1})$

  $p_{\theta}(x)$ : Likelihood of image X

  $\prod_{i=1}^{n} p_{\theta}(x_i|x_1, ..., x_{i-1})$ : Probability of i'th pixel value given all previous pixels

- Then maximize likelihood of training data.

- Complex distribution over pixel values (So express using a neural network)

- Need to define ordering of **previous pixels**

- #### PixelRNN

  ![03_PixelRNN](.\images\lecture13-03_PixelRNN.png) 

  image reference: [[1](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture13.pdf)]

  - Generate image pixels starting from corner.
  - Dependency on previous pixels now modeled using an RNN (LSTM).
    - In this example, the pixels to the left and top of the current pixel are defined as the previous pixels.
    - If no previous pixel, use  padding.
  - Drawback: 
    - Sequential generation is slow.

- #### PixelCNN

  ![04_PixelCNN](.\images\lecture13-04_PixelCNN.png)image reference: [[1](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture13.pdf)]

  - Generate image pixels starting from corner

  - Dependency on previous pixels now modeled using a CNN over context region

  - Training: maximize likelihood of training images

    $p_{\theta}(x)=\prod_{i=1}^{n} p_{\theta}(x_i|x_1, ..., x_{i-1})$

  - Drawback: 

    - The generation process is still slow. (Because generation must still proceed sequentially)
    - The major drawback of Pixel CNN is that it’s performance is worse than Pixel RNN.
    - Another drawback is the presence of a Blind Spot in the receptive field. (Masking all pixels next to i'th pixels (e.g. $i+1$, $i+2$, ...) in the receptive field at training step.)



---

## Variational Autoencoders (VAE) (It's 1st version using  Korean, it would be updated using English in the near future.)

#### Optimize variational lower bound on likelihood Useful latent representation, inference queries. But current sample quality not the best.

# Bayesian Framework

먼저, 베이지안의 기본적인 사고방식부터 알고 가자. 빈도론자와 베이지안의 가장 큰 차이점은 우리가 추정하고자하는 parameter를 확률변수로 보냐/아니냐이다. 빈도론자는 고정된 상수라고 보고 베이지안은 어떤 확률분포를 따르는 확률변수라고 생각한다.

예를 들어, 우리나라 사람들의 키를 수집한 데이터가 있다고 가정하자. 아무래도 우리가 관심있어 하는 parameter는 우리나라 사람들의 평균 키일 것이다. 빈도론자들은 이 평균 키($\mu$라고 하자.)가 고정된 상수(ex: 168cm)라고 가정하고 ML 방식으로 모수를 추정한다. 그에 비해, 베이지안은 $\mu$에 대한 사전 분포를 먼저 정의한 후(이를 prior belief라고 한다.) 주어진 데이터로 부터 사후분포를 추정한다. 

우리나라 사람들의 평균 키가 매우 작다고 믿는 베이지안은 평균이 160cm인 정규분포를 사전분포로 가정할 것이다. 그런데 데이터에 180cm 이상인 사람들이 많다면 데이터를 본 이후 사후분포는 평균이 175cm 정도인 정규분포가 된다. 한편, 빈도론자는 평균 키는 185cm구나!라고 결론을 지을 것이다.

글이 길어졌는데, 여튼 베이지안의 핵심은 **데이터에 사전믿음을 결합한다는 것**에 있다.

우리가 추정하고자 하는 모수를 $\theta$, 데이터를 $x$라고 할 때 결국 **베이지안의 목표는 사전분포 + 데이터로부터 사후분포를 추론하는 것이다.** 즉, 다음과 같다.

$p(\theta\vert X) = \frac{\prod_{i=1}^{n}{p(x_{i}\vert\theta)p(\theta)}}{\int {\prod_{i=1}^{n}{p(x_{i}\vert\theta)p(\theta)d\theta}}} \text{ where } x_{i}'s \text{ are i.i.d samples}$

그럼 이제 본격적으로 베이지안 입장에서 본 머신러닝 모델에 대해서 이야기해보도록 하자. $x$를 features, $y$를 class label/latent vector, $\theta$를 추정할 parameter로 정의하겠다. 그렇다면 우리가 관심있는 분포는 $x$가 given일 때 $y, \theta$의 결합 분포에 해당한다.

- $p(y, \theta \vert x) = p(y \vert \theta, x)p(\theta) \text{ } \because x \perp\theta$ 
- $p(\theta \vert X, Y) = \frac{p(Y \vert X, \theta)p(\theta)}{\int p(Y \vert X, \theta)p(\theta)}\text{ where X, Y denote whole training set}$
- test: $p(y \vert x, X, Y) = \int{ p(y \vert x, \theta)p(\theta \vert X, Y)d\theta}$

그러나 바로 여기에서 문제가 생긴다.  $p(\theta \vert X, Y)$를 구하기 위해서는 분모에 있는 적분이 가능해야 하는데, 
$p(y \vert x, \theta)$와 $p(\theta)$가 conjugate하지 않으면 대부분의 경우에서 적분이 어렵다는 것. test시에도 마찬가지.

(**[conjugate prior](https://en.wikipedia.org/wiki/Conjugate_prior)**: conjugacy에 대해서는 자세히 언급하지 않겠지만, 궁금하신 분들은 이 링크를 참조하길 바란다. 대표적인 conjugate distributions은 beta-binom, poission-gamma 등이 있다.)

여튼, conjugacy가 없다면 posterior distribution(사후분포)을 구하기가 매우 힘들고 빈도론자들 처럼 $\theta$에 대한 point estimation을 할 수 밖에 없다. 이 경우를 Poor Bayes라고도 한다고... test시에도 이러한 point estimation을 통해 얻어진 $\theta_{MP}$를 가지고 $y$에 대한 추론을 하게 된다.

- $\theta_{MP} = argmax_{\theta}p(\theta \vert X, Y) = argmax_{\theta}P(Y \vert X, \theta)p(\theta)$
- $p(y \vert x, X, Y) \approx p(y \vert x, \theta_{MP})$

덧붙여서 말하자면, 빈도론자들이 overfitting을 막기 위해 쓰는 regularization 기법(ex: L2-loss)가 사실 이 Poor Bayes와 본질적으로 동등하다.

# Variational Inference

## Main Goal: to estimate $p(\theta \vert x)$

그러면 conjugacy가 없고, 다시 말해 analytical하게 푸는 것이 불가능한 상황에서 우리는 어떻게 해야할까? 방법은 크게 두 가지로 나뉜다.

1. **variational inference**: $q(\theta) \approx p(\theta \vert x)$
2. **sampling based method**: $p(x \vert \theta)p(\theta)$로 부터 샘플링하는 방법. MCMC 등이 있으나 시간이 오래 걸린다.

우리는 여기서 첫 번째 방법인 variational inference에 대해 알아보려고 한다. approximate posterior를 가정하고, true posterior과 최대한 가깝게 approximate posterior를 추정하는 방법이다. 분포의 거리를 측정하기 위해 우리는 KL-divergence를 사용한다. KL-divergence는 워낙 유명한 토픽이고 서치하기도 쉬우니까 생략..

$\hat{q}(\theta) = argmin_{q}D_{KL}(q(\theta) \vert\vert p(\theta \vert x)) = argmin_{q} \int q(\theta)log\frac{q(\theta)}{p(\theta \vert x)}d\theta$

- 문제1: $p(\theta \vert x)$를 모른다.
- 문제2: 분포에 대한 optimization은 어떻게 할 수 있나?

**Sol)**

$logp(x) = E_{q(\theta)}[logp(x)] = \int q(\theta)logp(x)d\theta = \int q(\theta)log\frac{p(x, \theta)}{p(\theta \vert x)}d\theta= \int q(\theta)log\frac{p(x, \theta)}{p(\theta \vert x)}\frac{q(\theta)}{q(\theta)}d\theta$

 $= \int q(\theta) log\frac{p(x, \theta)}{q(\theta)}d\theta + \int q(\theta) log\frac{q(\theta)}{p(\theta \vert x)}d\theta = \mathcal{L}(q(\theta)) + D_{KL}(q(\theta) \vert\vert p(\theta \vert x))$

 따라서, $D_{KL}(q(\theta) \vert\vert p(\theta \vert x))$를 minimize하는 문제는 $\mathcal{L}(q(\theta))$를 maximize하는 문제와 동등해진다.

 $\mathcal{L}(q(\theta)) = \int q(\theta) log\frac{p(x, \theta)}{q(\theta)}d\theta = \int q(\theta) log\frac{p(x \vert \theta)p(\theta)}{q(\theta)}d\theta$

 <b><font color='red'>$= E_{q(\theta)}[logp(x \vert \theta)] - D_{KL}(q(\theta) \vert\vert p(\theta)) = \text{data likelihood + KL-regularizer term}$</font></b>

 이제 남은 부분은 $q(\theta)$를 어떻게 최적화하는지인데, 크게 두 가지 방법이 있다.

1. **[mean field approximation](https://en.wikipedia.org/wiki/Variational_Bayesian_methods)**: $\theta$끼리 독립일 때 사용하는 방법.
2. **parametric approximation**: 대부분의 neural network에서 사용하는 방법. $q(\theta)=q(\theta \vert \lambda)$라고 정의한 후 $\lambda$에 대해서 최적화.

지금까지 배운 것들을 요약해보자면 다음과 같다.

- Full Bayesian inference: $p(\theta \vert x)$
- MP inference: $\theta_{MP} = argmax_{\theta}p(\theta \vert X, Y)$
- Mean field variational inference:  $p(\theta \vert x) \approx q(\theta) = \prod_{j=1}^{m}q_{j}(\theta_{j})$
- Parametric variational inference: $p(\theta \vert x) \approx q(\theta) = q(\theta \vert \lambda)$

# Latent Variable Models

그럼 VAE를 배우기 전에 먼저 latent variable models에 대해서 짚고 넘어가자. variational inference에 대해서 신나게 공부하다가 갑자기 잠재변수모델이라니 조금 뜬금없어보이지만 VAE는 잠재변수 모델의 일종이기 때문에 반드시 짚고 넘어가야 한다.

왜 **잠재변수**를 학습해야하는가? 이미지 데이터를 예로 들어보자. RGB 채널을 갖는 32x32 짜리 이미지 데이터는 32x32x3 = 3072 차원을 갖는다. 그러나 통상적으로 생각해보았을때, 3072 차원을 통째로 다 feature로 쓰기 보다는 이미지를 결정하는 잠재변수가 있다고 보고 이를 바탕으로 추론을 하는 것이 타당하다.

예를 들어 MNIST 데이터에서 28x28=784개의 픽셀이 모두 의미있는 값이라고 보기보다는 숫자의 모양을 결정하는 변수(가장자리의 빈 정도, 선의 굽은 모양 등)가 있다고 보는 것이 맞다.

![MNIST](https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/MnistExamples.png/440px-MnistExamples.png)

잠재변수 모델을 설명하는데 가장 흔하게 쓰이는 분포가정이 **Mixture of Gaussians**이다. 즉, 여러개의 가우시안 분포가 혼합되어 있는 분포로 아래 그림과 같다.앞서 말한 대한민국 평균 키로 설명해보자면, 우리나라 사람들의 키의 분포는 남성/여성/성인/아동 등 여러 분포로 나뉠 수 있다.  ([이미지 출처](https://towardsdatascience.com/gaussian-mixture-models-explained-6986aaf5a95))

![Mixture of Gaussians](https://miro.medium.com/max/1200/1*lTv7e4Cdlp738X_WFZyZHA.png)

그럼 $i$번째 표본을 $x_{i}$라고 하고 그 표본이 속한 집단을 $z_{i}$(잠재변수)라고 해보자. 그러면 우리가 가진 데이터의 likelihood는 다음과 같이 나타낼 수 있다.

$p(X, Z \vert \theta)=\prod_{i=1}^{n}p(x_{i}, z_{i} \vert \theta) = \prod_{i=1}^{n}p(x_{i} \vert z_{i},\theta)p(z_{i} \vert \theta) = \prod_{i=1}^{n}\pi_{z_{i}} \mathcal{N}(x_{i} \vert \mu_{z_{i}}, \sigma_{z_{i}}^{2})$

여기서 $\pi_{j}=p(z_{i}=j)$로 $j$번째 그룹에 속할 확률을 의미하고 추정해야 할 파라미터는 $\theta = ( \mu_{j}, \sigma_{j}, \pi_{j} )_{j=1}^{K}$를 뜻한다.

만약 $X, Z$를 모두 안다면 $\hat{\theta} = argmax_{\theta}logP(X, Z \vert \theta)$로 쉽게 추정할 수 있겠지만 **문제는 우리는 Z를 모른다는 것이다. 따라서 우리는 $X$의 log likelihood를 최대화**하게 되고 목표식은 아래와 같다.

$logP(X \vert \theta)=\int q(Z)logP(X \vert \theta)dZ=\int q(Z) log \frac{P(X, Z \vert \theta)}{P(Z \vert \theta)} \frac{q(Z)}{q(Z)}dZ = \mathcal{L(q(Z))}+D_{KL}(q(Z) \vert\vert p(Z \vert \theta))$

항상 KL-divergence는 0 이상이므로 $logP(X \vert \theta)$의 lower-bound는 $\mathcal{L}(q(Z))$가 된다. **이를 Variational lower bound 또는 ELBO라고 칭한다.** 결국, 우리는 이 하한값을 maximize하는 $q, \theta$를 찾는 것으로 목표를 바꾸게 된다. <font color='red'>결국, 잠재변수만 추가되었을 뿐 위에서 배운 variational inference와 완전히 똑같은 문제다!</font>

이를 푸는 방법으로 **EM 알고리즘**이 존재한다. EM은 Expectation-Maximization의 약자로, 이름 그대로 Expectation step과 Maximization step이 있다.

1. E-step: $q(Z)$를 추론하는 과정으로, 이때 $\theta=\theta_{0}$으로 고정된다.  
   $q(Z) = argmax_{q}\mathcal{L}(q, \theta_{0}) = argmin_{q}D_{KL}(q(z) \vert\vert p(z \vert \theta))=p(Z \vert X, \theta_{0})$  
   자세히 풀어서 설명하자면 다음과 같다. $q(Z)$는 Multinomial 분포임을 기억하자.  
   $q(z_{i}=k)=p(z_{i}=k \vert x, \theta) = \frac{p(x_{i} \vert k, \theta)p(z_{i}=k \vert \theta)}{\sum_{l=1}^{K}p(x_{i} \vert l, \theta)p(z_{i}=l \vert \theta)}$
2. M-step: $q(Z)$를 고정시켜놓고 $\theta$를 추론하는 과정이다.  
   $\hat{\theta} = argmax_{\theta} \mathcal{L}(q, \theta) = argmax_{\theta} \mathbb{E_{Z}}[logp(X, Z \vert \theta)]=\sum_{i=1}^{n}\sum_{k=1}^{K}q(z_{i}=k)logp(x_{i}, k \vert \theta)$
3. repeat 1, 2 until convergence.

자, 여기서 드는 의문점이 있다. 위의 상황에서는 $Z$가 categorical variable이니까 단순합으로 E-step에서 $P(Z \vert X, \theta)$를 계산할 수 있다. **하지만 $Z$가 만약 continuous variable이라면? $p(x \vert z, \theta), p(z \vert \theta)$가 conjugate 하지 않다면 intractable 하게 된다!**

continuous latent variable을 학습하는 것은 dimension reduction(차원축소) 또는 **representation learning**에 해당하고 사실 머신러닝에서 매우매우 중요하면서도 어려운 부분이다. 적분으로 인한 intractable 문제를 VAE에서는 어떻게 해결하는지 다음 섹션에서 알아보겠다.

# Stochastic Variational Inference and VAE

우리는 지금까지 Bayesian framework를 이용한 variational inference와 latent variable model에 대해서 배웠다. 실제로 관측되지 않는 잠재변수를 모델링하기 위해 variational inference를 사용($q(Z)$를 추론)해 학습을 진행하는 방법이었다. 하지만 사후분포를 추론할 때 처럼 잠재변수 $Z$가 continuous 하다면 intractability 문제에 직면하게 된다. 앞서 잠깐 언급한 바와 같이 이 문제를 해결하기 위해 여러 sampling 방법들이 고안되었다. 하지만 역시 시간이 많이 걸린다. 또한 Monte Carlo로 추정한 gradient는 분산이 매우 커진다고 한다. **이런 한계점을 극복하기 위해 VAE는 reparameterization trick을 이용하였고, end-to-end learning이 가능해졌다!**

<font color='red'>지금까지와 다르게, VAE는 generative model인 동시에 representaion learning을 학습하는 모델인 것을 기억하자.</font> 즉, 우리의 목표는 두 가지다. 

1. Generation을 제대로 할 것 => $logP(X)$를 maximize하는 목표
2. Latent variable Z의 분포를 제대로 학습할 것 => $q(Z \vert X) \approx p(Z \vert X)$

먼저, 첫 번째 목표를 이루기 위해 $logP(X)$를 풀어쓰면 다음과 같다. ([이미지 출처](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture13.pdf))

![logP(X)](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbo6sRJ%2FbtqM4yIGX6T%2FjfBk3Mab5Dx4KFsi8QHeZk%2Fimg.png)

지금까지와 같이, 맨 마지막 KL-term을 제외한 나머지 것들이 lower bound가 된다. **결국, $logP(X)$를 최대화하는 목표는 lower bound를 최대화하는 목표로 바뀌고 이는 동시에 두 번째 목표까지 이루게 된다!** lower bound 식은 아래와 같다.

**$\mathcal{L}(\theta, \phi; x^{(i)}) = D_{KL}(q(z \vert x^{(i)}) \vert\vert p(z))+\mathbb{E_{q(z \vert x^{(i)})}}[logp(x^{(i)} \vert z)]$**

**앞부분은 prior과 approximate posterior와의 KL term이고, 뒷부분은 decoder probability에 해당한다.** 대부분 잠재변수 Z의 prior 분포를 $\mathcal{N}[0, 1]$와 같은 다루기 쉬운 분포로 정한다. 그러면 $q(z \vert x)$는 어떻게 정의했을까? VAE original paper에서는 다변량 정규분포로 정의하는데, 다음과 같다.

$q(z_{i} \vert x_{i}, \phi) = \prod_{j=1}^{d}\mathcal{N}[\mu_{j}(x_{i}), \sigma_{j}^{2}(x_{i})]$

이때 $\mu_{j}(x_{i}), \sigma_{j}^{2}(x_{i})$는 $x_{i}$가 DNN을 통과한 output에 해당한다. 그래서 구현된 코드를 보면 알겠지만, VAE의 encoder에서는 $\mu_{j}(x_{i}), \sigma_{j}^{2}(x_{i})$를 구한다. 그러면 $p(z), q(z \vert x)$의 KL-divergence를 구할 수 있게 된다. (둘 다 정규분포이므로) **사실 이 term은 approximate posterior가 prior와 너무 달라지지 않게 하는 regularizer 역할을 해준다.**

decoder probability에 해당하는 뒷부분을 보면 $q(z \vert x)$에 기반하여 $log(x \vert z)$의 평균을 구해야 한다. 바로 여기서 intractability에 직면한다. 앞서 말했다시피 Monte Carlo 방법으로 평균을 추정하게 되면 gradient의 분산이 매우 커지는 동시에 수렴할 때까지 시간이 오래걸리는 문제가 있다. **게다가 무엇보다도, sampling은 미분가능한 연산이 아니기 때문에 역전파로 학습할 수가 없게 된다.** VAE의 저자들을 똑똑하게도, **reparameterization trick**을 이용했다.

<font color='red'>$q_{\phi}(z \vert x) \rightarrow g(\epsilon, x)$
</font>

사실 이 수식이 reparam trick의 전부인데, 처음에는 수식만 보고 읭?했었다. 그런데 회귀분석의 문제로 이해하면 쉬운 문제다.

간단하게 언급하자면, $y$변수(타겟변수)가 $x$변수(feature)와 linear한 관계에 있다고 가정하고 $y = ax+b+\epsilon$식에서 $a, b$를 푸는 것인데 결국 이는 $p(y \vert x)$를 구하는 태스크가되고 $x$는 given, $a, b$는 constant라고 가정하기 때문에 random factor은 $\epsilon$ ~ $N(0, 1)$에서만 생긴다. 즉, $p(y \vert x)$는 $ax+b$를 평균으로하고 1을 분산으로 하는 정규분포가 된다. 따라서 $a, b$는 MLE 방법으로 closed-form solution이 나오게 된다. 지금까지 설명한 VAE와 개념적으로 상당히 비슷함을 알 수 있다.

결국 $g(\epsilon, x)$는 본인은 deterministic한 function인데 외부에서 noise $\epsilon$이 들어왔다고 이해하게 되고, 미분이 가능해진다. **end-to-end learning이 가능해지는 것이다!**

마지막으로 VAE의 단점인 blurry generation을 짚고 넘어가려고한다. approximate posterior가 regularizer 역할을 하고, reconstruction loss가 실제 cost에 해당한다고 볼 수 있기 때문에 $logp(x \vert z)$를 높이는 방향으로 학습이 된다. 이는 일종의 Linear Regression(MLE)으로 볼 수 있고, 결국 $x$의 평균과 가까워지게 된다. 따라서 VAE로 생성된 이미지는 보다 흐리다.

VAE로 학습된 Z를 통해 이미지를 생성한 결과는 다음과 같다. ([이미지 출처](https://arxiv.org/pdf/1312.6114.pdf))

D=2인 Z축에서 매우 smooth하게 변하고 있음을 볼 수 있다.

---



## Generative Adversarial Networks  (GANs)

- The ultimate goal of GANs is generating data approximating real data distribution.

- Take game-theoretic approach, learn to generate from training distribution through 2-player game.  But can be tricky and unstable to train, no inference queries such as $p(x)$, $p(z|x)$.

![lecture13-05_realOrFake](.\images\lecture13-05_realOrFake.png) Fake and real images [[1](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture13.pdf)]

- Problem: Want to sample from complex, high-dimensional training distribution. No way to do this.
- Solution: Sample from a simple distribution, e.g. random noise. Learn transformation to training distribution.



### Training GANs: Two-player game

- Minmax objective function:

![lecture13-06_GAN_objectiveFunction1](.\images\lecture13-06_GAN_objective_function1.png) Minmax objective loss function [[1](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture13.pdf)]

- Generator($\theta_g$) network: try to fool the discriminator by generating real-looking images
  
  - Generator($\theta_g$) wants to minimize objective such that D(G(z)) is close to 1 (discriminator is fooled into thinking generated G(z) is real).
- Discriminator($\theta_d$) network: try to  distinguish between real and fake images
  - Discriminator($\theta_d$) wants to maximize objective such that D(x) is close to 1 (real) and D(G(Z)) is close to 0 (fake).
  - Discriminator outputs likelihood in (0,1) of real image

- Gradient ascent and descent of GANs in practice

  1. **Gradient ascent** on discriminator:

     ![lecture13-08_GAN_objective_function2](.\images\lecture13-08_GAN_objective_function2.png) 

  2. **Gradient descent** on generator in origin:

     ![lecture13-09](.\images\lecture13-09_GAN_objective_function3.png)

     - In practice, optimizing the generator objective function does not work well.

     ![11_generator_gradient_descent](.\images\lecture13-11_generator_gradient_descent.png) image reference: [[1](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture13.pdf)]

     - When sample is likely fake, want to learn from it to improve generator. But gradient in this region is relatively flat.

     - Gradient signal dominated by region where sample is already good.

  3. **Gradient ascent** on generator **in standard practice (Instead of the "2. Gradient descent on generator in origin"):**![lecture13-12](.\images\lecture13-12.png) image reference: [[1](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture13.pdf)]

     - Instead of minimizing likelihood of discriminator being correct, now maximize likelihood of discriminator being wrong.

     ![lecture13-13](.\images\lecture13-13.png) image reference: [[1](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture13.pdf)]  

     - Same objective of fooling discriminator, but now higher gradient signal for bad samples. So it works better.

     - Jointly training two networks is challenging, can be unstable.
       - Choosing objectives with better loss landscapes helps training.



### Generative model with KL-divergence [9]

Generative models create a model $\theta$ that maximizes the maximum likelihood estimation (MLE). to find the best model parameters that fit the training data the most.

![lecture13-16](.\images\lecture13-16.png)

- This is the same as minimizing the KL-divergence $KL(p,q)$ which measures how the estimated probability distribution $q$ diverges from the real world expected distribution p. ([proof in detail](https://jhui.github.io/2017/01/05/Deep-learning-Information-theory/))

![lecture13-17](.\images\lecture13-17.png)

- KL-divergence is not symmetrical.

![lecture13-18](.\images\lecture13-18.png)

- The KL-divergence $DL(p,q)$ penalizes the generator if it misses some modes of images: the penalty is high where $p(x) > 0$ but $q(x) → 0$. Nevertheless, it is acceptable that some images do not look real. The penalty is low when $p(x) → 0$ but $q(x) > 0$. **(Poorer quality but more diverse samples)**

  ![lecture13-19:  probability density function of p and q (left), KL-divergence of p and q (right) ](.\images\lecture13-19.png) Figure: probability density function of p and q (left), KL-divergence of $p$ and $q$ (right) [9]



### GANs problems

- **Non-convergence**: the model parameters oscillate, destabilize and never converge.
- **Mode collapse**: the generator collapses which produces limited varieties of samples.
- **Diminished gradient**: the discriminator gets too successful that the generator gradient vanishes and learns nothing.
- Unbalance between the generator and discriminator causing overfitting.
- Highly sensitive to the hyper parameter selections.



### Mode collapse

**Mode collapse** refers to the phenomenon that the model we are trying to train does not cover all the distribution of the actual data and loses diversity. This is a case where *G* cannot find the entire data distribution because it is only learning to reduce the loss, and it is strongly concentrated in only one *mode* at a time as shown in the figure below. For example, this is the case where *G* trained on MNIST generates only certain numbers. [7]

![lecture13-14](.\images\lecture13-14.png)The problem that the probability density functions of generator and discriminator are alternatively vibrating without converging is related to mode collapse [7]

![lecture13-15](.\images\lecture13-15.png) mode collapse example [7], [9]



### Solutions of mode collapse problem

The key to solve the model collapse is to train the model to learn the boundaries of the entire data distribution evenly and keep it remembered.

- **feature matching** : Add **least square error** between fake data and real data to the objective function 
- **mini-batch discrimination** : Add the sum of distance difference between fake data and real data for each *mini-batch* to the objective function.

- **historical averaging** : Update the loss function to incorporate history.



## Conclusion

![conclusion of lecture 13](.\images\lecture13-20.png)



## References

- [1] [cs231n 2017 lecture13](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture13.pdf)
- [2] [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
- [3] [Pixel RNN](https://arxiv.org/pdf/1601.06759.pdf)
- [4] [Pixel CNN](https://arxiv.org/abs/1606.05328v2), [Pixel CNN v2](https://arxiv.org/abs/1606.05328)
- [5] https://towardsdatascience.com/auto-regressive-generative-models-pixelrnn-pixelcnn-32d192911173
- [6] [cs231n 2020 lecture11](http://cs231n.stanford.edu/slides/2020/lecture_11.pdf)
- [7] [mode collapse in GANs](https://ratsgo.github.io/generative%20model/2017/12/20/gan/)
- [8] [developers.google.com: mode collapse](https://developers.google.com/machine-learning/gan/problems)
- [9] solutions of mode collapse [https://jonathan-hui.medium.com/gan-why-it-is-so-hard-to-train-generative-advisory-networks-819a86b3750b#4987]