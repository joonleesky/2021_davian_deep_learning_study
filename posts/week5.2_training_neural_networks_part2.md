# DAVIAN Lab. Deep Learning Winter Study (2021)

- **Writer:** Seungwoo Ryu

## Information

- **Title:** (cs231n) Lecture 7 : Training Neural Networks, part II  
- **Link:** http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture7.pdf  
- **Keywords:**  Update rules, ensembles, data augmentation, transfer learning
-------------------------------------------------------  

## Update rules (Optimization)  

```  
1. SGD
2. SGD + Momentum  
3. Nesterov  
4. Adagrad  
5. RMSProp  
6. Adam  
7. Appendix  
```  

#### Introducing Optimization  
![1](https://user-images.githubusercontent.com/43376853/107843558-83d0dc00-6e0f-11eb-91a0-384080071932.png)  
- Goal: Finding the most red region  
  (Each color of this graph represents the value of loss, and the most red region is the spot recording the lowest loss.)  
- However, relatively simple optimization algorithm such as SGD has quite a lot of problems.  

---  

### 1.SGD  
- Update rules  
  ![2](https://user-images.githubusercontent.com/43376853/107843734-cc3cc980-6e10-11eb-9e30-9b71b910cfc7.png)  

#### Problems of SGD  
- `Problem1`. Occurrence of learning speed difference along the different direction  
  ![3](https://user-images.githubusercontent.com/43376853/107843846-b8de2e00-6e11-11eb-9147-550055255022.png)  
  - SGD causes 'zig-zag' shaped unstable learning.  
  - Unstable 'zig-zag' behavior deepens as dimension increases.  
  
- `Problem2`: Stucking in __local minima__ or __saddle point__   
  ![4](https://user-images.githubusercontent.com/43376853/107843896-50438100-6e12-11eb-90ae-604acbc24b59.png)  
  - Far way from 'real' optimum point  
  
    ||Local Minima|Saddle Point|  
    |-|------------|------------|
    |Frequent in|Low-dimension|High-dimension|   

  
- `Problem3`: Too noisy movement  
  ![5](https://user-images.githubusercontent.com/43376853/107843960-ea0b2e00-6e12-11eb-9ae8-2a8ca941a8e8.png)  
  - Using whole training data at computing loss is very expensive.  
  - Instead we uses minibatches at computing loss, so it becomes noisy.  
    It is not actually getting the true information about the gradient at every time step, but getting some noisy estimate of the gradient at current point.  

---  

### 2.SGD + Momentum  
- Update rules  
  ![6](https://user-images.githubusercontent.com/43376853/107844134-47ec4580-6e14-11eb-9bfe-2048b566f236.png)  
  - Start to consider `velocity` as a running mean of gradients  
  - ρ: friction (usually 0.9 or 0.99)  
  
- It alleviates `poor conditioning` problem, and more stable than vanilla SGD.  
  ![8](https://user-images.githubusercontent.com/43376853/107844273-5f77fe00-6e15-11eb-969a-0649f91a5aba.png)  
  
---    
  
### 3.Nesterov   
- Update rules  
  - Original Version  
    ![9](https://user-images.githubusercontent.com/43376853/107844304-9a7a3180-6e15-11eb-9468-b96013fcd3b2.png)  
      
    - For the convenience of updating, introduced improved version  
  - Improved Version   
    ![10](https://user-images.githubusercontent.com/43376853/107844339-db724600-6e15-11eb-9bd2-d60c1c94a100.png)  

- It has nice theoretial property in terms of convex optimization, but a Neural Network is non-convex environment.  
- It shows similar movement in terms of `overshooting` behavior.  
  ![11](https://user-images.githubusercontent.com/43376853/107844453-d82b8a00-6e16-11eb-8851-4e3daaf42b10.png)  

---  

### 4.Adagrad  
- Update rules  
  ![12](https://user-images.githubusercontent.com/43376853/107844560-b252b500-6e17-11eb-9e72-f33ab03124c3.png)  
  - Now, it considers grad-squared term instead of velocity.  
  
- Similar as 'Nesterov', it shows nice property in case of convex environment, but not in the non-convex environment case.  

#### Problem of Adagrad  
- Problem arises from `np.sqrt(grad_squared) + 1e-7)` term  
  - As time goes by, grad_squared term is accumulated. → Stepsize keep decreases.  
  - RMSProp addressed this problem.  
  
---  

### 5.RMSProp  
- Update rules  
  ![13](https://user-images.githubusercontent.com/43376853/107844654-a74c5480-6e18-11eb-8213-cc96be5a03c5.png)  
  - It lets the squared estimate actually decay.  
  - It solves the problem suggested in Adagrad, greatly.  
  
- It greatly adjusts its trajectory such a way that making approximately equal progress among all the dimensions.  
  ![14](https://user-images.githubusercontent.com/43376853/107844688-0f9b3600-6e19-11eb-84ae-ff15ed2ba294.png)  
  
---  
 
#### Before introducing Adam...  
- `SGD with momentum` & `Nesterov` consider kind of `velocity` when updating the gradient.  
  → Showed kind of `overshooting` movement  
- `AdaGrad` & `RMSProp` consider `grad-squared` term when updating the gradient.  
  → Showed kind of `adjusting` movement  
  
- Those two considerations are pretty nice!  
  Then, why don't we consider them both at the same time?  → `Adam` is devised.  

### 6.Adam   
- Update rules  
  ![15](https://user-images.githubusercontent.com/43376853/107844816-1bd3c300-6e1a-11eb-8114-4871f135714b.png) 2=0.999, learning_rate=1e-3 or 5e-4 is a great starting point!  
  - Considered momentum and grad-squared term at the same time!  
  - With bias correction, `first_unbias` and `second_unbias` become unbiased estimator of 1st and 2nd moment.     
  - β1=0.9, β2=0.999, learning_rate=1e-3 or 5e-4 is a great starting point!  

- Optimization movement reflects on both characteristics of momentum-based methods and grad-squared-based methods  
  ![16](https://user-images.githubusercontent.com/43376853/107844915-e4194b00-6e1a-11eb-91aa-074c091a77f5.png)  

---  

### 7.Appendix  
- All algorithms 1~6 have `learning rate` as hyperparameter.  
- All algorithms 1~6 are specific example of `First-Order Optimization`.  

#### First-Order Optimization  
- Gradient updates happens through 'Linear' approximation.  
  ![17](https://user-images.githubusercontent.com/43376853/107845047-fd6ec700-6e1b-11eb-991e-cebfaaf89329.png)  
  
#### Second-Order Optimization  
- Gradient updates happens through 'Quadratic' Approximation    
  ![18](https://user-images.githubusercontent.com/43376853/107845048-fe9ff400-6e1b-11eb-9b38-42c4342eaf51.png)  
  
  ![19](https://user-images.githubusercontent.com/43376853/107845049-ff388a80-6e1b-11eb-9aa0-23ec2c7c4c40.png)  
  
  Second-order optimization is really nice optimization method for two reasons:  
  1. J(θ) can be calculated easily using Taylor expansion.  
  2. No hyperparameters such as learning rate are needed. (Closed form Newton parameter update is enough.)  
  
  In spite of these conveniences, second-order optimization is not really practical in Deep Learning domain.  
  It's because computing a Hessian matrix requires O(N^2) complexities, and computing the inverse matrix of Hessian metrix requires O(N^3) complexities, meaning that computer gets too much burden.  
  
  Of course, other variants of second-order optimization exist such as `Quasi-Newton methods` and `L-BFGS`.   
  

- Adam is a good default choice in most cases.  
- L-BFGS can be tried if full batch update is affordable.  

---  

## Ensembles  
- Model Ensembling is a good way to decrease the performance gap between train set and validation set.  
- It trains multiple independent models and average their results at test time.  

- However there are a lot of considerations for performance improvement using a single model rather than using multiple models.  
  ```  
  1. Using multiple snapshots of a single model during training time.  
  2. Use Polyak averaging
  (3. Methods such as Deep Ensemble / SWA are also good alternatives.)  
  ```  
---  

## Regularization  
```  
1.Adding term to loss  
2.Dropout  
3.Data Augmentation  

- DropConnect, Fractional Max Pooling, Stochastic Depth etc.
```   

### 1.Adding term to loss  
- It was mentioned on earlier lectures.  
  ![20](https://user-images.githubusercontent.com/43376853/107845576-cc909100-6e1f-11eb-971a-3f8146b730eb.png)  
  
### 2.Dropout  
#### At training time  
- At 'forward' pass, it randomly 'turns-off' the neurons per each layer by probability 'p'.  
  p=0.5 is common.  
  ![21](https://user-images.githubusercontent.com/43376853/107845621-242efc80-6e20-11eb-88c9-8d6a4cc890ed.png)  
  
- Two Interpretations of Dropout  
  1. When turning-off some neurons, it excludes a redundant representation of features.  
    It prevents co-adaptation of features.  
  2. In some ways, dropout is a large ensemble of models.   
  
#### At test time  
- Little bit different from dropout at training time.  
- It is undesirable to assign a randomness at test time.  
- When updating during test time, just `multiply` by dropout probability.  
  ![22](https://user-images.githubusercontent.com/43376853/107845717-1d54b980-6e21-11eb-94d7-fd05eb0f27cb.png)  
  
- Idea at test time is similar with that of `Batch Normalization`.  


### 3.Data Augmentation  
- Kind of tweak that is augmenting the number of train dataset by transforming the original data with same labels.  
  ![24](https://user-images.githubusercontent.com/43376853/107845906-838e0c00-6e22-11eb-9ee4-f6efad748913.png)

- A lot of skills which can be considered  
  - Horizontal flips  
  - Random crops & scales  
  - Color Jitter  
  - Translation  
  - Rotation  
  - Strecthing  
  - Shearing  
  - lens distortions, ...  
  - etc.  
  
---  

## Transfer Learning  
- ["Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task."](https://machinelearningmastery.com/transfer-learning-for-deep-learning/)  

- Example for transfer learning with CNNs  
  ![25](https://user-images.githubusercontent.com/43376853/107845956-0747f880-6e23-11eb-8a4b-4cc117011c93.png)  
  
  - The model on the left is the model already trained on Imagenet.  
  - For my own task, I can use this model as a good starting point.  
  - Just randomly re-initialize and train the top-layer, freezing other layers.  
  - It works pretty well.  
  
- The performance of Transfer Learning is decided according to the characteristics of my own dataset.  
  Need to consider two stuffs: the number of data, similarity of my data with data for pre-trained model.  
  Table below is prescription for this problem.  
  ![26](https://user-images.githubusercontent.com/43376853/107846043-dcaa6f80-6e23-11eb-85e2-f8a65bf9cc00.png)  
  
- Of course, according to tasks, we can use more than two pre-trained models.  
  ![27](https://user-images.githubusercontent.com/43376853/107846073-1da28400-6e24-11eb-8710-ca82e00a2027.png)  
  
- Using a pre-trained model is pervasive.  

---  

## Reference & Further readings
- https://machinelearningmastery.com/transfer-learning-for-deep-learning/  
- https://arxiv.org/abs/1912.02757  
- https://arxiv.org/abs/1803.05407  

