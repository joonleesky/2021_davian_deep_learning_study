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
```  

#### Introducing Optimization  
![1](https://user-images.githubusercontent.com/43376853/107843558-83d0dc00-6e0f-11eb-91a0-384080071932.png)  
- Goal: Finding the most red region  
  (Each color of this graph represents the value of loss, and the most red region is the spot recording the lowest loss.)  
- However, relatively simple optimization algorithm such as SGD has quite a lot of problems.  

---  

### SGD  
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

### SGD + Momentum  
- Update rules  
  ![6](https://user-images.githubusercontent.com/43376853/107844134-47ec4580-6e14-11eb-9bfe-2048b566f236.png)  
  - Start to consider `velocity` as a running mean of gradients  
  - ρ: friction (usually 0.9 or 0.99)  
  
- It alleviates `poor conditioning` problem, and more stable than vanilla SGD.  
  ![8](https://user-images.githubusercontent.com/43376853/107844273-5f77fe00-6e15-11eb-969a-0649f91a5aba.png)  
  
---    
  
### Nesterov   
- Update rules  
  - Original Version  
    ![9](https://user-images.githubusercontent.com/43376853/107844304-9a7a3180-6e15-11eb-9468-b96013fcd3b2.png)  
      
    For the convenience of updating, introduced improved version  
  - Improved Version   
    ![10](https://user-images.githubusercontent.com/43376853/107844339-db724600-6e15-11eb-9bd2-d60c1c94a100.png)  

- It has nice theoretial property in terms of convex optimization, but a Neural Network is non-convex environment.  
- It shows similar movement in terms of `overshooting` behavior.  
  ![11](https://user-images.githubusercontent.com/43376853/107844453-d82b8a00-6e16-11eb-8851-4e3daaf42b10.png)  

---  

### Adagrad  



  


