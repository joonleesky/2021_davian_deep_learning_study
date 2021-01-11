## Information

- **Title:** (cs229) Lecture 2: Linear Regression and Gradient Descent
- **Link:** http://cs229.stanford.edu/notes2020fall/notes2020fall/cs229-notes1.pdf
- **Keywords:** Machine Learning, Bayesian Inference, Maximum Liklihood Estimation, Linear Regression


### Machine Learning
* Definition of Machine Learning  
   - Finding a model and its parameters so that the resulting predictor performs well on unseen data
   
   
### Linear Regression
   
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  **Cost function**: ![cost function](https://latex.codecogs.com/gif.latex?J%28%5Ctheta%29%20%3D%20%5Cfrac%7B1%7D%7B2%7D%20%5Csum_%7Bi%20%3D%201%7D%5E%7Bn%7D%28h_%5Ctheta%28x%5E%7B%28i%29%7D%29%20-%20y%5E%7B%28i%29%7D%29)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (1)    
 where ![theta](https://latex.codecogs.com/gif.latex?%5Ctheta) are parameters, ![x](https://latex.codecogs.com/gif.latex?x) are training examples, and ![y](https://latex.codecogs.com/gif.latex?y) are targets.  
   
* Section 1: LMS algorithm  

   **Gradeint Descent**: ![gradient descent](https://latex.codecogs.com/gif.latex?%5Ctheta_j%20%3A%3D%20%5Ctheta_j%20-%20%5Calpha%20%5Cfrac%7B%5Cpartial%20%7D%7B%5Cpartial%20%5Ctheta_j%7DJ%28%5Ctheta%29). This becomes ![gradient descent2](https://latex.codecogs.com/gif.latex?%5Ctheta_j%20%3A%3D%20%5Ctheta_j%20&plus;%20%5Calpha%28y%5E%7B%28i%29%7D-h_%5Ctheta%20%28x%5E%7B%28i%29%7D%29%29x_j%5E%7B%28i%29%7D). This is called batch gradient descent becuase you are using entire training set.  
On the other hand, if you update the following way,  
```
Loop{  
      for i = 1 to n,  
      {
 ```
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  ![sto gd](https://latex.codecogs.com/gif.latex?%5Ctheta_j%20%3A%3D%20%5Ctheta_j%20&plus;%20%5Calpha%28y%5E%7B%28i%29%7D-h_%5Ctheta%20%28x%5E%7B%28i%29%7D%29%29x_j%5E%7B%28i%29%7D)&nbsp;&nbsp;&nbsp; (for every j)
```
      }  
   }  
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;you're using stochastic gradient descent. 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;But you have to update parameters at the same time, i.e., you can't update the first element of parameters before updating a second parameter. 

* Section 2: The normal equations  
   Using matrix, you can also transform (1) into ![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B1%7D%7B2%7D%28X%5Ctheta%20-%20y%29%5ET%28X%5Ctheta%20-%20y%29)  
   Then you can take gradient with respect to ![](https://latex.codecogs.com/gif.latex?%5Ctheta) and get ![](https://latex.codecogs.com/gif.latex?%5Ctheta) that minimizes the cost function.  
   ![](https://latex.codecogs.com/gif.latex?%5Cnabla_%5Ctheta%20J%28%5Ctheta%29%20%3D%20X%5ETX%5Ctheta%20-%20X%5ETy%5C%5C%5C%5C%20%5Cindent%20X%5ETX%5Ctheta%20%3D%20X%5ETy%5C%5C%5C%5C%20%5Cindent%20%5Ctheta%20%3D%20%28X%5ETX%5Ctheta%29%5E%7B-1%7DX%5ETy)
   
   
* Section 3: Probabilistic Interpretation
    * P(Y_test|X_text, D)
    

