## Information

- **Title:** (cs229) Lecture 2: Linear Regression and Gradient Descent
- **Link:** http://cs229.stanford.edu/notes2020fall/notes2020fall/cs229-notes1.pdf
- **Keywords:** Machine Learning, Bayesian Inference, Maximum Liklihood Estimation, Linear Regression


### Machine Learning
* Definition of Machine Learning:  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Finding a model and its parameters so that the resulting predictor performs well on unseen data  
    
   ![](https://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20p%28y%27%20%5Cmid%20x%27%2CD%29%20%26%3D%20%5Cint%20p%28y%27%2C%5Ctheta%20%5Cmid%20x%27%2CD%29%20d%5Ctheta%5C%5C%20%26%3D%20%5Cint%20%5Cfrac%7Bp%28y%27%20%5Cmid%20x%27%2C%5Ctheta%2C%20D%29%20%5Ccdot%20p%28%5Ctheta%20%5Cmid%20x%27%2CD%29%20%5Ccdot%20p%28D%20%5Cmid%20x%27%29%7D%7Bp%28D%20%5Cmid%20x%27%29%7D%20d%5Ctheta%5C%5C%20%26%3D%20%5Cint%20p%28y%27%20%5Cmid%20x%27%2C%20D%3B%20%5Ctheta%29%20%5Ccdot%20p%28%5Ctheta%20%5Cmid%20x%27%2C%20D%29%20d%5Ctheta%5C%5C%20%26%3D%20%5Cint%20p%28y%27%5Cmid%20x%27%3B%5Ctheta%29%20%5Ccdot%20p%28%5Ctheta%20%5Cmid%20D%29%20d%5Ctheta%5C%5C%20%26%3D%20E_%7B%5Ctheta%20%5Csim%20p%28%5Ctheta%20%5Cmid%20D%29%7D%5Bp%28y%27%20%5Cmid%20x%27%3B%5Ctheta%29%5D%20%5Cend%7Baligned%7D)  
   **Estimation:**  
   
   ![](https://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20%5Ctheta_%7B%5Ctext%20%7BMAP%7D%7D%20%26%3D%20%5Cunderset%7B%5Ctheta%7D%7B%5Coperatorname%7Bargmax%7D%7D%5C%3Bp%28%5Ctheta%20%5Cmid%20D%29%5C%5C%20%26%3D%20%5Cunderset%7B%5Ctheta%7D%7B%5Coperatorname%7Bargmax%7D%7D%5C%3B%20%5Cfrac%7Bp%28D%5Cmid%20%5Ctheta%29%5Ccdot%20p%28%5Ctheta%29%7D%7Bp%28D%29%7D%5C%5C%20%26%3D%20%5Cunderset%7B%5Ctheta%7D%7B%5Coperatorname%7Bargmax%7D%7D%5C%3B%20p%28D%5Cmid%20%5Ctheta%29%5Ccdot%20p%28%5Ctheta%29%5C%5C%20%26%3D%20%5Cunderset%7B%5Ctheta%7D%7B%5Coperatorname%7Bargmax%7D%7D%5C%3B%20p%28y%5Cmid%20x%3B%20%5Ctheta%29%20%5Ccdot%20p%28%5Ctheta%29%5C%5C%20%5Ctheta_%7B%5Ctext%20%7BMLE%7D%7D%26%3D%20%5Cunderset%7B%5Ctheta%7D%7B%5Coperatorname%7Bargmax%7D%7D%5C%3B%20p%28y%20%5Cmid%20x%3B%5Ctheta%29%20%5C%5C%20%26%3D%20%5Cunderset%7B%5Ctheta%7D%7B%5Coperatorname%7Bargmax%7D%7D%5C%3B%20p%28y%5Cmid%20x%3B%5Ctheta%29%5C%5C%20%26%3D%20%5Cunderset%7B%5Ctheta%7D%7B%5Coperatorname%7Bargmin%7D%7D%5C%3B%20-%5Ctext%20%7Blog%7D%20%5C%3Bp%28y%5Cmid%20x%3B%5Ctheta%29%5C%5C%20%26%3D%20%5Cunderset%7B%5Ctheta%7D%7B%5Coperatorname%7Bargmin%7D%7D%5C%3B%20-%5Cfrac%7B1%7D%7B2%5Csigma%5E2%7D%28y-X%5Ctheta%29%5ET%28y-X%5Ctheta%29%20%5Cend%7Baligned%7D)
   
### Linear Regression
   
* **Cost function**: ![cost function](https://latex.codecogs.com/gif.latex?J%28%5Ctheta%29%20%3D%20%5Cfrac%7B1%7D%7B2%7D%20%5Csum_%7Bi%20%3D%201%7D%5E%7Bn%7D%28h_%5Ctheta%28x%5E%7B%28i%29%7D%29%20-%20y%5E%7B%28i%29%7D%29)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **(1)**  
where ![theta](https://latex.codecogs.com/gif.latex?%5Ctheta) are parameters, ![x](https://latex.codecogs.com/gif.latex?x) are training examples, and ![y](https://latex.codecogs.com/gif.latex?y) are targets.  
   
* **Section 1: LMS algorithm**  

   * **Gradeint Descent**: ![gradient descent](https://latex.codecogs.com/gif.latex?%5Ctheta_j%20%3A%3D%20%5Ctheta_j%20-%20%5Calpha%20%5Cfrac%7B%5Cpartial%20%7D%7B%5Cpartial%20%5Ctheta_j%7DJ%28%5Ctheta%29). This becomes ![gradient descent2](https://latex.codecogs.com/gif.latex?%5Ctheta_j%20%3A%3D%20%5Ctheta_j%20&plus;%20%5Calpha%28y%5E%7B%28i%29%7D-h_%5Ctheta%20%28x%5E%7B%28i%29%7D%29%29x_j%5E%7B%28i%29%7D). This is called batch gradient descent becuase you are using an entire training set.  
On the other hand, if you update the following way, you're using stochastic gradient descent. But you have to update parameters at the same time, i.e., you can't update the first element of parameters before updating a second parameter.   
```
        Loop{  
              for i = 1 to n,  
                  {
 ```
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  ![sto gd](https://latex.codecogs.com/gif.latex?%5Ctheta_j%20%3A%3D%20%5Ctheta_j%20&plus;%20%5Calpha%28y%5E%7B%28i%29%7D-h_%5Ctheta%20%28x%5E%7B%28i%29%7D%29%29x_j%5E%7B%28i%29%7D)&nbsp;&nbsp;&nbsp; (for every j)
```
                  }//for  
            }//loop  
```


* **Section 2: The normal equations**  
   Using matrix, you can also transform **(1)** into ![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B1%7D%7B2%7D%28X%5Ctheta%20-%20y%29%5ET%28X%5Ctheta%20-%20y%29)  
   Then you can take gradient with respect to ![](https://latex.codecogs.com/gif.latex?%5Ctheta) and get ![](https://latex.codecogs.com/gif.latex?%5Ctheta) that minimizes the cost function.  
   ![](https://latex.codecogs.com/gif.latex?%5Cnabla_%5Ctheta%20J%28%5Ctheta%29%20%3D%20X%5ETX%5Ctheta%20-%20X%5ETy%5C%5C%5C%5C%20%5Cindent%20X%5ETX%5Ctheta%20%3D%20X%5ETy%5C%5C%5C%5C%20%5Cindent%20%5Ctheta%20%3D%20%28X%5ETX%5Ctheta%29%5E%7B-1%7DX%5ETy)
   
   
* **Section 3: Probabilistic Interpretation**  
When approaching regression problem, why bother using specifically the least square function J?  
Let's redefine the relation between the inputs and target varaibles as the following: ![](https://latex.codecogs.com/gif.latex?y%5E%7B%28i%29%7D%20%3D%20%5Ctheta%5ETx%5E%7B%28i%29%7D&plus;%5Cepsilon%5E%7B%28i%29%7D), where ![](https://latex.codecogs.com/gif.latex?%5Cepsilon%5E%7B%28i%29%7D) is error term that represents, e.g. random noise. We assume the error follows Normal distrubution (or Gaussian distribution).  
Then we can that ![](https://latex.codecogs.com/gif.latex?p%28%5Cepsilon%5E%7B%28i%29%7D%29%20%3D%20p%28y%5E%7B%28i%29%7D%20%5Cmid%20x%5E%7B%28i%29%7D%3B%20%5Ctheta%29%20%3D%20%5Cfrac%7B1%7D%7B%5Csqrt%7B2%5Cpi%7D%5Csigma%7D%20e%5E%7B-%5Cfrac%7B%28y%5E%7B%28i%29%7D%20-%20%5Ctheta%5ETx%5E%7B%28i%29%7D%29%5E2%7D%7B2%5Csigma%5E2%7D%7D) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **(2)**  
Interpreting **(2)** as a function of ![](https://latex.codecogs.com/gif.latex?%5Ctheta), we can instead call it the likelihood function:  
![](https://latex.codecogs.com/gif.latex?L%28%5Ctheta%29%20%3D%20L%28%5Ctheta%3BX%2Cy%29%20%3D%20%5Cprod_%7Bi%3D1%7D%5E%7Bn%7Dp%28y%5E%7B%28i%29%7D%20%5Cmid%20x%5E%7B%28i%29%7D%3B%5Ctheta%29%29)  
According to maximum likelihoold, we should choose ![](https://latex.codecogs.com/gif.latex?%5Ctheta) that make the data as high probability as possible. For the convenience of calculation, we use log likelihood as the following:  
![](https://latex.codecogs.com/gif.latex?l%28%5Ctheta%29%20%3D%20logL%28%5Ctheta%29%20%3D%20nlog%5Cfrac%7B1%7D%7B%5Csqrt%7B2%5Cpi%7D%5Csigma%7D%20-%20%5Cfrac%7B1%7D%7B%5Csigma%5E2%7D%5Ccdot%20%5Cfrac%7B1%7D%7B2%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28y%5E%7B%28i%29%7D%20-%20%5Ctheta%5ETx%5E%7B%28i%29%7D%29%5E2)  
To make it the maximum, we need to minimize ![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B1%7D%7B2%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28y%5E%7B%28i%29%7D%20-%20%5Ctheta%5ETx%5E%7B%28i%29%7D%29%5E2)


