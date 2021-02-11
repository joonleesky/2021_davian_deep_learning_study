# DAVIAN Lab. Deep Learning Winter Study (2021)

- **Writer:** Sunjun Kweon

## Information

- **Title:** (cs231n) Lecture 4 : Introduction to Neural Networks 
- **Link:** http://cs231n.stanford.edu/slides/2020/lecture_4.pdf
- **Keywords:** Neural Networks, Jacobians, Backpropagation
-------------------------------------------------------

## Neural Network

- Motivation : Linear classifers are not very powerful. It is hard to classify things which cannot be divided by a single line(hyperplane)
![1](https://user-images.githubusercontent.com/59158426/106470232-98de6e80-64e3-11eb-85c2-0257ecc74dde.PNG)

- Stacking multiple layers with non-linearity expresses more

![2](https://user-images.githubusercontent.com/59158426/106471208-bf50d980-64e4-11eb-8093-1c9ecde09bbe.PNG)
- Instead of just having linear score **s=W1x** 2-layer neural network's score is **s=W2f(W1x)**
- f which contributes to non-linearity is called activation function. Below are popular choices for activation functions
![3](https://user-images.githubusercontent.com/59158426/106471511-0b038300-64e5-11eb-82e1-dd38e0bbd6ab.PNG)

## Jacobian

-Consider a vector function f(x)=(f1(x),f2(x),...fm(x)). Consider a small change Δx.

 f(x+Δx)=[f1(x+Δx),f2(x+Δx),...fm(x+Δx)]=f(x)+[∇f1(x),...∇fm(x)]TΔx
 
 The Jacobian of f at x is 
 
![image](https://user-images.githubusercontent.com/59158426/106474024-d0e7b080-64e7-11eb-9ff5-a4aceaa35250.png)

-Example : R^m to R^n (y=Ax)

![5](https://user-images.githubusercontent.com/59158426/106474533-5c614180-64e8-11eb-8327-1040f4f69cee.PNG)

Jacobian is considered as a partial derivative of multidimensional mapping

-Example : R^(m*n) to R (y=f(X)) 

 matrix derivative
 
 ![6](https://user-images.githubusercontent.com/59158426/106475524-5cae0c80-64e9-11eb-8d56-386a8e3a1614.PNG)
 
-Example : R^(m*n) to R^k (y=Wx where W is m*n matrix, x is n-dim vector)
 
 The Jacobian has dimension k*(m*n) where ith row is given by (k=m)
 
 ![7](https://user-images.githubusercontent.com/59158426/106475799-b44c7800-64e9-11eb-9694-a5b9d96547a5.PNG)
 






## Backpropagtion

-Directly optimizing the whole neural network(getting the gradient) is complicated. Therefore we use backpropagation.

-Backpropgation comes from the chain rule

![4](https://user-images.githubusercontent.com/59158426/106472698-481c4500-64e6-11eb-9525-e67264e1fd88.PNG)

When we want to calculate the gradient(jacobian) of Loss with respect to a certain weight, we multiply the upstream gradient(which comes backward from the loss) with
the local gradient. Then we use gradient descent algorithm to optimize the loss. 
(Note : the derivative can be either gradient or jacobians, but must have the same dimension with the variable to update)


-Scalar example with computational graph

![8](https://user-images.githubusercontent.com/59158426/106476627-992e3800-64ea-11eb-8d9f-63e0774da091.PNG)

q=x+y and f=q*z

df/dz can be directly calcaluated from f=q*z

df/dx's upstream gradient is df/dq and local gradient is dq/dx

df/dy's upstream gradient is df/dq and local gradient is dq/dy


-Backpropagation in neural network

![9](https://user-images.githubusercontent.com/59158426/106477297-591b8500-64eb-11eb-9ed9-512ad974cb9f.PNG)

We have to get the gradient of Wn for the update and Xn to deliver it to the next layer. dL/dXn+1 is received from next or (n+1)th layer.




