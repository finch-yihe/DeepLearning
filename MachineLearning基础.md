## Chapter 1 Introduction of Machine Learning

### 1. brief introduction  of  Artificial Intelligence

​	Since an early flush of optimism in the 1950s, smaller subsets of artificial intelligence - first machine learning, then deep learning, a subsets of machine learning - have created ever larger disruptions :

1. Target - Artificial Intelligence : early artificial intelligence stirs excitement
2. Means - Machine Learning : Machine Learning begins to flourish
3. Method - Deep Learning : Deep Learning breakthroughs drive AI boom

### 2. What is Machine Learning?

​	It means you write the program for learning. In other words, you are looking for a function from data that has "input" and "output". So, Machine Learning is so simple:

* Step 1 : define a set of function
* Step 2 : goodness of function
* Step 3 : pick the best function

### 3. Framework : Training  and Testing 

1. A set of function : model
2. Goodness of function $f​$ : pick the "Best" function $f^*​$ 
3. Training data and testing data

### 4. Learning Map(scenario-task-method)

1. Supervised Learning : Learning from teacher
   * Regression : the output of the target function is "scalar"
   * Classification : Linear model and non-linear model
     * Binary Classification : Yes or No
     * Multi-class Classification : Class 1, Class 2, ... Class n
   * Structured Learning
2. Semi-supervised Learning 
3. Transfer Learning
4. Unsupervised Learning 
5. Reinforcement Learning : Learning from critics

## Chapter 2 Regression

### 1. What is regression in machine learning?

​	Regression is basically a statistical approach to find the relationship between variables. In machine learning, this is used to predict the outcome of an event based on the relationship between variables obtained from the data-set. 

### 2. Case Study

1. Example Application

   * Estimating the Combat Power( $CP​$ ) of a `pokemon` after evolution
   * $f(x_{cp})$ = $CP$ after evolution $\hat y$ 

2. Steps

   * Step 1 : Model 

     * Find a set of function : $y = b + w * x_{cp}$ ($w$ and $b$ are parameters and them can be any value)
     * Linear model : $y = b + \sum w_ix_i$ ($x_i $ : an attribute of input x = feature, $w_i $ : weight, $b$ : bias)
   * Step 2 : Goodness of function

     * Training Data : 10 `pokemon` - $(x^1,\hat y^1), (x^2,\hat y^2) ,\dots ,(x^{10},\hat y^{10})$ 
     * Loss function L : $L(f) = L(w,b) = \sum_{n = 1}^{10}(\hat y^n - (b + w * x_{cp}^n))^2​$
       * Input : a function 
       * Output : how bad it is
   * Step 3 : Best Function

     * $f^*$ = ${arg\min}_f L(f)$ 
     * $w^*,b^* = {arg\min}_{w,b} L(w,b) = {arg\min}_{w,b} \sum_{n = 1}^{10}(\hat y^n - (b + w * x_{cp}^n))^2$
     * Gradient Descent :

       * Consider loss function $L(w)$ with one parameter w :
         * (Randomly)Pick an initial value $w^0​$
         * Compute $\frac{dL}{dw}|_{w = w^0}​$ and $w^1 \leftarrow w^0 - \eta\frac{dL}{dw}|_{w=w^0}​$ ( $\eta​$ is called "learning rate")
         * Many iteration and you can get a local optimal value(it maybe not global optimal)
       * How about two parameters? $w^*,b^* = {arg \min}_{w,b} L(w,b) $ 
         * (Randomly)Pick an initial value $w^0,b^0​$
         * Compute $\frac{\partial L}{\partial w}|_{w = w^0,b=b^0}​$ , $\frac{\partial L}{\partial b}|_{w = w^0,b=b^0}​$ and $w^1 \leftarrow w^0 - \eta\frac{\partial L}{\partial w}|_{w=w^0,b=b^0}​$  $b^1 \leftarrow b^0 - \eta\frac{\partial L}{\partial b}|_{w=w^0,b=b^0}​$
         * Many iteration and you can get a local optimal value(it maybe not global optimal)
       * In linear regression, the loss function $L​$ is convex and there is no local optimal value
   * Step 4 : How is the result? - Generalization
     * What we really care about is the error on new data(testing data)
     * How can we do better? So, we can select another model or redesign the model.
     * We can choose the model : $y = b + w_1 * x_{cp} + w_2 * x_{cp}^2​$ and so on.
     * `Overfitting` : A more complex model yields lower error on training data but the model does not always lead to better performance on testing data. So, we should select a suitable model.
     * In order to avoid `overfitting`, we can  go back to step 2 using Regularization
       * $L(f) = \sum_{n = 1}^{10}(\hat y^n - (b + w * x_{cp}^n))^2 + \lambda\sum(w_i)^2$ 
       * The functions with smaller $w_i$ are better.
       * Why smooth functions are preferred? If some noises corrupt input $x_i​$ when testing, a smoother function has less influence. We prefer smooth function, but don't be too smooth.
   * Conclusion : Original $CP​$ and species almost decide the $CP​$ after evolution(there are probably other hidden factors).

3. Regression Demo

   ```python
   # -*- coding: utf-8 -*-
   # Copyright (C) 2018 HuangYihe Technologies Inc.
   # name: Demo/RegressionDemo.py
   
   import numpy as np
   import matplotlib.pyplot as plt
   
   x_data = np.int8(np.random.randint(1, 1000, 10))
   y_data = -188.4 + 2.67 * x_data
   
   x = np.arange(-200, -100, 1)
   y = np.arange(-5, 5, 0.1)
   Z = np.zeros((len(x), len(y)))
   for i in range(len(x)):
       for j in range(len(y)):
           b = x[i]
           w = y[j]
           Z[j][i] = 0
           for n in range(len(x_data)):
               Z[j][i] = Z[j][i] + (y_data[n] - b - w * x_data[n])**2
           Z[j][i] = Z[j][i]/len(x_data)
   
   b = -120
   w = -4
   iteration = 100000
   b_history = [b]
   w_history = [w]
   lr_b = 0
   lr_w = 0
   
   for i in range(iteration):
       b_grad = 0.0
       w_grad = 0.0
       for n in range(len(x_data)):
           b_grad = b_grad - 2.0 * (y_data[n] - b - w * x_data[n]) * 1.0
           w_grad = w_grad - 2.0 * (y_data[n] - b - w * x_data[n]) * x_data[n]
   
       lr_b = lr_b + b_grad ** 2
       lr_w = lr_w + w_grad ** 2
   
       b = b - 1/np.sqrt(lr_b) * b_grad
       w = w - 1/np.sqrt(lr_w) * w_grad
   
       b_history.append(b)
       w_history.append(w)
   
   plt.contourf(x, y, Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
   plt.plot([-188.4], [2.67], 'x', ms=12, markeredgewidth=3, color='orange')
   plt.plot(b_history, w_history, 'o-', ms=1, lw=1.5, color='black')
   plt.xlim(-200, -100)
   plt.ylim(-5, 5)
   plt.xlabel(r'$b$', fontsize=16)
   plt.ylabel(r'$w$', fontsize=16)
   plt.show()
   ```

## Chapter 3 Where does the error come from?

   ### 1. Review

1. Error due to bias
2. Error due to variance

### 2. Bias and Variance of Estimator

* Estimate the mean of a variable x
  * assume the mean of x is $\mu $
  * assume the variance of x is $\sigma^2$ 
* Estimator of mean $\mu​$ :
  * Sample N points : $\{x^1,x^2,\dots,x^N\}$
  * $m = \frac{1}{N}\sum_nx^n \not= \mu​$
  * $E[m] = E[\frac{1}{N}\sum_nx^n] = \frac{1}{N}\sum_nE[x^n] = \mu​$ and it is unbiased.
  * $Var[m] = \frac{\sigma^2}{N}​$ and variance depends on the number of samples.
* Estimator of variance $\sigma ^2$ :
  * Sample N point : $\{x^1,x^2,\dots,x^N\}$ 
  * $m = \frac{1}{N}\sum_nx^n \ and \ s^2 = \frac{1}{N}\sum_n(x^n - m)^2 ​$ 
  * Biased estimator : $E[s^2] = \frac{N - 1}{N}\sigma ^2 \not= \sigma ^ 2$ 
* What is bias and variance? $E[f^*] = \bar f​$ and $f^*​$ is an estimator of $\hat f ​$
  * bias is the distance between $\hat f$ and $\bar f​$ 
  * variance is the distance between $f^*$ and $\bar f$ 
* Parallel Universes in quantum mechanics
  * In different universes, we use the same model, but obtain different $f^*​$.
  * Simpler model is less influenced by the sampled data. So, the simpler the model is, the smaller variance the model has, but larger bias the model has.
  * Usually there are two situations : large bias small variance and small bias large variance.
  * Large bias leads to `underfitting` and large variance leads to `overfitting`.
  * Diagnosis:
    * If your model cannot even fit the training examples, then you have large bias.
    * If you can fit the training data, but large error on testing data, then you probably have large variance. 
  * For bias, redesign your model :
    * Add more features as input
    * A more complex model
  * For variance, add more data :
    * More data : very effective, but not always practical
    * Regularization
  * Model Selection
    * There is usually a trade-off between bias and variance.
    * Select a model that balances two kinds of error to minimize total error.
    * So, we can divide training set into N-fold Cross Validation(training set and validation set)

## Chapter 4 Gradient Descent

### 1. Review

* In step 3, we have to solve the following optimization problem :
  $$
  \theta^* = arg\min_\theta L(\theta) (\ L:loss\ function \quad \theta:parameters\ )
  $$

* Suppose that $\theta$ has two variables $\{\theta_1,\theta_2\}$ 

* Randomly start at $\theta^0​$ =  $\begin{bmatrix} \theta_1^0 \\ \theta_2^0\end{bmatrix}​$

* $\begin{bmatrix} \theta_1^1 \\ \theta_2^1\end{bmatrix} = \begin{bmatrix} \theta_1^0 \\ \theta_2^0\end{bmatrix} - \eta\begin{bmatrix} \partial L(\theta_1^0)/\partial(\theta_1) \\ \partial L(\theta_2^0)/\partial (\theta_2)\end{bmatrix} \Rightarrow \theta^1 = \theta^0 - \eta\nabla L(\theta^0) \quad (\nabla L(\theta) = \begin{bmatrix} \partial L(\theta_1)/\partial \theta_1 \\ \partial L(\theta_2)/\partial \theta_2\end{bmatrix})​$

* Many iteration and you can get a local optimal value(it maybe not global optimal).

### 2. Tip 1 : Tuning your learning rates

* $\theta^i = \theta^{i-1} - \eta\nabla L(\theta^{i-1})$ and Set the learning rate $\eta $ carefully.

* You can visualize the change of the loss with the update of the parameters.

* Adaptive Learning Rates : Popular & Simple Idea is reduce the learning rate by some factor every few epochs

  * At the beginning, we are far from the destination, so we use larger learning rate
  * After several epochs, we are close to the destination, so we reduce the learning rate
  * E.g. 1/t decay : $\eta ^t = \eta / \sqrt{t+1} ​$ 

* Learning rate cannot be one-size-fits-all : Giving different parameters different learning rates.

* `Adagrad` : $\eta ^t = \frac{\eta}{\sqrt{t+1}} \quad g^t = \frac{\partial L(\theta^t)}{\partial w}​$

  * Divide the learning rate of each parameter by the root mean square of its previous derivatives.

  * Vanilla Gradient descent : $ w^{t+1} \leftarrow w^t - \eta^tg^t​$ (w is one parameters)

  * `Adagrad` : $w^{t+1} \leftarrow w^t - \frac{\eta^t}{\sigma^t}g^t$ ($\sigma^t$ : root mean square of the previous derivatives of parameter w ) and $ \sigma^t $ is :
    $$
    \sigma^t = \sqrt \frac{\sum_{i=0}^t(g^i)^2}{t+1}
    $$

  * So, we get $ w^{t+1} \leftarrow w^t - \frac{\eta}{\sum_{i=0}^t(g^i)^2}g^t$  and the best step is $\frac{|First\ derivative|}{Second \ derivative}$.

### 3. Tip 2 : Stochastic Gradient Descent

* Difference between normal gradient descent and stochastic gradient descent
  * Normal : Loss is the summation over all training examples and update after seeing all examples.
  * Stochastic : Loss for only one example and update for each example.

### 4. Tip 3 : Feature Scaling

* Feature scaling is a method used to standardize the range of independent variables or features of data.
* $x^r_i \leftarrow \frac{x^r_i - m_i}{\sigma_i}$ and the means of all dimensions are 0, the variances are all 1 ( For each dimension i, $m_i$ is mean and $ \sigma_i$ is standard deviation ).

### 5. Gradient Descent Theory

1. Question : When solving $ \theta^* = arg\min_\theta L(\theta) ​$ by gradient descent, each time we update the parameters, we obtain $\theta ​$ that makes $L(\theta)​$ smaller. Is this statement correct?

   * Answer : No. 

2. Warning of Math : 

   * Formal Derivation

     * Suppose that $ \theta $ has two variables $\{ \theta_1, \theta_2\}$ .
     * Given a point, we can easily find the point with the smallest value nearby.

   * Taylor Series : Let $h(x)$ be any function infinitely differentiable around $x = x_0$.
     $$
     h(x) = \sum_{k=0}^\infty\frac{h^{(k)}(x_0)}{k!}(x-x_0)^k=h(x_0) + h'(x_0)(x-x_0)+\frac{h''(x_0)}{2!}(x-x_0)^2+\dots
     $$
     When $x​$ is close to $x_0 \Rightarrow h(x) \approx h(x_0) + h'(x_0)(x-x_0)​$.

   * Multi-variable Taylor Series : 
     $$
     h(x,y) = h(x_0,y_0)+\frac{\partial h（x_0,y_0)}{\partial x}(x-x_0)+\frac{\partial h（x_0,y_0)}{\partial y}(y-y_0)
     $$

     $$
     +something \ related \ to \ (x-x_0)^2 \ and \ (y-y_0)^2 + \dots
     $$

     When  $x​$ and $y​$ is close to $x_0​$ and $y_0​$ 
     $$
     h(x,y) \approx h(x_0,y_0)+\frac{\partial h(x_0,y_0)}{\partial x}(x-x_0)+\frac{\partial h(x_0,y_0)}{\partial y}(y-y_0)
     $$

   * So, back to formal derivation : 
     $$
     L(\theta) \approx L(a,b) + \frac{\partial L(a,b)}{\partial \theta _1}(\theta_1-a)+\frac{\partial L(a,b)}{\partial \theta_2}(\theta_2-b)
     $$
     and $s = L(a,b), u = \frac{\partial L(a,b)}{\partial \theta _1}, v = \frac{\partial L(a,b)}{\partial \theta_2} are \ constant ​$. So, we get $L(\theta) \approx s + u(\theta_1 -a )+ v (\theta_2 - b) ​$. In order to find $\theta_1​$ and $\theta_2​$ in the circle minimizing $L(\theta )​$ : $(\theta_1 - a)^2 + (\theta_2 - b)^2 \leq d^2​$ and we suppose $(\theta_1 - a ) = \triangle \theta_1 \ and \ (\theta_2 - b) = \triangle \theta_2​$. To minimize $L( \theta) ​$,
     $$
     \begin{bmatrix} \triangle\theta_1 \\ \triangle\theta_2\end{bmatrix} = -\eta \begin{bmatrix}u\\v\end{bmatrix} \Rightarrow \begin{bmatrix} \theta_1 \\ \theta_2\end{bmatrix} = \begin{bmatrix}a\\b\end{bmatrix} - \eta \begin{bmatrix}u\\v\end{bmatrix} = \begin{bmatrix}a\\b\end{bmatrix} - \eta \begin{bmatrix}\frac{\partial L(a,b)}{\partial \theta _1}\\\frac{\partial L(a,b)}{\partial \theta_2}\end{bmatrix}
     $$
     but the premise is that $b$ is small enough.

   * More Limitation of Gradient Descent : Stuck at local minimum, Stuck at saddle point or very slow at the plateau.

## Chapter 5 Classification : Probabilistic Generative Model

### 1. What is Classification

* $x \Rightarrow Function \Rightarrow Class \ n$

### 2. Ideal Alternatives - generative model

1. Function(model) : 

$$
x \Rightarrow \begin{matrix}g(x) > 0 \quad Output = class \ 1 \\ else \quad Output = class \ 2 \end{matrix} = f(x)
$$

2. Loss function : The number of times $f​$ get incorrect results on training data.

$$
L(f) = \sum_n\delta(f(x^n) \not= \hat y^n)
$$
3. Find the best function : 

   * Example : `Perceptron`, `SVM`

   * Generative Model : $P(x) = P(x|C_1)P(C_1) + P(x|C_2)P(C_2)$

     * Two Classes : Given an x, which class does it belong to
       $$
       P(C_1|x) = \frac{P(x|C_1)P(C_1)}{P(x|C_1)P(C_1) + P(x|C_2)P(C_2)}
       $$

       * Probability from Class ：
         * Assume the points are sampled from a Gaussian distribution
         * Find the Gaussian distribution behind them $\Rightarrow$ Probability for new points

       * Maximum Likelihood : The Gaussian with any mean $\mu $ and covariance matrix $\Sigma $ can generate these points $\Rightarrow ​$ Different Likelihood

         * The Gaussian Distribution : 
           $$
           f_\mu,_\Sigma(x)=\frac{1}{(2\pi)^{D/2}}\frac{1}{|\Sigma|^{1/2}}exp\{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\}
           $$

           * Input : vector x

           * Output : probability of sampling x 

           * The shape of the function determines by mean $\mu $ and covariance matrix $\Sigma $ 

       * Likelihood of a Gaussian with mean $\mu ​$ and covariance matrix $\Sigma ​$ = the probability of Gaussian samples $x^1,x^2,x^3,\dots,x^n =L(\mu,\Sigma) = f_\mu,_\Sigma(x^1)f_\mu,_\Sigma(x^2)f_\mu,_\Sigma(x^3)\dots f_\mu,_\Sigma(x^n)​$ 

       * So, we assume $x^1,x^2,x^3,\dots,x^n​$ generate from the Gaussian $( \mu^*,\Sigma ^* )​$ with the maximum likelihood  $\Rightarrow ​$ $\mu^*,\Sigma ^* = arg\max_{\mu,\Sigma}L(\mu,\Sigma)​$ And we know $\mu^* = \frac{1}{n}\Sigma^n_{n=1}x^n​$ and $\Sigma^*=\frac{1}{n}\Sigma^n_{n=1}(x^n-\mu^*)(x^n-\mu^*)^T​$  

       * Finally, if $P(C_1|x) > 0.5 \Rightarrow x \ belong \ to \ class \ 1$ and vice versa. 

### 3. Modifying Model - discriminative model 

 * Share the same covariance matrix $\Sigma ​$ and get less parameters

   * Find $\mu^1,\mu^2,\Sigma​$ maximizing the likelihood $L(\mu^1,\mu^2,\Sigma)​$

   * $L(\mu^1,\mu^2,\Sigma) = f_{\mu^1},_\Sigma(x^1)f_{\mu^1},_\Sigma(x^2)\dots f_{\mu^1},_\Sigma(x^m) * f_{\mu^2},_\Sigma(x^1)f_{\mu^2},_\Sigma(x^2)\dots f_{\mu^2},_\Sigma(x^n)$  

   * $\mu^1,\mu^2 = \frac{1}{n}\Sigma^n_{n=1}x^n​$ and $\Sigma =\frac{m}{m+n}\Sigma^1+\frac{n}{m+n}\Sigma^2​$

   * And this is a linear model

 * For binary features, you may assume they are from Bernoulli distributions.

 * If you assume all the dimensions are independent, then you are using Naive Bayes Classifier.

   * What is Naive Bayes?$P(x|C_i)=P(x_1|C_i)P(x_2|C_i)$ 

 * Posterior Probability
   $$
   P(C_1|x) = \frac{P(x|C_1)P(C_1)}{P(x|C_1)P(C_1) + P(x|C_2)P(C_2)}=\frac{1}{1+\frac{P(x|C_2)P(C_2)}{P(x|C_1)P(C_1)}} =\frac{1}{1+exp(-z)} = \sigma(z)
   $$

 * And finally we get $P(C_1|x) = \sigma(z)​$ and $z = ln\frac{|\Sigma^2|^{1/2}}{|\Sigma^1|^{1/2}}-\frac{1}{2}x^T(\Sigma^1)^{-1}x+(\mu^1)^T(\Sigma^1)^{-1}x-\frac{1}{2}(\mu^1)^T(\Sigma^1)^{-1}\mu^1​$ $+\frac{1}{2}x^T(\Sigma^2)^{-1}x-(\mu^2)^T(\Sigma^2)^{-1}x+\frac{1}{2}(\mu^2)^T(\Sigma^2)^{-1}\mu^2+ln\frac{m}{n}​$ 

 * If we assume $\Sigma^1 = \Sigma^2 = \Sigma​$, we get $z = (\mu^1-\mu^2)^T(\Sigma^1)^{-1}x-\frac{1}{2}(\mu^1)^T(\Sigma^1)^{-1}\mu^1+\frac{1}{2}(\mu^2)^T(\Sigma^2)^{-1}\mu^2+ln\frac{m}{n} = w^Tx + b​$ 

 * So $P(C_1|x) = \sigma(w * x+b)​$ and in generative model, we estimate $m,n,\mu^1,\mu^2,\Sigma​$ ,then we have $w​$ and $b​$. And How about directly find  $w​$ and $b​$ ?

## Chapter 6 Logistic Regression

### 1. Step 1 : Function Set

* We want to find $P_{w,b}(C_1|x)​$
  * If  $P_{w,b}(C_1|x) \geq 0.5$, output $C_1​$
  * Otherwise, output $C_2$ 
* $P_{w,b}(C_1|x) = \sigma(z)$ and $ \sigma(z) = \frac{1}{1+exp(-z)} $
  * $z = w * x + b = \Sigma_iw_ix_i+b​$ 
* So, we get the Function Set : $f_{w,b}(x) = P_{w,b}(C_1|x)$ including all different $w $ and $b$.

### 2. Step 2 : Goodness of a Function

Training Data : $\begin{matrix}x^1 \ x^2 \ x^3 \ \dots \ x^N \\ C_1 \ C_1 \ C_2 \ \dots \ C_1 \end{matrix}​$

* Assume the data is generated based on $f_{w,b}(x) = P_{w,b}(C_1|x)​$
* Given a set of $w​$ and $b​$, what is its probability of generating the data?
* $L(w,b)=f_{w,b}(x^1)f_{w,b}(x^2)(1-f_{w,b}(x^3))\dots f_{w,b}(x^N)​$ 
* The most likely $w^*$ and $b^*$ is the one with the largest $L(w,b)$. 
* But we can change it like this : 
  * $w^*,b^* = arg\max_{w,b}L(w,b)=arg\min_{w,b}-lnL(w,b)​$ 
  * And we assume $\hat{y}^n​$ : 1 for class 1, 0 for class 2.
  * $-lnL(w,b) = -\Sigma_n[\hat{y}^nlnf_{w,b}(x^n)+(1-\hat y^n)ln(1-f_{w,b}(x^n))]$ 
  * This is Cross entropy between two Bernoulli distribution.
  * Cross entropy : $H(p,q) = -\Sigma_xp(x)ln(q(x))​$ 

### 3. Step 3 : Find the best function

* $\frac{\partial(-lnL(w,b))}{\partial(w_i)} = \Sigma_n-(\hat y^n-f_{w,b}(x^n))x_i^n​$ 
* $w^i \leftarrow w^{i} - \eta\Sigma_n-(\hat y^n-f_{w,b}(x^n))x_i^n$ : Larger difference, larger update.

### 4. Difference between logistic regression and linear regression

1. Step 1 : 
   - The former : Output a number between 0 and 1
   - The latter : Output any value
2. Step 2 : 
   - The former : $\hat{y}^n$ : 1 for class 1, 0 for class 2 and $L(f) = \Sigma_nC(f(x^n),\hat y^n)$ 
   - The latter : $\hat{y}^n$ : a real number and $L(f)=\frac{1}{2}\Sigma_n(f(x^n)-\hat y^n)^2​$ 
   - Cross entropy : $C(f(x^n),\hat y^n)=-[\hat y^nlnf(x^n) + (1-\hat y^n)ln(1-f(x^n))]​$ 
3. Step 3 : 
   - The former : $w^i \leftarrow w^i - \eta\Sigma_n-(\hat y^n-f_{w,b}(x^n))x_i^n​$ 
   - The latter : $w^i \leftarrow w^i - \eta\Sigma_n-(\hat y^n-f_{w,b}(x^n))x_i^n​$ 

### 5. Discriminative v.s. Generative

* Model : $P(C_1|x) = \sigma(w * x+b)​$ 
* Different :
  * The former : directly find $w$ and $b​$.
  * The latter : find $\mu^1,\mu^2,\Sigma^{-1}$ and then compute $w$ and $b​$.
* Will we obtain the same set of $w$ and $b​$?
  * No, the same model(function set),but different function is selected by the same training data.
* Benefit of generative model :
  * With the assumption of probability distribution, less training data is needed.
  * With the assumption of probability distribution, more robust to the noise.
  * Priors and class-dependent probabilities can be estimated from different sources.

### 6. Multi-class Classification

* Training data :

$$
\begin{matrix}C_1:w^1,b_1 \quad z_1=w^1*x+b_1 \\ C_2:w^2,b_2 \quad z_2=w^2*x+b_2 \\ C_3:w^3,b_3 \quad z_3=w^3*x+b_3 \end{matrix}
$$

* Through `Softmax` Function : $y_i = \frac{exp(z_i)}{\Sigma_{i=1}^3exp(z_i)}​$ and output $y_i = P(C_i|x)​$, and we get $y_1,y_2,y_3​$ 
* Then we compute cross entropy between $y_{i}$ and $\hat y_i$ : $\Sigma_{i=1}^3-\hat y_ilny_i​$ 

### 7. Limitation of Logistic Regression

* The boundary between data is straight.
* Solution : 
  * Feature Transformation, but not always easy to find a good transformation.
  * So, we can use cascading logistic regression models : Feature Transformation + Classification.
  * Finally we call a Logistic Regression a Neuron and name the network consisting of some Logistic Regression after Neural Network.

## Chapter 7 Brief Introduction of Deep Learning 

### 1. Ups and downs of Deep Learning

* 1958 : `Perceptron`linear model)
* 1969 : `Perceptron` has limitation
* 1980s : Multi-layer `perceptron`
  * Do not have significant difference from `DNN` today
* 1986 : `Backpropagation`
  * Usually more than 3 hidden layers is not helpful
* 1989 : 1 hidden layer is "good enough", why deep?
* 2006 : `RBM(Restricted Boltzmann Machine)` initialization(breakthrough)
* 2009 : `GPU`
* 2011 : Start to be popular in speech recognition
* 2012 : win `ILSVRC` image competition

### 2. Three Steps for Deep Learning

* Step 1 : define a set of function
* Step 2 : goodness of function : Find a function in function set that minimizes total loss $L =\sum_{n=1}^NC^n(Cross \ Entropy \ :C(y,\hat y)=-\sum_{i=1}^{10}\hat y_ilny_i)$ and fing the network parameters $\theta^* $ that minimizes total loss $L$. 
* Step 3 : pick the best function

### 3. Neural Network

* Model : $x^n \rightarrow \begin{matrix}NN \\ \theta \end{matrix} \rightarrow y^n \leftrightarrows _{C^n} \hat y^n$ 

* Different connection leads to different network structures

* Network parameter $\theta ​$ : all the weights and biases int the "neuron"

* Example : Fully Connect Feed-forward Network

  * We can get neural network a function that inputs vector and outputs vector.

  * So, Given network structure, we can define a function set.

  * Matrix Operation : Feature Transformation
    $$
    \sigma(\left[\begin{matrix}w_{11} & w_{21}\\w_{12} & w_{22} \end{matrix}\right]\left[\begin{matrix}x_1\\x_2\end{matrix}\right]+\left[\begin{matrix}b_1\\b_2\end{matrix}\right]) = \left[\begin{matrix}y_1\\y_2\end{matrix}\right]
    $$

  * Input Layer = a vector with feature

  * Hidden layer = Feature extractor replacing feature engineering and we need to decide the network structure to let a good function in our function set.

  * Output Layer = Multi-class Classification

  * `GPU` : Using parallel computing techniques to speed up matrix operation.

* Deep = Many hidden layers

* Example Application : Handwriting Digit Recognition

  * Input : image(ink=1,no ink = 0)
  * Output : Each dimension represents the confidence of a digit.
  * Hidden Layers + Output Layer = A function set containing the candidates for Handwriting Digit Recognition.And we need to decide the network structure to let a good function in our function set.

* Questions : How many layers? How many neurons for each layer?

  * Trial and Error + Intuition

* Questions : Can the structure be automatically determined?

  * E.g. Evolutionary Artificial Neural Networks

* Question : Can we design the network structure?

  * `Convolutional Neural Network`(CNN)

* `Backpropagation` : an efficient way to compute $\partial L / \partial w$ in neural network.

* Universality Theorem : Any continuous function $f : R^N \rightarrow R^M$ can be realized by a network with one hidden layer(given enough hidden neurons) 

  * shallow network can represent any function but using deep structure is more effective.

## Chapter 8 Backpropagation

### 1. Gradient Descent

* Network parameters $\theta = \{w_1,w_2,\dots,b_1,b_2,\dots\}$ 

* Starting Parameters : $\theta^0 \rightarrow \theta^1​$ 

* $$
  \bigtriangledown L(\theta)=\left[\begin{matrix}\partial L(\theta)/\partial w_1 \\ \partial L(\theta)/\partial w_2 \\ \vdots \\ \partial L(\theta)/\partial b_1 \\ \partial L(\theta)/\partial b_2 \\ \vdots \end{matrix}\right] and \ Compute \ \bigtriangledown L(\theta^n) \ by \ \theta^i = \theta^{i-1} - \eta\bigtriangledown L(\theta^0) 
  $$

* To compute the gradients efficiently, we use `backpropagation`.

### 2. Chain Rule

* If $y = g(x) \ z=h(y)$ then $\frac{dz}{dx} = \frac{dz}{dy}\frac{dy}{dx}$
* If $x=g(s) \ y=h(s) \ z=k(x,y)$ then $\frac{dz}{ds} = \frac{\partial z}{\partial x}\frac{dx}{ds} + \frac{\partial z}{\partial y}\frac{dy}{ds}$ 

### 3. Backpropagation

* $L(\theta)=\sum^N_{n=1}C^n(\theta) \rightarrow \frac{\partial L(\theta)}{\partial w} = \sum_{n=1}^N\frac{\partial C^n(\theta)}{\partial w}$  

* $\frac{\partial C}{\partial w} = \frac{\partial z}{\partial w}\frac{\partial C}{\partial z}​$ and the value of input connected bu the weight.
  * Forward pass : Compute $\partial z / \partial w​$ for all parameters
  * Backward pass : Compute $\partial C/\partial z $ for all activation function inputs $z$ 
* If we assume $a = \sigma(z)$, then $\frac{\partial C}{\partial z}=\frac{\partial a}{\partial z}\frac{\partial C}{\partial a}$ and $\frac{\partial C}{\partial a}=\frac{\partial z'}{\partial a}\frac{\partial C}{\partial z'}+\frac{\partial z''}{\partial a}\frac{\partial C}{\partial z''}(Chain \ rule)$ and $\sigma'(z)$ is a constant because $z$ is already determined in the forward pass.
* Compute $\partial C/\partial z​$ recursively until we reach the output layer.

## Chapter 9 "Hello World" of deep learning

### 1. Speed - Mini-batch

* We do not really minimize total loss.
  * Randomly initialize network parameters
  * Pick one batch
  * Update parameters once
  * Many iteration until all mini-batches have been picked
* Smaller batch size means more updates in one epoch.
  * E.g. 50000 examples
  * batch size = 1, 50000 updates in one epoch
  * batch size = 10, 5000 updates in one epoch
  * with `GPU` the more batch size is, the faster and more stably machine runs. But very large batch size can yield worse performance.

### 2. Implement

* Dataset : `MNIST`

* Description :  The `MNIST` database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples. The digits have been size-normalized and centered in a fixed-size `28x28` image.

* Example with `keras`:

  ```python
  # -*- coding: utf-8 -*-
  # Copyright (C) 2018 HuangYihe Technologies Inc.
  # name: MNIST-DNN-Keras-Demo.py
  
  import os
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  import tensorflow as tf
  mnist = tf.keras.datasets.mnist
  
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  
  model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(500, activation=tf.nn.relu),
      tf.keras.layers.Dense(500, activation=tf.nn.relu),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
  ])
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  
  model.fit(x_train, y_train, batch_size=100, epochs=20)
  score = model.evaluate(x_test, y_test)
  print("Total loss on Testing Set:", score[0])
  print("Accuracy of Testing Set:", score[1])
  ```

* Example with `tensorflow`

  ```python
  # -*- coding: utf-8 -*-
  # Copyright (C) 2018 HuangYihe Technologies Inc.
  # name: Demo/MNIST-DNN-Tensorflow-Demo.py
  
  import os
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  import tensorflow as tf
  import math
  from tensorflow.examples.tutorials.mnist import input_data
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
  
  def add_layer(inputs, in_size, out_size, activation_function=None,):
      Weights = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=1.0 / math.sqrt(float(in_size))))
      biases = tf.Variable(tf.zeros([1, out_size]))
      Wx_plus_b = tf.matmul(inputs, Weights) + biases
      if activation_function is None:
          outputs = Wx_plus_b
      else:
          outputs = activation_function(Wx_plus_b,)
      return outputs
  
  def compute_accuracy(v_xs, v_ys):
      y_pre = sess.run(prediction, feed_dict={xs: v_xs})
      correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
      return result
  
  xs = tf.placeholder(tf.float32, [None, 784])
  ys = tf.placeholder(tf.float32, [None, 10])
  
  layer1 = add_layer(xs, 784, 500,  activation_function=tf.nn.relu)
  layer2 = add_layer(layer1, 500, 500,  activation_function=tf.nn.relu)
  prediction = add_layer(layer2, 500, 10,  activation_function=tf.nn.softmax)
  
  cross_entropy = -tf.reduce_sum(ys * tf.log(prediction))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  
  sess = tf.Session()
  init = tf.global_variables_initializer()
  sess.run(init)
  
  for i in range(12000):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
      if i % 600 == 0:
          print("Accuracy of Testing Set:{:.3f} in epoch {:0>2d}".format(compute_accuracy(mnist.test.images, mnist.test.labels),int(i/600+1)))
  ```

* Difference between `categorical_crossentropy` and `sparse_categorical_crossentropy`

  * If your targets are one-hot encoded, use `categorical_crossentropy`.
  * But if your targets are integers, use `sparse_categorical_crossentropy`.

## Chapter 10 Tips for Training DNN

### 1. Recipe of Deep Learning

* When no good results on Training Data
  * New activation function
  * Adaptive Learning Rate
* When no good results on Testing Data(`Overfitting`)
  * Early Stopping
  * Regularization
  * Dropout
* Application

### 2. Problem in training and testing

1. New activation function : Vanishing Gradient Problem (With Sigmoid Function on `MNIST` dataset)

   * Layers near the Input Layer : 
     * Smaller gradients
     * Learn very slow
     * Almost random
   * Layers near the Output Layer : 
     * Larger gradients
     * Learn very fast
     * Already converge
   * Solution : Using `ReLU` Function
   * The variant of `ReLU` : `Leaky ReLU` and `Parametric ReLU` 
     * `Maxout` : `Learnable activation function` and `ReLU` is a special cases of `Naxout` 
       * Activation function in `maxout` network can be any piecewise linear convex function
       * How many pieces depending on how many elements in a group
       * Given a training data x, we know which z would be the max and it becomes a thin and linear network
       * Different thin and linear network for different examples

2. Adaptive Learning Rate : The variant of `Adagrad` : `RMSProp`

   * Error surface can be very complex when training `NN` 
   * $w^i \leftarrow w^{i-1} - \frac{\eta}{\sigma^{i-1}}g^{i-1}$ 
   * $\sigma^0 = g^0  \ and \ \sigma^i=\sqrt{\alpha(\sigma^{i-1})^2+(1-\alpha)(g^i)^2}​$ 
   * Root Mean Square of the gradients with previous gradients being decayed

3. Adaptive Learning Rate : Hard to find optimal network parameters : Momentum 

   * Movement : movement of last step minus gradient at present and it is bot just based on gradient, but previous movement.
   * $v^i=\lambda v^{i-1} - \eta\bigtriangledown L(\theta^{i-1}) \ and \ \theta^i = \theta^{i-1} + v^i​$ 
   * $v^i$ is actually the weighted sum of all the previous gradient
   * Still not guarantee reaching global minimum, but give some hope.
   * Adam = `RMSProp` + `Momentum` 

4. Early Stopping : Using Validation set to decide when to stop training 

5. Regularization : New loss function to be minimized

   * Find a set of weight not only minimizing original cost but also close to zero
     * L2 regularization : 
       * $L'(\theta) = L(\theta) + \lambda \frac{1}{2}||\theta ||_2 \ and \ ||\theta||_2 = (w_1)^2 + (w_2)^2 + \dots​$  usually not consider biases
       * Update : $w^{t+1} = (1-\eta\lambda)w^t-\eta\frac{\partial L}{\partial w}​$ we call it Weight Decay
     * L1 regularization :
       * $L'(\theta) = L(\theta) + \lambda \frac{1}{2}||\theta ||_2 \ and \ ||\theta||_2 = |w_1| + |w_2| + \dots​$ 
       * Update : $w^{t+1} = w^t-\eta\frac{\partial L}{\partial w} - \eta\lambda sgn(w^t)​$ it always delete

6. Dropout : Each neuron has p% to dropout before updating the parameters

   * Using the new network for training
   * For each mini-batch, we resample the dropout neurons.
   * When testing, no dropout.
   * If the dropout rate at training is p%, all the weights times (1-p)% at testing
   * Dropout is a kind of ensemble

7. The story of fizz_buzz

    ```python
   # -*- coding: utf-8 -*-
   # Copyright (C) 2018 HuangYihe Technologies Inc.
   # name: Demo/HardTrain.py
   
   import os
   os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
   import numpy as np
   import tensorflow as tf
   
   def binary_encode(i, num_digits):
       return np.array([i >> d & 1 for d in range(num_digits)])[::-1]
   
   def fizz_buzz_encode(i):
       if i % 15 == 0:
           return np.array([0, 0, 0, 1])
       elif i % 5 == 0:
           return np.array([0, 0, 1, 0])
       elif i % 3 == 0:
           return np.array([0, 1, 0, 0])
       else:
           return np.array([1, 0, 0, 0])
   
   def init_weights(shape):
       return tf.Variable(tf.random_normal(shape, stddev=0.01))
   
   def fizz_buzz(i, prediction):
       return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]
   
   NUM_DIGITS = 10
   x_train = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
   y_train = np.array([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])
   x_test = np.array([binary_encode(i, NUM_DIGITS) for i in range(1, NUM_DIGITS ** 2)])
   y_test = np.array([fizz_buzz_encode(i) for i in range(1, NUM_DIGITS ** 2)])
   
   model = tf.keras.models.Sequential([
     tf.keras.layers.Flatten(input_shape=(10,)),
     tf.keras.layers.Dense(1000, activation=tf.nn.relu),
     tf.keras.layers.Dense(4, activation=tf.nn.softmax)
   ])
   model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
   model.fit(x_train, y_train, batch_size=20, epochs=100)
   score = model.evaluate(x_test, y_test)
   print("Acc:", format(score[1], "0.2f"))
   ```

## Chapter 11 Convolutional Neural Network

### 1. Why CNN for Image?

* Can the network be simplified by considering the properties of images?
  * Some patterns are much smaller than the whole image and a neuron does not to see the whole image to discover the pattern that connect to small region with less parameters.
  * The same patterns appear in different regions and we do not almost the same thing  because they can use the same set of parameters.
  * Sub-sampling the pixels will not change the object

### 2. The whole CNN

* Input the image and only modify the network structure and input format (vector -> 3-D tensor)
* Convolution and we use filter by setting stride and padding to engender the feature map. Each filter is a channel and the number of the channel is the number of filters. - some patterns are much smaller than the whole image and the same patterns appear in different regions
* Max Pooling - sub-sampling the pixels will not change the object
* Many iterations Step 2 and Step 3
* Flatten
* Fully Connected Feed-forward network
* Output the class

### 3. Implement

```python
# -*- coding: utf-8 -*-
# Copyright (C) 2018 HuangYihe Technologies Inc.
# name: Demo/MNIST-CNN-Keras-Demo.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

model = tf.keras.models.Sequential([
    tf.keras.layers.Convolution2D(25, (3, 3), input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Convolution2D(50, (3, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=100, epochs=20)
score = model.evaluate(x_test, y_test)
print("Total loss on Testing Set:", score[0])
print("Accuracy of Testing Set:", score[1])
```

## Chapter 12 Why deep?

### 1. Deeper is Better?

* Not surprised, more parameters, better performance.

### 2. Modularization

* Deep $\rightarrow$ Modularization $\rightarrow $ Less training data and do not put everything in your main function.
* This means each basic classifier can have sufficient training example and we use more classifiers called a module for the attributes that shares by the following classifiers.
* The modularization is automatically learned from data.

### 3. Application - Speech

* The hierarchical structure of human languages : Phoneme $\rightarrow $ Tri-phone $\rightarrow$ State
* The first stage of speech recognition
  * Classification : input $\rightarrow$ acoustic feature, output $\rightarrow$ state
  * Determine the state each acoustic feature belongs to
  * Each state has a stationary distribution for acoustic features

