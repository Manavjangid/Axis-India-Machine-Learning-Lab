#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


class BinaryLogisticRegression():
    def __init__(self, BatchSize, LearningRate = 10**(-6), ErrorTolerance=10**(-5), Verbose = True):
        self.LearningRate = LearningRate
        self.ErrorTolerance = ErrorTolerance
        self.Verbose = Verbose
        self.BatchSize = BatchSize
        
    def __posterior(self, theta_hat_0, theta_hat, x):
        x = np.array(x)
        return np.array(1/(1+np.exp(-(theta_hat_0 + np.matmul(theta_hat, x.T)))))

    def __log_loss_func(self, theta_hat_0, theta_hat, x, y):
        x = np.array(x)
        y = np.array(y)

        p = self.__posterior(theta_hat_0,theta_hat,x)

        log_p = np.array(pd.Series(np.log(p)).replace(to_replace=[np.inf,np.NINF],value=[0,0]))
        log_p_ = np.array(pd.Series(np.log(1-p)).replace(to_replace=[np.inf,np.NINF],value=[0,0]))

        lhs = np.matmul(y,log_p)
        rhs = np.matmul((1-y),log_p_)

        return -(lhs + rhs)
    
    def __derivative_theta_hat_0(self, theta_hat_0, theta_hat, x, y):
#         o_0 represents theta hat zero
        x = np.array(x)
        y = np.array(y)

        o_0 = ((y-1) * np.exp(theta_hat_0+np.matmul(theta_hat, x.T)) + y) / (np.exp(theta_hat_0+np.matmul(theta_hat, x.T)) + 1)

        o_0 = np.array(pd.Series(o_0).replace(to_replace=[np.nan],value=[0]))

        return np.sum(o_0)
    
    def __derivative_theta_hat(self, theta_hat_0, theta_hat, x, y):
#         o represents theta hat
        x = np.array(x)
        y = np.array(y)

        o = ((y-1) * np.exp(theta_hat_0+np.matmul(theta_hat, x.T)) + y) / (np.exp(theta_hat_0+np.matmul(theta_hat, x.T)) + 1)

        o = np.array(pd.Series(o).replace(to_replace=[np.nan],value=[0]))

        o = np.matmul(x.T, o)

        return o
    
    def fit(self, X_train, Y_train):
        epoch_counter = 1
        
        X = np.array(X_train)
        Y = np.array(Y_train)
        
        theta_hat_0_initial = 1
        theta_hat_initial = np.array([1 for _ in range(X.shape[1])])
                                                       
        while True:
            for i in range(X.shape[0]//self.BatchSize):
                
                random_indices = np.random.choice(a=np.arange(0,X.shape[0]),size=self.BatchSize,replace=False)
                X_batch = X[random_indices]
                Y_batch = Y[random_indices]
                                                       
                self.theta_hat_0_final = theta_hat_0_initial-(self.LearningRate*self.__derivative_theta_hat_0(theta_hat_0_initial
                                                                                              ,theta_hat_initial,X_batch,Y_batch))

                self.theta_hat_final = theta_hat_initial-(self.LearningRate*self.__derivative_theta_hat(theta_hat_0_initial
                                                                                        ,theta_hat_initial,X_batch,Y_batch))

                log_loss_func_initial = self.__log_loss_func(theta_hat_0_initial,theta_hat_initial,X_batch,Y_batch)
                log_loss_func_final = self.__log_loss_func(self.theta_hat_0_final,self.theta_hat_final,X_batch,Y_batch)


                

                theta_hat_0_initial = self.theta_hat_0_final
                theta_hat_initial = self.theta_hat_final
            
            if abs(log_loss_func_initial - log_loss_func_final) < self.ErrorTolerance:
                if self.Verbose:
                    print(f'\nConvergence has Acheived. Epoch = {epoch_counter}, Value of Log Loss Function = {log_loss_func_final}.\n')
                break
            
            if self.Verbose:
                print(f'Epoch = {epoch_counter}, Value of Log Loss Function = {log_loss_func_final}.\n\n')
            
            epoch_counter += 1
        
    
    def predict(self, X_test):
        
        self.predicted_labels = np.array(pd.Series(self.__posterior(self.theta_hat_0_final,self.theta_hat_final,X_test) > 0.5).replace(to_replace=[True, False], value=[1,0]))
        return self.predicted_labels


# In[3]:


if __name__ == "__main__":
    print(1)

