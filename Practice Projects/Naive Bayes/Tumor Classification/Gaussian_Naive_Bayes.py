#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy.stats as s


# In[2]:


class GaussianNaiveBayes():
    def __init__(self,discriminant_analysis_assumption='qda', rda_param=np.nan):
        
        self.discriminant_analysis_assumption = discriminant_analysis_assumption
        self.rda_param = rda_param
        
        
    def fit(self,X_train,Y_train):
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        self.unique_class = np.unique(Y_train)
        data=pd.DataFrame(X_train)
        data['label'] = Y_train
        
        self.cov_mat = list()
        self.mean_vec = list()
        self.priors = list()
        
        for i in self.unique_class:
            self.cov_mat.append(np.reshape(np.array(data[data['label'] == i].iloc[:,:-1].cov()), newshape=(X_train.shape[1],-1)))
            self.mean_vec.append(np.reshape(np.array(data[data['label'] == i].iloc[:,:-1].mean(axis=0)), newshape=(X_train.shape[1],)))
            self.priors.append(data[data['label'] == i].shape[0]/data.shape[0])
            
        if self.discriminant_analysis_assumption != 'qda':
            self.pooled_cov_mat = np.zeros(shape=(X_train.shape[1],X_train.shape[1]))
            for i in range(len(self.unique_class)):
                self.pooled_cov_mat += (self.cov_mat[i] * (np.count_nonzero(data['label'] == self.unique_class[i]) - 1))
            self.pooled_cov_mat /= (data.shape[0] - len(self.unique_class))
            if self.discriminant_analysis_assumption == 'rda':
                self.rda_cov_mat = list()
                for i in range(len(self.unique_class)):
                    rda_1 = ((1-self.rda_param[0]) * self.pooled_cov_mat) + (self.rda_param[0] * self.cov_mat[i])
                    self.rda_cov_mat.append(((1-self.rda_param[1]) * rda_1) + (self.rda_param[1] * (np.sum(rda_1.diagonal())/(rda_1.shape[1])) * np.eye(X_train.shape[1],X_train.shape[1])))
                del self.pooled_cov_mat, rda_1
            del self.cov_mat
        
        del X_train, Y_train, data
        
        
        
    def predict(self, X_test):
        
        self.predicted_labels = []
        
        if self.discriminant_analysis_assumption == 'rda':
            for i in range(len(self.unique_class)):
                self.predicted_labels.append(np.reshape(s.multivariate_normal.pdf(np.array(X_test), self.mean_vec[i], self.rda_cov_mat[i]) * self.priors[i], (np.array(X_test).shape[0], -1)))
        elif self.discriminant_analysis_assumption == 'lda':
            for i in range(len(self.unique_class)):
                self.predicted_labels.append(np.reshape(s.multivariate_normal.pdf(np.array(X_test), self.mean_vec[i], self.pooled_cov_mat) * self.priors[i], (np.array(X_test).shape[0], -1)))
        else:
            for i in range(len(self.unique_class)):
                self.predicted_labels.append(np.reshape(s.multivariate_normal.pdf(np.array(X_test), self.mean_vec[i], self.cov_mat[i]) * self.priors[i], (np.array(X_test).shape[0], -1)))
        
        self.predicted_labels = pd.DataFrame(np.concatenate(self.predicted_labels, axis=1), columns=self.unique_class)
        self.predicted_labels = np.array(self.predicted_labels.idxmax(axis=1))
        return self.predicted_labels


# In[3]:


if __name__ == "__main__":
    print(1)

