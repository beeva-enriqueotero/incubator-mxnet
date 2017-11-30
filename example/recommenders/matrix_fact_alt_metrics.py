# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import math
import mxnet as mx
import numpy as np
import mxnet.notebook.callback

import logging
logging.basicConfig(level=logging.DEBUG)

def RMSE(label, pred):
    ret = 0.0
    n = 0.0
    pred = pred.flatten()
    for i in range(len(label)):
        ret += (label[i] - pred[i]) * (label[i] - pred[i])
        n += 1.0
    return math.sqrt(ret / n)

def binomial_cross_entropy(label, pred):
    eps = 1e-6 # Avoid -inf when taking log(0)
    eps1 = 1. + eps
    eps_1 = 1. - eps
    
    ret = 0.0
    n = 0.0
    
    pred = pred.flatten()
    
    for i in range(len(label)):        
        
        ret += label[i] * np.log(pred[i]+eps) + (1.-label[i]) * np.log((eps1) - pred[i])
        
        n += 1.0
    loss = -1.0 *ret / n
    
    return loss

def f1_score(label, pred):
    eps = 1e-8# Avoid -inf when taking log(0)
    eps1 = 1. + eps
    eps_1 = 1. - eps
    
    precision = eps
    n_precision = eps
    recall = eps
    n_recall = eps
    
    pred = pred.flatten()
    
    for i in range(len(label)):
        pred[i] = round(pred[i])
        label[i] = round(label[i])
        
        if (pred[i] == 0):
            # precision
            if (pred[i] ==label[i]):
                precision+=1.0
            n_precision += 1.0
            
        if (label[i] == 0):
            # recall
            if (pred[i] ==label[i]):
                recall+=1.0
            n_recall += 1.0
            
    precision = precision/n_precision
    recall = recall/n_recall
    
    f1_score = 2*(precision*recall)/(precision+recall)
    
    loss = 1-f1_score

    return loss

def train(network, data_pair, num_epoch, learning_rate, optimizer='sgd', opt_args=None, ctx=[mx.gpu(0)]):
    np.random.seed(123)  # Fix random seed for consistent demos
    mx.random.seed(123)  # Fix random seed for consistent demos
    if not opt_args:
        opt_args = {}
    if optimizer=='sgd' and (not opt_args):
        opt_args['momentum'] = 0.9

    model = mx.model.FeedForward(
        ctx = ctx,
        symbol = network,
        num_epoch = num_epoch,
        optimizer = optimizer,
        learning_rate = learning_rate,
        wd = 1e-4,
        **opt_args
    )

    train, test = (data_pair)

    lc = mxnet.notebook.callback.LiveLearningCurve('f1_score', 1)
    model.fit(X = train,
              eval_data = test,
              eval_metric = f1_score,
              **mxnet.notebook.callback.args_wrapper(lc)
              )
    return lc
