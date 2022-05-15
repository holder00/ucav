# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 18:37:06 2022

@author: OokiT1
"""
import pickle
def save_weights(policy_id,trainer):
    a = trainer.get_weights(policy_id)
    f = open(policy_id+"weights"+".pkl",'wb')
    pickle.dump(a,f)
    f.close()
    
def load_weights(policy_id):
    f = open(policy_id+"weights"+".pkl",'rb')
    a = pickle.load(f)
    f.close()
    
    return a

def reload_weights(policy_id,trainer,set_policy_id):
    weight = load_weights(set_policy_id)
    trainer.set_weights({policy_id:weight[set_policy_id]})

