#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 22:05:21 2017

@author: apm13
"""
import numpy as np
import matplotlib.pyplot as plt
import csv

def csv_writer(data, path):
    """
    Write data to a CSV file path
    """
    with open(path, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)

SEQ_LEN = 10
def create_time_series():
    freq = (np.random.random()*0.5) + 0.1  # 0.1 to 0.6
    ampl = np.random.random() + 0.5  # 0.5 to 1.5
    x = np.sin(np.arange(0,SEQ_LEN) * freq) * ampl
    return x
  
if __name__ == "__main__" :
    #==============================================================================
    # Generate data.     
    #==============================================================================
    data = [0]*80
    for i in range(8):
        tmp = create_time_series()
    
    data2 = zip(range(len(data)), data)
    csv_writer(data2, "data/valid"+str(i)+".csv",)
    

    