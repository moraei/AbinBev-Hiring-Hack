# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 18:40:52 2018

@author: vprayagala2
"""
import logging
#%%
def create_logger(log_file):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # create a file handler
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    # add the handlers to the logger
    logger.addHandler(handler)
    
    logger.info("Start of Log\n")
    return logger