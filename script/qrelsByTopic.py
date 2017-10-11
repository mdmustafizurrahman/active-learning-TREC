'''Active learning for labeling the relevant document for TREC-8 dataset
@author: Md Mustafizur Rahman (nahid@utexas.edu)'''

import os
import numpy as np
import sys
from bs4 import BeautifulSoup
import re
import math
import nltk
import copy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from libact.base.dataset import Dataset, import_libsvm_sparse
from libact.models import *
from libact.query_strategies import *
from libact.labelers import IdealLabeler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import collections
compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
import pickle
from math import log
import pandas as pd
import Queue

import logging
logging.basicConfig()

TEXT_DATA_DIR = '/home/nahid/UT_research/TREC/TREC8/IndriData/'
RELEVANCE_DATA_DIR = '/home/nahid/UT_research/TREC/TREC8/relevance.txt'
docrepresentation = "TF-IDF"  # can be BOW, TF-IDF
sampling=False # can be True or False
command_prompt_use = True

#if command_prompt_use == True:
'''
datasource = sys.argv[1] # can be  dataset = ['TREC8', 'gov2', 'WT']
protocol = sys.argv[2]
use_ranker = sys.argv[3]
iter_sampling = sys.argv[4]
correction = sys.argv[5] #'SAL' can be ['SAL', 'CAL', 'SPL']
train_per_centage_flag = sys.argv[6]
'''

#parameter set # all FLAGS must be string

datasource = 'WT2013'  # can be  dataset = ['TREC8', 'gov2', 'WT']
protocol = 'CAL'  # 'SAL' can be ['SAL', 'CAL', 'SPL']
use_ranker = 'True'
iter_sampling = 'True'
correction = 'False'
train_per_centage_flag = 'True'

preloaded = True

print "Ranker_use", use_ranker
print "iter_sampling", iter_sampling
print "correction", correction
print "train_percenetae", train_per_centage_flag


test_size = 0    # the percentage of samples in the dataset that will be
#test_size_set = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
test_size_set = [0.2]
train_per_centage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

ranker_location = {}
ranker_location["WT2013"] = "/media/nahid/Windows8_OS/unzippedsystemRanking/WT2013/input.ICTNET13RSR2"
ranker_location["WT2014"] = "/media/nahid/Windows8_OS/unzippedsystemRanking/WT2014/input.Protoss"
ranker_location["gov2"] = "/media/nahid/Windows8_OS/unzippedsystemRanking/gov2/input.indri06AdmD"
ranker_location["TREC8"] = "/media/nahid/Windows8_OS/unzippedsystemRanking/TREC8/input.ibmg99b"

n_labeled =  10 #50      # number of samples that are initially labeled
batch_size = 25 #50

topicSkipList = [202, 209, 225, 237, 245, 255,269, 278, 803]

skipList = []
topicBucketList = []
processed_file_location = ''
start_topic = 0
end_topic = 0

base_address = "/home/nahid/UT_research/clueweb12/nooversample_result1/"
base_address = base_address +str(datasource)+"/"
if use_ranker == 'True':
    base_address = base_address + "ranker/"
    use_ranker = True
else:
    base_address = base_address + "no_ranker/"
    use_ranker = False
if iter_sampling == 'True':
    base_address = base_address + "oversample/"
    iter_sampling = True
if iter_sampling == 'False':
    iter_sampling = False
if correction == 'True':
    base_address = base_address + "htcorrection/"
    correction = True
if correction == 'False':
    correction = False

if train_per_centage_flag == 'True':
    train_per_centage_flag = True
else:
    train_per_centage_flag = False

print "base address:", base_address


if iter_sampling == True and correction == True:
    print "Over sampling and HT correction cannot be done together"
    exit(-1)

#if iter_sampling == False and correction == False:
#    print "Over sampling and HT correction cannot be both false together"
#    exit(-1)


if datasource=='TREC8':
    processed_file_location = '/home/nahid/UT_research/TREC/TREC8/processed.txt'
    RELEVANCE_DATA_DIR = '/home/nahid/UT_research/TREC/TREC8/relevance.txt'
    start_topic = 401
    end_topic = 451
elif datasource=='gov2':
    processed_file_location = '/home/nahid/UT_research/TREC/gov2/processed.txt'
    RELEVANCE_DATA_DIR = '/home/nahid/UT_research/TREC/qrels.tb06.top50.txt'
    start_topic = 801
    end_topic = 851
elif datasource=='WT2013':
    processed_file_location = '/media/nahid/Windows8_OS/clueweb12/pythonprocessed/processed_new.txt'
    RELEVANCE_DATA_DIR = '/media/nahid/Windows8_OS/clueweb12/qrels/qrelsadhoc2013.txt'
    start_topic = 201
    end_topic = 251
else:
    processed_file_location = '/media/nahid/Windows8_OS/clueweb12/pythonprocessed/processed_new.txt'
    RELEVANCE_DATA_DIR = '/media/nahid/Windows8_OS/clueweb12/qrels/qrelsadhoc2014.txt'
    start_topic = 251
    end_topic = 301



all_reviews = {}

stats = ''
for test_size in test_size_set:
    seed = 1335
    for fold in xrange(1,2):
        np.random.seed(seed)
        seed = seed + fold
        result_location = base_address + 'result_protocol:' + protocol + '_batch:' + str(batch_size) + '_seed:' + str(n_labeled) +'_fold'+str(fold)+ '.txt'
        predicted_location = base_address + 'prediction_protocol:' + protocol + '_batch:' + str(batch_size) + '_seed:' + str(n_labeled) +'_fold'+str(fold)+ '.txt'
        predicted_location_base = base_address + 'prediction_protocol:' + protocol + '_batch:' + str(batch_size) + '_seed:' + str(n_labeled) +'_fold'+str(fold) + '_'
        human_label_location = base_address + 'prediction_protocol:' + protocol + '_batch:' + str(batch_size) + '_seed:' + str(n_labeled) +'_fold'+str(fold) + '_'

        learning_curve_location = base_address + 'learning_curve_protocol:' + protocol + '_batch:' + str(batch_size) + '_seed:' + str(n_labeled) +'_fold'+str(fold)+ '.txt'
        s = "";
        pred_str = ""
        #for topic in sorted(topic_to_doclist.keys()):
        for topic in xrange(start_topic,end_topic):
            print "Topic:", topic
            if topic in topicSkipList:
                print "Skipping Topic :", topic
                continue
            topic = str(topic)

            topic_to_doclist = {}  # key is the topic(string) and value is the list of docNumber
            docNo_label = {}  # key is the DocNo and the value is the label
            docIndex_DocNo = {} # key is the index used in my code value is the actual DocNo
            docNo_docIndex = {} # key is the DocNo and the value is the index assigned by my code
            best_f1 = 0.0  # best f1 considering per iteraton of active learning
            print('Reading the relevance label')
            # file open
            f = open(RELEVANCE_DATA_DIR)
            print f
            tmplist = []
            #g = 0
            for lines in f:
                #print lines
                #g = g + 1
                #if g>2739:
                #    break
                values = lines.split()
                topicNo = values[0]
                if topicNo != topic:
                    #print "Skipping", topic, topicNo
                    continue
                docNo = values[2]
                label = int(values[3])
                if label > 1:
                    label = 1
                if label < 0:
                    label = 0
                #lines = values[0] + " "+ values[1] + " "+ values[2] + " "+ str(label) + "\n"
                docNo_label[docNo] = label
                if (topic_to_doclist.has_key(topicNo)):
                    tmplist.append(lines)
                    topic_to_doclist[topicNo] = tmplist
                else:
                    tmplist = []
                    tmplist.append(lines)
                    topic_to_doclist[topicNo] = tmplist
            f.close()


            qrels_by_topic_file_location = "/home/nahid/UT_research/qrelsByTopic/" + datasource + "/"+ topic + ".txt"
            text_file = open(qrels_by_topic_file_location, "w")
            stats = ''
            for topic, linelist in topic_to_doclist.iteritems():
                for line in linelist:
                    stats = stats + line
            text_file.write(stats)
            text_file.close()



