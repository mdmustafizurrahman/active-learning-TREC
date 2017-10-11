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

#topicSkipList = [202,225,255,278]
topicSkipList = [202, 225, 237, 245, 255,269, 278, 803]
#topicSkipList = [202,210,225,234,235,238,244,251,255,262,269,271,278,283,289,291,803,805]

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


#print result_location
#exit(0)
class relevance(object):
    def __init__(self, priority, index):
        self.priority = priority
        self.index = index
        return
    def __cmp__(self, other):
        return -cmp(self.priority, other.priority)

def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    #review_text = BeautifulSoup(raw_review).get_text()
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", raw_review)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return( " ".join( meaningful_words ))


def run(trn_ds, tst_ds, lbr, model, qs, quota, batch_size):
    E_in, E_out = [], []

    for _ in range(quota):


        # Standard usage of libact objects
        ask_id = qs.make_query()
        #print  ask_id
        X, _ = zip(*trn_ds.data)
        lb = lbr.label(X[ask_id])
        trn_ds.update(ask_id, lb)


        model.train(trn_ds)
        #model.predict(tst_ds)
        E_in = np.append(E_in, 1 - model.score(trn_ds))
        E_out = np.append(E_out, 1 - model.score(tst_ds))

    return E_in, E_out, model




all_reviews = {}
learning_curve = {} # per batch value for  validation set

if preloaded==False:
    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        #print name
        #if name == "ft":
        path = os.path.join(TEXT_DATA_DIR, name)
        print path

        f = open(path)



        docNo = name[0:name.index('.')]
        #print docNo

        # counting the line number until '---Terms---'
        count = 0
        for lines in f:
            if lines.find("Terms")>0:
                count = count + 1
                break
            count = count + 1

        # skipping the lines until  '---Terms---' and reading the rest
        c = 0
        tmpStr = ""
        #print "count:", count
        #f = open(path)
        for lines in f:
            if c < count:
                c = c + 1
                continue
            values = lines.split()
            c = c + 1
            #print values[0], values[1], values[2]
            tmpStr = tmpStr + " "+ str(values[2])
        print tmpStr
        #exit(0)

        #if docNo in docNo_label:
        all_reviews[docNo] = (review_to_words(tmpStr))

        f.close()

    output = open(processed_file_location, 'ab+')
    # data = {'a': [1, 2, 3], }

    pickle.dump(all_reviews, output)
    output.close()

else:
    input = open(processed_file_location, 'rb')
    all_reviews = pickle.load(input)
    print "pickle loaded"


print('Reading the Ranker label Information')
f = open(ranker_location[datasource])
print "Ranker:", f
tmplist = []
Ranker_topic_to_doclist = {}
for lines in f:
    values = lines.split()
    topicNo = values[0]
    docNo = values[2]
    if (Ranker_topic_to_doclist.has_key(topicNo)):
        tmplist.append(docNo)
        Ranker_topic_to_doclist[topicNo] = tmplist
    else:
        tmplist = []
        tmplist.append(docNo)
        Ranker_topic_to_doclist[topicNo] = tmplist
f.close()
# print len(topic_to_doclist)


for topic, list in Ranker_topic_to_doclist.iteritems():
    print topic, len(list)

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
                docNo_label[docNo] = label
                if (topic_to_doclist.has_key(topicNo)):
                    tmplist.append(docNo)
                    topic_to_doclist[topicNo] = tmplist
                else:
                    tmplist = []
                    tmplist.append(docNo)
                    topic_to_doclist[topicNo] = tmplist
            f.close()
            #print len(topic_to_doclist)
            docList = topic_to_doclist[topic]
            print 'number of documents', len(docList)
            #print docList
            #print ('Processing news text for topic number')
            relevance_label = []
            judged_review = []

            docIndex = 0
            for documentNo in docList:
                if all_reviews.has_key(documentNo):
                    #print "in List", documentNo
                    #print documentNo, 'len:', type(all_reviews[documentNo])

                    #print all_reviews[documentNo]
                    #exit(0)
                    docIndex_DocNo[docIndex] = documentNo
                    docNo_docIndex[documentNo] = docIndex
                    docIndex = docIndex + 1
                    judged_review.append(all_reviews[documentNo])
                    relevance_label.append(docNo_label[documentNo])


            if docrepresentation == "TF-IDF":
                print "Using TF-IDF"
                vectorizer = TfidfVectorizer( analyzer = "word",   \
                                         tokenizer = None,    \
                                         preprocessor = None, \
                                         stop_words = None,   \
                                         max_features = 15000)

                bag_of_word = vectorizer.fit_transform(judged_review)


            elif docrepresentation == "BOW":
                # Initialize the "CountVectorizer" object, which is scikit-learn's
                # bag of words tool.
                print "Uisng Bag of Word"
                vectorizer = CountVectorizer(analyzer = "word",   \
                                             tokenizer = None,    \
                                             preprocessor = None, \
                                             stop_words = None,   \
                                             max_features = 15000)

                # fit_transform() does two functions: First, it fits the model
                # and learns the vocabulary; second, it transforms our training data
                # into feature vectors. The input to fit_transform should be a list of
                # strings.
                bag_of_word = vectorizer.fit_transform(judged_review)

            # Numpy arrays are easy to work with, so convert the result to an
            # array
            bag_of_word = bag_of_word.toarray()
            print bag_of_word.shape
            #vocab = vectorizer.get_feature_names()
            #print vocab

            # Sum up the counts of each vocabulary word
            #dist = np.sum(bag_of_word, axis=0)

            # For each, print the vocabulary word and the number of times it
            # appears in the training set
            #for tag, count in zip(vocab, dist):
            #    print count, tag

            print "Bag of word completed"

            X= bag_of_word
            y= relevance_label

            # print len(y)
            # print y
            numberOne = y.count(1)
            # print "Number of One", numberOne

            numberZero = y.count(0)
            print "Number of One", numberOne
            print "Number of Zero", numberZero
            datasize = len(X)
            prevelance = (numberOne * 1.0) / datasize
            # print "Number of zero", numberZero


            '''
            print type(X)
            print X[0]
            print len(X[0])
            print type(y)
            X = pd.DataFrame(bag_of_word)
            y = pd.Series(relevance_label)

            print type(X)
            print len(X)
            '''
            #exit(0)
            print "=========Before Sampling======"

            print "Whole Dataset size: ", datasize
            print "Number of Relevant", numberOne
            print "Number of non-relevant", numberZero
            print "prevelance ratio", prevelance * 100

            #print "After", y_train


            print '----Started Training----'
            model = LogisticRegression()
            size = len(X) - n_labeled

            if size<0:
                print "Train Size:", len(X) , "seed:", n_labeled
                size = len(X)

            if use_ranker == True:

                initial_X_train = []
                initial_y_train = []

                train_index_list = []

                # collecting the seed list from the Rankers
                seed_list = Ranker_topic_to_doclist[topic]
                print "number of documents under topic " + topic + " for RDS: " + str(len(seed_list))
                seed_counter = 0
                seed_one_counter = 0
                seed_zero_counter = 0
                ask_for_label = 0
                loopCounter = 0

                seed_size_limit = math.ceil(train_per_centage[loopCounter] * len(X))
                print "Initial Seed Limit", seed_size_limit
                seed_start = 0
                seed_counter = 0

                while seed_one_counter < 5:
                    documentNumber = seed_list[seed_counter]
                    seed_counter = seed_counter + 1
                    if documentNumber not in docNo_docIndex:
                        continue
                    index = docNo_docIndex[documentNumber]
                    train_index_list.append(index)
                    labelValue = int(docNo_label[documentNumber])
                    ask_for_label = ask_for_label + 1
                    initial_X_train.append(X[index])
                    initial_y_train.append(labelValue)
                    if labelValue == 1:
                        seed_one_counter = seed_one_counter + 1
                    if labelValue == 0:
                        seed_zero_counter = seed_zero_counter + 1

                print "Topic:"+topic+", total_documents_visited: "+str(seed_counter)+", number_of_related_docs:"+str(seed_one_counter)+"\n"
                ratio = (seed_one_counter * 1.0 ) / seed_counter
                stats = stats + topic + "," + str(seed_counter) + "," + str(seed_one_counter) + ", "+ str(ratio) + "\n"
                        #print seed_one_counter, seed_zero_counter


            #text_file = open(predicted_location, "w")
            #text_file.write(pred_str)
            #text_file.close()

stats_file_location = "/home/nahid/UT_research/topicDifficulty/"+datasource+".txt"
text_file = open(stats_file_location, "w")
text_file.write(stats)
text_file.close()

