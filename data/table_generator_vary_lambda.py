import os
from scipy.integrate import simps
from numpy import trapz
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(5, 2)

baseAddress = "/media/nahid/Windows8_OS/TREC/"

base_address1 = "/media/nahid/Windows8_OS/TREC/"
plotAddress =  "/media/nahid/Windows8_OS/TREC/"


protocol_list = ['SAL', 'CAL', 'SPL']
#dataset_list = ['WT2013','WT2014']
dataset_list = ['WT2014', 'WT2013','gov2', 'TREC8']
ranker_list = ['True', 'False']
sampling_list = ['True','False']
train_per_centage_flag = 'True'
train_per_centage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
x_labels_set_name = ['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%']
#x_labels_set =[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
x_labels_set =[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
linestyles = ['_', '-', '--', ':']

lambda_list = [0.0, 0.25, 0.50, 0.75, 1.0]
alpha_list = [1, 2]
n_labeled =  10 #50      # number of samples that are initially labeled
seed_size =  [10] #50      # number of samples that are initially labeled
batch_size = [25] #50

use_ranker = 'True'
iter_sampling = 'True'
correction = 'False'
train_per_centage_flag = 'True'


ht_estimation = False
crowd = True



result_location = ''
counter = 0
missing = 0
list = []
protocol_result = {}

base_address1 = "/media/nahid/Windows8_OS/TREC/"

'''
if ht_estimation == True:
    #print "TRUE"
    base_address1 = base_address1 + "estimation/"
    plotAddress = plotAddress + "estimation/"
    protocol_list = ['CAL', 'SAL', 'SPL']

'''
#estimation with HT SPL
if ht_estimation == True:
    #print "TRUE"
    base_address1 = base_address1 + "estimationHTSPL/"
    plotAddress = plotAddress + "estimationHTSPL/plots/"
    protocol_list = ['CAL', 'SAL', 'SPL']

if crowd == True:
    print "TRUE"
    base_address1 = base_address1 + "estimationCrowd/"
    plotAddress = plotAddress + "estimationCrowd/plots/"
    dataset_list = ['WT2014']

for alpha_param in alpha_list:
    best_AUC = 0.0
    best_protocol = ''
    best_lambda = -1.0
    best_alpha = 0
    for lambda_param in lambda_list:
        fig, ax = plt.subplots(nrows=5, ncols=4, figsize=(20, 15))
        var = 1
        print "$\\alpha$ = ", str(alpha_param),", & $\lambda$ = ", str(lambda_param),
        for datasource in dataset_list:  # 1

            base_address2 = base_address1 + str(datasource) + "/"
            base_address2 = base_address2 + "result/"
            if use_ranker == 'True':
                base_address3 = base_address2 + "ranker/"
                s1="RDS "
            else:
                base_address3 = base_address2 + "no_ranker/"
                s1 = "IS "
            if iter_sampling == 'True':
                base_address4 = base_address3 + "oversample/"
                s1 = s1+"with Oversampling"
            else:
                base_address4 = base_address3 + "oversample/"
                s1 = s1+"w/o Oversampling"

            base_address4 = base_address4 + str(lambda_param)+"/" + str(alpha_param) + "/"
            training_variation = []
            for seed in seed_size: # 2
                for batch in batch_size: # 3
                    for protocol in protocol_list: #4
                            #print "Dataset", datasource,"Protocol", protocol, "Seed", seed,"Batch", batch
                            s = "Dataset:"+ str(datasource)+", Seed:" + str(seed) + ", Batch:"+ str(batch)

                            for fold in xrange(1, 2):
                                learning_curve_location = base_address4 + 'learning_curve_protocol:' + protocol + '_batch:' + str(
                                    batch) + '_seed:' + str(seed) + '_fold' + str(fold) + '.txt'

                            list = []

                            f = open(learning_curve_location)
                            length = 0
                            for lines in f:
                                values = lines.split(",")
                                for val in values:
                                    if val == '':
                                        continue
                                    list.append(float(val))
                                    length = length + 1
                                break
                            #print list

                            #if datasource == 'TREC8' or datasource == 'gov2':
                            list = list[1:11]
                            #list1 = list[1:len(list)]
                            #if use_ranker == "True":
                            #    list1 = list[0:len(list)-2]
                            #    list1.append(list[len(list)-1])
                            #else:
                            #    list1 = list[1:len(list)]
                            #print length
                            counter = 0
                            protocol_result[protocol] = list
                            if protocol == 'SAL':
                                start = 10
                                end = start + (length - 1)*25
                                while start <= end:
                                    training_variation.append(start)
                                    start = start + 25


            #print len(training_variation)
            auc_SAL = str(trapz(protocol_result['SAL'], dx=10))[:5]
            auc_CAL = str(trapz(protocol_result['CAL'], dx=10))[:5]
            auc_SPL = str(trapz(protocol_result['SPL'], dx=10))[:5]


            print "&", auc_CAL, "&", auc_SAL, "&", auc_SPL,

            #print auc_SAL, auc_CAL, auc_SPL

            protocol_names = ['CAL', 'SAL', 'SPL']
            max_AUC = max(auc_CAL, auc_SAL, auc_SPL)
            import numpy as np
            max_index = np.argmax([auc_CAL, auc_SAL, auc_SPL])
            max_protocol_name = protocol_names[max_index]

            if max_AUC > best_AUC:
                best_AUC = max_AUC
                best_alpha = alpha_param
                best_lambda = lambda_param
                best_protocol = max_protocol_name
        print "\\\\ \n"







    #print "Datasource:", datasource, "best AUC:", best_AUC, "best_alpha:", best_alpha, "best_lambda:", best_lambda, "best_protocol:", best_protocol