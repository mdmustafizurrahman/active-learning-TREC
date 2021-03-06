import os
from scipy.integrate import simps
from numpy import trapz
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(5, 2)



base_address1 = "/media/nahid/Windows8_OS/TREC/"
plotAddress =  "/media/nahid/Windows8_OS/TREC/plots/"
protocol_list = ['SAL', 'CAL', 'SPL']

'''
base_address1 = "/media/nahid/Windows8_OS/estimation/"
plotAddress =  "/media/nahid/Windows8_OS/TREC/plots/"
protocol_list = ['SAL', 'CAL', 'SPL']
'''

ht_estimation = True
uniform_sampling = False # uniform topic sampling
deterministic = False

lambda_list = [0.0, 0.25, 0.50, 0.75, 1.0]
alpha_list = [1, 2]
number_of_rows = 5

'''# ht_estimation with normal SPL
if ht_estimation == True:
    print "TRUE"
    base_address1 = base_address1 + "estimation/"
    plotAddress = plotAddress + "estimation/"
'''
# ht_estimation with HT Corrected SPL
if ht_estimation == True:
    base_address1 = "/media/nahid/Windows8_OS/TREC/estimationHTSPL/"
    plotAddress = "/media/nahid/Windows8_OS/TREC/estimationHTSPL/plots/"
    lambda_list = [0.25]
    alpha_list = [2]
    number_of_rows = 1
    protocol_list = ['SPL', 'CAL', 'SAL']

'''
if uniform_sampling == True:
    base_address1 = base_address1 + "UniformTopicSPL/"
    plotAddress = plotAddress + "UniformTopicSPL/"
    lambda_list = [0.0]
    alpha_list = [1]
'''

if uniform_sampling == True:
    base_address1 = "/media/nahid/Windows8_OS/random2/"
    plotAddress = "/media/nahid/Windows8_OS/random2/plots/"
    lambda_list = [0.0]
    alpha_list = [1]
    number_of_rows = 1


# roundrobin with normal SPL
if deterministic == True:
    base_address1 = "/media/nahid/Windows8_OS/roundrobinBasicSPL1/"
    plotAddress = "/media/nahid/Windows8_OS/roundrobinBasicSPL1/plots/"
    lambda_list = [0.0]
    alpha_list = [1]
    number_of_rows = 1


'''
# roundrobin with HT SPL
if deterministic == True:
    base_address1 = "/media/nahid/Windows8_OS/roundrobinHTSPL1/"
    plotAddress = "/media/nahid/Windows8_OS/roundrobinHTSPL1/plots/"
    lambda_list = [0.0]
    alpha_list = [1]
    number_of_rows = 1
'''

protocol_list = ['SAL', 'CAL', 'SPL']
dataset_list = ['WT2014', 'WT2013','gov2','TREC8']
#dataset_list = ['WT2014']
#protocol_list = ['SL', 'CAL', 'SAL']
ranker_list = ['True', 'False']
sampling_list = ['True','False']
train_per_centage_flag = 'True'
train_per_centage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
x_labels_set_name = ['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%']
#x_labels_set =[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
x_labels_set =[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
linestyles = ['_', '-', '--', ':']


n_labeled =  10 #50      # number of samples that are initially labeled
seed_size =  [10] #50      # number of samples that are initially labeled
batch_size = [25] #50

use_ranker = 'True'
iter_sampling = 'True'
correction = 'False'
train_per_centage_flag = 'True'

result_location = ''
counter = 0
missing = 0
list = []
protocol_result = {}

for alpha_param in alpha_list:
    fig, ax = plt.subplots(nrows=number_of_rows, ncols=4, figsize=(20, 5))
    var = 1
    for lambda_param in lambda_list:
        for datasource in dataset_list:  # 1
            s=""

        for datasource in dataset_list: # 1
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
            print base_address4
            base_address4 = base_address4 + str(lambda_param)+"/" + str(alpha_param) + "/"
            training_variation = []
            for seed in seed_size: # 2
                for batch in batch_size: # 3
                    for protocol in protocol_list: #4

                            #if protocol == 'SPL':
                            #    base_address4 = "/media/nahid/Windows8_OS/TREC/"+str(datasource)+"/result/ranker/oversample/"+ str(lambda_param)+"/" + str(alpha_param) + "/"

                            print "Dataset", datasource,"Protocol", protocol, "Seed", seed,"Batch", batch
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
                            print list

                            #if datasource == 'TREC8' or datasource == 'gov2':
                            list = list[1:11]
                            #list1 = list[1:len(list)]
                            #if use_ranker == "True":
                            #    list1 = list[0:len(list)-2]
                            #    list1.append(list[len(list)-1])
                            #else:
                            #    list1 = list[1:len(list)]
                            #print length
                            auc_SAL = trapz(list, dx=10)
                            print auc_SAL

                            counter = 0
                            protocol_result[protocol] = list
                            if protocol == 'SAL':
                                start = 10
                                end = start + (length - 1)*25
                                while start <= end:
                                    training_variation.append(start)
                                    start = start + 25


            #plt.figure(var)
            print len(training_variation)
            #plt.subplot(subplot_loc[var])
            plt.subplot(number_of_rows,4, var)
            '''plt.plot(x_labels_set, protocol_result['SAL'], '-r', label='SAL',linewidth=2.0)
            plt.plot(x_labels_set, protocol_result['CAL'], '-b', label = 'CAL',linewidth=2.0)
            plt.plot(x_labels_set, protocol_result['SPL'], '-g', label= 'SPL',linewidth=2.0)
'''
            '''plt.plot(x_labels_set, protocol_result['SAL'],   marker='o', color = 'black',label='SAL', linewidth=1.0)
            plt.plot(x_labels_set, protocol_result['CAL'],  marker = '^',color = 'black',label='CAL', linewidth=1.0)
            plt.plot(x_labels_set, protocol_result['SPL'],  marker = 's',color = 'black', label='SPL', linewidth=1.0)
'''

            auc_SAL = trapz(protocol_result['SAL'], dx=10)
            print auc_SAL
            auc_CAL = trapz(protocol_result['CAL'], dx=10)
            #if ht_estimation == False:
            auc_SPL = trapz(protocol_result['SPL'], dx=10)

            #print auc_SAL, auc_CAL, auc_SPL
            #exit(0)

            print "Equality Check:", len(x_labels_set), len(protocol_result['CAL'])
            plt.plot(x_labels_set, protocol_result['CAL'], '-b', marker='^', label='CAL, AUC:' + str(auc_CAL)[:5],linewidth=2.0)

            plt.plot(x_labels_set, protocol_result['SAL'],  '-r', marker='o',  label='SAL, AUC:'+str(auc_SAL)[:5], linewidth=2.0)

            #if ht_estimation == False:
            plt.plot(x_labels_set, protocol_result['SPL'],  '-g', marker = 'D', label='SPL, AUC:'+str(auc_SPL)[:5], linewidth=2.0)

            if var > 16:
                plt.xlabel('% of human judgments', size = 16)

            #if var == 1 or var == 5 or var == 9 or var == 13 or var == 17:
            #    plt.ylabel("lambda = " + str(lambda_param)+'\n F1', size = 16)
                #plt.yticks(True)
            plt.ylim([0.5,1])
            #plt.tick_params(axis='x',          # changes apply to the x-axis
            #which='both',      # both major and minor ticks are affected
            #bottom='off',      # ticks along the bottom edge are off
            #top='off',         # ticks along the top edge are off
            #labelbottom='off') # labels along the bottom edge are off)
            #if var == 1:
            #if var == 7 or var == 8:
            #    plt.legend(loc=2, fontsize = 16)
            #else:
            plt.legend(loc=4, fontsize=16)
            if datasource == 'gov2':
                plt.title('TB\'06', size= 16)
            elif datasource == 'WT2013':
                plt.title('WT\'13', size = 16)
            elif datasource == 'WT2014':
                plt.title('WT\'14', size=16)
            else:
                plt.title('Adhoc\'99', size=16)
            plt.grid()
            var = var + 1

    #plt.suptitle("alpha = "+str(alpha_param), size=20)
    plt.tight_layout()

    #plt.show()
    plt.savefig(plotAddress+str(alpha_param)+'_with_complete_SPL.pdf', format='pdf')

    #exit(0)