from scipy.stats.stats import kendalltau
from numpy import trapz
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(5, 2)

os.chdir('/home/nahid/trec_eval.9.0/')

#base_address1 = "/home/nahid/UT_research/clueweb12/bpref_result/"
#plotAddress = "/home/nahid/UT_research/clueweb12/bpref_result/plots/tau/mapbpref/"
#baseAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/"


base_address1 = "/media/nahid/Windows8_OS/TREC/"
plotAddress =  "/media/nahid/Windows8_OS/TREC/"

ht_estimation = False
crowd = True

dataset_list = ['WT2014','WT2013', 'gov2','TREC8']

if ht_estimation == True:
    print "TRUE"
    base_address1 = base_address1 + "estimationHTSPL/"
    plotAddress = plotAddress + "estimationHTSPL/plots/"

if crowd == True:
    print "TRUE"
    base_address1 = base_address1 + "estimationCrowd/"
    plotAddress = plotAddress + "estimationCrowd/plots/"
    dataset_list = ['WT2014']

protocol_list = ['SAL','CAL', 'SPL']
#dataset_list = ['gov2']
#map_list = ['map']
#map_list = ['bpref']


#dataset_list = ['TREC8']

ranker_list = ['True']
sampling_list = ['True']
map_list = ['map', 'bpref']
train_per_centage_flag = 'True'
seed_size =  [10] #50      # number of samples that are initially labeled
batch_size = [25] #50
train_per_centage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # skiping seed part which is named as 0.1
#train_per_centage = [0.1]
#train_per_centage = [0.2, 0.3] # skiping seed part which is named as 0.1
x_labels_set_name = ['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%']
#x_labels_set =[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
x_labels_set =[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#x_labels_set =[10,20]



result_location = ''
counter = 0
missing = 0
list = []
protocol_result = {}
#subplot_loc = [521, 522, 523, 524,525, 526, 527, 528, 529]
#subplot_loc = [331, 332, 333, 334,335, 336, 337, 338, 339]
#subplot_loc = [221,222,223,224]

var = 1
stringUse = ''
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20,6))
for stringUse in map_list:
    for use_ranker in ranker_list:
        for iter_sampling in sampling_list:


            s = ""
            s1 = ""
            originalqrelMap = []
            predictedqrelMap = []
            for datasource in dataset_list:  # 1
                originalqrelMap = []
                predictedqrelMap = []
                if datasource == 'gov2':
                    originAdress = "/media/nahid/Windows8_OS/unzippedsystemRanking/" + datasource + "/"
                    #qrelAdress = '/media/nahid/Windows8_OS/finalDownlaod/TREC/gov2/qrels.tb06.top50.txt'
                    qrelAdress = '/media/nahid/Windows8_OS/finalDownlaod/TREC/gov2/modified_qreldocsgov2.txt'
                    originalMapResult = '/media/nahid/Windows8_OS/finalDownlaod/TREC/gov2/'
                    destinationBase = "/media/nahid/Windows8_OS/modifiedSystemRanking/" + datasource + "/"
                    predictionAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/gov2/prediction/"
                    predictionModifiedAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/gov2/modifiedprediction/"
                    alpha_param = 1
                    train_per_centage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                         1.0,]  # skiping seed part which is named as 0.1
                    lambda_param = 0.75
                    alpha_param = 2

                elif datasource == 'TREC8':
                    originAdress = "/media/nahid/Windows8_OS/unzippedsystemRanking/" + datasource + "/"
                    qrelAdress = '/media/nahid/Windows8_OS/finalDownlaod/TREC/TREC8/relevance.txt'
                    originalMapResult = '/media/nahid/Windows8_OS/finalDownlaod/TREC/TREC8/'
                    destinationBase = "/media/nahid/Windows8_OS/modifiedSystemRanking/" + datasource + "/"
                    predictionAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/TREC8/prediction/"
                    predictionModifiedAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/TREC8/modifiedprediction/"
                    lambda_param = 0.75
                    alpha_param = 1
                    train_per_centage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                         1.0, ]  # skiping seed part which is named as 0.1

                elif datasource == 'WT2013':
                    originAdress = "/media/nahid/Windows8_OS/unzippedsystemRanking/" + datasource + "/"
                    qrelAdress = '/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2013/modified_qreldocs2013.txt'
                    originalMapResult = '/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2013/'
                    destinationBase = "/media/nahid/Windows8_OS/modifiedSystemRanking/" + datasource + "/"
                    predictionAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2013/prediction/"
                    predictionModifiedAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2013/modifiedprediction/"
                    lambda_param = 0.75
                    alpha_param = 2
                    train_per_centage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                         1.0, ]  # skiping seed part which is named as 0.1


                else:
                    originAdress = "/media/nahid/Windows8_OS/unzippedsystemRanking/" + datasource + "/"
                    qrelAdress = '/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2014/modified_qreldocs2014.txt'
                    originalMapResult = '/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2014/'
                    destinationBase = "/media/nahid/Windows8_OS/modifiedSystemRanking/" + datasource + "/"
                    predictionAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2014/prediction/"
                    predictionModifiedAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2014/modifiedprediction/"
                    train_per_centage = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                         1.0, 1.1]  # skiping seed part which is named as 0.1

                    if crowd == True:
                        alpha_param = 2
                        lambda_param = 1.0
                        qrelAdress = '/media/nahid/Windows8_OS/clueweb12/qrels/qrelsadhoc2014crowd.txt'

                    else:
                        alpha_param = 2
                        lambda_param = 1.0

                print "Original Part"


                ######################
                fileList = os.listdir(originAdress)
                for fileName in fileList:
                    system = originAdress + fileName
                    #shellCommand = './trec_eval -m map ' + qrelAdress + ' ' + system
                    shellCommand = './trec_eval -m '+stringUse+' ' + qrelAdress + ' ' + system

                    print shellCommand
                    import subprocess
                    p = subprocess.Popen(shellCommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    for line in p.stdout.readlines():
                        print line
                        values = line.split()
                        map = float(values[2])
                        originalqrelMap.append(map)

                    retval = p.wait()


                originalMapResult = originalMapResult + stringUse+'.txt'
                tmp = ""

                for val in originalqrelMap:
                    tmp = tmp + str(val) + ","
                text_file = open(originalMapResult, "w")
                text_file.write(tmp)
                text_file.close()

                #exit(0)


                #originalMapResult = originalMapResult + stringUse+'.txt'
                f = open(originalMapResult)
                length = 0
                tmplist = []
                for lines in f:
                    values = lines.split(",")
                    for val in values:
                        if val == '':
                            continue
                        tmplist.append(float(val))
                        length = length + 1
                    break
                originalqrelMap = tmplist



                base_address2 = base_address1 + str(datasource) + "/result/"
                if use_ranker == 'True':
                    base_address3 = base_address2 + "ranker/"
                    s1 = "Ranker and "
                else:
                    base_address3 = base_address2 + "no_ranker/"
                    s1 = "Interactive Search and "
                if iter_sampling == 'True':
                    base_address4 = base_address3 + "oversample/"
                    s1 = s1 + "oversampling"
                else:
                    base_address4 = base_address3 + "htcorrection/"
                    s1 = s1 + "HT correction"

                training_variation = []
                base_address4 = base_address4 + str(lambda_param) + "/" + str(alpha_param) + "/"
                for seed in seed_size:  # 2
                    for batch in batch_size:  # 3
                        for protocol in protocol_list:  # 4
                            print "Dataset", datasource, "Protocol", protocol, "Seed", seed, "Batch", batch
                            s = "Dataset:" + str(datasource) + ", Seed:" + str(seed) + ", Batch:" + str(batch)
                            list = []
                            for fold in xrange(1, 2):
                                predicted_location_base = base_address4 + 'prediction_protocol:' + protocol + '_batch:' + str(
                                    batch) + '_seed:' + str(seed) + '_fold' + str(fold) + '_'
                                for percentage in train_per_centage:

                                    ##############################

                                    predictionQrel = ""
                                    if stringUse == 'bpref':
                                        predictionQrel = predicted_location_base + str(
                                            percentage) + '_human_.txt'
                                        print predictionQrel
                                    else:
                                        predictionQrel = predicted_location_base + str(
                                        percentage) + '.txt'
                                        print predictionQrel

                                    predictionModifiedQrel = predicted_location_base + str(
                                        percentage) + '_human_modified.txt'

                                    # reading the prediction file and modifyinh it insert one extra column at position 2
                                    f = open(predictionQrel)
                                    #print f
                                    tmpstring = ""
                                    for lines in f:
                                        values = lines.split()
                                        tmpstring = tmpstring + values[0] + " 0 "+values[1]+" "+values[2]+"\n"

                                    text_file = open(predictionModifiedQrel, "w")
                                    text_file.write(tmpstring)
                                    text_file.close()


                                    fileList = os.listdir(originAdress)
                                    for fileName in fileList:
                                        system = originAdress + fileName
                                        # shellCommand = './trec_eval -m map ' + qrelAdress + ' ' + system
                                        shellCommand = './trec_eval -m '+stringUse +' '+ predictionModifiedQrel + ' ' + system

                                        print shellCommand
                                        import subprocess
                                        p = subprocess.Popen(shellCommand, shell=True, stdout=subprocess.PIPE,
                                                             stderr=subprocess.STDOUT)
                                        for line in p.stdout.readlines():
                                            print line
                                            values = line.split()
                                            map = float(values[2])
                                            predictedqrelMap.append(map)

                                        retval = p.wait()

                                    predictionMapResult = predicted_location_base + str(
                                        percentage) + '_' + stringUse + '.txt'

                                    tmp = ""

                                    for val in predictedqrelMap:
                                        tmp = tmp + str(val) + ","
                                    text_file = open(predictionMapResult, "w")
                                    text_file.write(tmp)
                                    text_file.close()

                                    #####################


                                    predictedqrelMap = []

                                    predictionMapResult = predicted_location_base + str(percentage) + '_'+ stringUse+'.txt'
                                    f = open(predictionMapResult)
                                    length = 0
                                    tmplist = []
                                    for lines in f:
                                        values = lines.split(",")
                                        for val in values:
                                            if val == '':
                                                continue
                                            tmplist.append(float(val))
                                            length = length + 1
                                        break
                                    predictedqrelMap = tmplist
                                    print len(predictedqrelMap)
                                    tau, p_value = kendalltau(originalqrelMap, predictedqrelMap)
                                    predictedqrelMap = []  # cleaning it for next trains_percenatge
                                    list.append(tau)


                            protocol_result[protocol] = list

                print len(training_variation)

                auc_SAL = trapz(protocol_result['SAL'], dx=10)
                auc_CAL = trapz(protocol_result['CAL'], dx=10)
                auc_SPL = trapz(protocol_result['SPL'], dx=10)

                print auc_SAL, auc_CAL, auc_SPL
                print var
                plt.subplot(2,4,var)

                plt.plot(x_labels_set, protocol_result['CAL'], '-b', marker='^', label='CAL, AUC:' + str(auc_CAL)[:4],
                         linewidth=2.0)

                plt.plot(x_labels_set, protocol_result['SAL'], '-r', marker='o', label='SAL, AUC:'+str(auc_SAL)[:4], linewidth=2.0)
                plt.plot(x_labels_set, protocol_result['SPL'], '-g', marker='D', label='SPL, AUC:'+str(auc_SPL)[:4], linewidth=2.0)



                if var == 1:
                    plt.ylabel('tau correlation \n using MAP',size = 16)
                if var == 5:
                    plt.ylabel('tau correlation \n using bpref', size=16)

                if var >=5:
                    plt.xlabel('% of human judgments', size=16)

                plt.ylim([0.7, 1])
                plt.yticks([0.7, 0.8, .9, 1.0])
                plt.legend(loc=4)
                param = "($\\alpha$ = " + str(alpha_param) + ", $\lambda$ = " + str(lambda_param) + ")"
                if datasource == 'gov2':
                    plt.title('TB\'06 ' + param, size=16)
                elif datasource == 'WT2013':
                    plt.title('WT\'13 ' + param, size=16)
                elif datasource == 'WT2014':
                    plt.title('WT\'14 ' + param, size=16)
                else:
                    plt.title('Adhoc\'99 ' + param, size=16)
                plt.grid()
                var = var + 1

#plt.suptitle(s1, size=16)
plt.tight_layout()
plt.savefig(plotAddress + s1 + 'map_bpref_crowd.pdf', format='pdf')





