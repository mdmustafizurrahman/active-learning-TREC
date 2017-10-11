from scipy.stats.stats import kendalltau
from numpy import trapz
import os
import subprocess
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(5, 2)

os.chdir('/home/nahid/Downloads/trec_eval.9.0/')

base_address1 = "/home/nahid/UT_research/clueweb12/bpref_result/"
plotAddress = "/home/nahid/UT_research/clueweb12/bpref_result/plots/tau/mapbpref/"
baseAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/"


protocol_list = ['SAL','CAL', 'SPL']
dataset_list = ['WT2013','WT2014']
#dataset_list = ['gov2', 'TREC8']
ranker_list = ['False']
sampling_list = ['True']
map_list = ['True', 'False']
train_per_centage_flag = 'True'
seed_size =  [10] #50      # number of samples that are initially labeled
batch_size = [25] #50
train_per_centage = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1] # skiping seed part which is named as 0.1
#train_per_centage = [0.2, 0.3] # skiping seed part which is named as 0.1
x_labels_set_name = ['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%']
#x_labels_set =[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
x_labels_set =[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#x_labels_set =[10,20]
topicSkipList = [202, 209, 225, 237, 245, 255,269, 278, 803]

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
for use_map in map_list:
    if use_map == "True":
        stringUse = 'map'
    else:
        stringUse = 'bpref'

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
        start_topic = 801
        end_topic = 851

    elif datasource == 'TREC8':
        originAdress = "/media/nahid/Windows8_OS/unzippedsystemRanking/" + datasource + "/"
        qrelAdress = '/media/nahid/Windows8_OS/finalDownlaod/TREC/TREC8/relevance.txt'
        originalMapResult = '/media/nahid/Windows8_OS/finalDownlaod/TREC/TREC8/'
        destinationBase = "/media/nahid/Windows8_OS/modifiedSystemRanking/" + datasource + "/"
        predictionAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/TREC8/prediction/"
        predictionModifiedAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/TREC8/modifiedprediction/"
        start_topic = 401
        end_topic = 451
    elif datasource == 'WT2013':
        originAdress = "/media/nahid/Windows8_OS/unzippedsystemRanking/" + datasource + "/"
        qrelAdress = '/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2013/modified_qreldocs2013.txt'
        originalMapResult = '/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2013/'
        destinationBase = "/media/nahid/Windows8_OS/modifiedSystemRanking/" + datasource + "/"
        predictionAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2013/prediction/"
        predictionModifiedAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2013/modifiedprediction/"
        start_topic = 201
        end_topic = 251

    else:
        originAdress = "/media/nahid/Windows8_OS/unzippedsystemRanking/" + datasource + "/"
        qrelAdress = '/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2014/modified_qreldocs2014.txt'
        originalMapResult = '/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2014/'
        destinationBase = "/media/nahid/Windows8_OS/modifiedSystemRanking/" + datasource + "/"
        predictionAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2014/prediction/"
        predictionModifiedAddress = "/media/nahid/Windows8_OS/finalDownlaod/TREC/WT2014/modifiedprediction/"
        start_topic = 251
        end_topic = 301

    print "Original Part"

    fileList = os.listdir(originAdress)

    for topic in xrange(start_topic, end_topic):
        originalqrelMap = []
        predictedqrelMap = []
        print "Topic:", topic
        if topic in topicSkipList:
            print "Skipping Topic :", topic
            continue
        qrelAdress = "/home/nahid/UT_research/qrelsByTopic/" + datasource + "/" + str(topic) + ".txt"

        for fileName in fileList:
            system = originAdress + fileName
            #shellCommand = './trec_eval -m map ' + qrelAdress + ' ' + system
            shellCommand = './trec_eval -m map ' + qrelAdress + ' ' + system

            print shellCommand
            p = subprocess.Popen(shellCommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in p.stdout.readlines():
                print line
                values = line.split()
                map = float(values[2])
                originalqrelMap.append(map)

            retval = p.wait()


        originalMapResult = "/home/nahid/UT_research/mapByTopic/" + datasource + "/" + str(topic) + '_map.txt'
        tmp = ""

        for val in originalqrelMap:
            tmp = tmp + str(val) + ","
        text_file = open(originalMapResult, "w")
        text_file.write(tmp)
        text_file.close()

        originalMapResult = "/home/nahid/UT_research/mapByTopic/" + datasource + "/" + str(topic) + '_map.txt'
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









