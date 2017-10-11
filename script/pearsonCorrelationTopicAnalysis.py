
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
topicSkipList = [202, 209, 225, 237, 245, 255,269, 278, 803]
datasource = 'WT2014'

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


stats_file_location = "/home/nahid/UT_research/topicDifficulty/"+datasource+".txt"

f = open(stats_file_location)
print f
tmplist = []
# g = 0
ratioList = []
mapList = []
for lines in f:

    values = lines.split(",")
    topic = int(values[0])
    ratio = float(values[3])

    originalMapResult = "/home/nahid/UT_research/mapByTopic/" + datasource + "/" + str(topic) + '_map.txt'
    print originalMapResult
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
    import numpy as np
    averageMap = np.mean(originalqrelMap)
    mapList.append(averageMap)
    ratioList.append(ratio)

print pearsonr(mapList, ratioList)
plt.plot(mapList, ratioList)
plt.show()