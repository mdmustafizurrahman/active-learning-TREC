from bs4 import BeautifulSoup
import os
import urllib2, base64
import os
import requests
from requests.auth import HTTPBasicAuth


fileList = []
fileList.append('/home/nahid/images/1.txt')
fileList.append('/home/nahid/images/2.txt')
fileList.append('/home/nahid/images/3.txt')
fileList.append('/home/nahid/images/4.txt')

s = ""

for fileName in fileList:
    print fileName
    file = open(fileName)
    soup = BeautifulSoup(file)

    for link in soup.find_all('a'):
        print link.next_element
        s = s +  link.next_element + "\n"
        # request = requests.get(downloadAddress, auth=('tipster', 'cdroms'))
        # request = urllib2.Request(downloadAddress)
        # base64string = base64.encodestring('%s:%s' % ('tipster', 'cdroms')).replace('\n', '')
        # request.add_header("Authorization", "Basic %s" % base64string)

        '''
        response = urllib2.urlopen(link)
        html = response.read()

        output = open('/home/nahid/images/' + fileName, "w")
        output.write(html.content)
        output.close()
        '''

output = open('/home/nahid/images/combined.txt', "w")
output.write(s)
output.close()
