import sys
'''
dataset = ['WT2013', 'WT2014']
variation = 61 # dataset WT 2013 and WT 2014 starts at 61 and we need 10 processor per job
mod = 10
filename1 = '/home/nahid/PycharmProjects/parser/scriptdeterministic/batch_command_WT.sh'
'''

'''
dataset = ['gov2']
variation = 31 # dataset gov2 starts at 31 and we need 3 processor per job
mod = 3
filename1 = '/home/nahid/PycharmProjects/parser/scriptdeterministic/batch_command_gov2.sh'
'''

dataset = ['WT2013', 'WT2014', 'gov2','TREC8']
variation = 1 # dataset starts at 1 and we need 2 processor per job
mod = 12
filename1 = '/home/nahid/PycharmProjects/parser/scriptdeterministic/batch_command_TREC8.sh'
protocol_list = ['SAL','CAL','SPL']

#protocol_list = ['SAL', 'CAL', 'SPL']
# deterministic so no lambda and alpha
lambda_list = [0.0]
alpha_list = [1]

#protocol_list = ['Basic']
#batch_size = [25]
#seed_size = [30, 50, 70]



shellcommand = '#!/bin/sh\n'
s=''
for datasource in dataset: # 1
    for protocol in protocol_list: #4
        for lambda_param in lambda_list:
            for alpha_param in alpha_list:
                print "python "+ "topic_distribution_real_cost_command_line.py " + datasource +" "+ protocol+" " + str(lambda_param) + " "+str(alpha_param)
                s = s + "\npython "+ "topic_distribution_real_cost_command_line.py " + datasource +" "+ protocol+" " + str(lambda_param) + " "+str(alpha_param) + " &"
                if variation%mod == 0:
                    tmp = '#!/bin/bash\n' \
                          '#SBATCH -J topic_distribution_real_cost_command_line' + str(variation) + '# job name\n' \
                          '#SBATCH -o topic_distribution_real_cost_command_line' + str(
                        variation) + '.o%j       # output and error file name (%j expands to jobID)\n' \
                                     '#SBATCH -n '+str(mod)+'              # total number of mpi tasks requested\n' \
                                     '#SBATCH -p gpu     # queue (partition) -- normal, development, etc.\n' \
                                     '#SBATCH -t 11:59:59        # run time (hh:mm:ss) - 1.0 hours\n' \
                                     '#SBATCH --mail-user=nahidcse05@gmail.com\n' \
                                     '#SBATCH --mail-type=begin  # email me when the job starts\n' \
                                     '#SBATCH --mail-type=end    # email me when the job finishes\n' \
                                     '\nmodule load python'

                    s = tmp + s + "\n\nwait"
                    filname = '/home/nahid/PycharmProjects/parser/scriptdeterministic/activeJobTREC'+ str(variation)
                    text_file = open(filname, "w")
                    text_file.write(s)
                    text_file.close()

                    s=''


                    shellcommand = shellcommand + '\nsbatch activeJobTREC'+ str(variation)
                variation = variation + 1


#shellcommand = shellcommand + '\nsbatch activeJob' + str(variation)
print shellcommand

text_file = open(filename1, "w")
text_file.write(shellcommand)
text_file.close()


print "Number of variations:" + str(variation)