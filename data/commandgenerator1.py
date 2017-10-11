import sys

dataset = ['TREC8','gov2', 'WT2013', 'WT2014']
protocol_list = ['SAL', 'CAL', 'SPL']
lambda_list = [0.0, 0.25, 0.50, 0.75, 1.0]
alpha_list = [1, 2]
#protocol_list = ['Basic']
#batch_size = [25]
#seed_size = [30, 50, 70]



shellcommand = '#!/bin/sh\n'
s=''
variation = 1
for datasource in dataset: # 1
    for protocol in protocol_list: #4
        for lambda_param in lambda_list:
            for alpha_param in alpha_list:
                print "python "+ "topic_distribution_real_cost_command_line.py " + datasource +" "+ protocol+" " + str(lambda_param) + " "+str(alpha_param)
                s = s + "\npython "+ "topic_distribution_real_cost_command_line.py " + datasource +" "+ protocol+" " + str(lambda_param) + " "+str(alpha_param)
                if variation%1 == 0:
                    tmp = '#!/bin/bash\n' \
                          '#SBATCH -J topic_distribution_real_cost_command_line' + str(variation) + '# job name\n' \
                          '#SBATCH -o topic_distribution_real_cost_command_line' + str(
                        variation) + '.o%j       # output and error file name (%j expands to jobID)\n' \
                                     '#SBATCH -n 2              # total number of mpi tasks requested\n' \
                                     '#SBATCH -p gpu     # queue (partition) -- normal, development, etc.\n' \
                                     '#SBATCH -t 11:59:59        # run time (hh:mm:ss) - 1.0 hours\n' \
                                     '#SBATCH --mail-user=nahidcse05@gmail.com\n' \
                                     '#SBATCH --mail-type=begin  # email me when the job starts\n' \
                                     '#SBATCH --mail-type=end    # email me when the job finishes\n' \
                                     '\nmodule load python'

                    s = tmp + s + "\n"
                    filname = '/home/nahid/PycharmProjects/parser/newscript2/activeJobTREC8'+ str(variation)
                    text_file = open(filname, "w")
                    text_file.write(s)
                    text_file.close()

                    s=''


                    shellcommand = shellcommand + '\nsbatch activeJobTREC8'+ str(variation)
                variation = variation + 1

tmp = '#!/bin/bash\n' \
                          '#SBATCH -J topic_distribution_real_cost_command_line' + str(variation) + '# job name\n' \
                          '#SBATCH -o topic_distribution_real_cost_command_line' + str(
                        variation) + '.o%j       # output and error file name (%j expands to jobID)\n' \
                                     '#SBATCH -n 2              # total number of mpi tasks requested\n' \
                                     '#SBATCH -p gpu     # queue (partition) -- normal, development, etc.\n' \
                                     '#SBATCH -t 11:59:59        # run time (hh:mm:ss) - 1.0 hours\n' \
                                     '#SBATCH --mail-user=nahidcse05@gmail.com\n' \
                                     '#SBATCH --mail-type=begin  # email me when the job starts\n' \
                                     '#SBATCH --mail-type=end    # email me when the job finishes\n' \
                                     '\nmodule load python'

s = tmp + s + "\nwait"
filname = '/home/nahid/PycharmProjects/parser/newscript2/activeJobTREC8' + str(variation)
text_file = open(filname, "w")
text_file.write(s)
text_file.close()
shellcommand = shellcommand + '\nsbatch activeJob' + str(variation)

filename1 = '/home/nahid/PycharmProjects/parser/newscript2/batch_command.sh'
print shellcommand
text_file = open(filename1, "w")
text_file.write(shellcommand)
text_file.close()


print "Number of variations:" + str(variation)