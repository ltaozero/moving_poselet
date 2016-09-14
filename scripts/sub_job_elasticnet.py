import subprocess
import sys

layer_set = [3]
if sys.argv[2] == 'MSR3D':# or sys.argv[2]=='CompAct':
    layer_set = [1]
for num_words in [100]:
    for layer in layer_set:
        for multi_ts in [2]:
            for l2 in [ 1e-3]:
                for l1 in range(11):
                    for rs in range(10):
                        sbatch_command="sbatch -o ~/logfiles/slurm-%j.out /home-3/ltao4@jhu.edu/Code/moving_poselet/scripts/{}.sh {} {} {} {} {} {} {} {}".format(sys.argv[1],sys.argv[2],num_words,layer,sys.argv[3],multi_ts,l2,0.1*l1,rs)
                        print(sbatch_command)
                        exit_status = subprocess.call(sbatch_command, shell=True)
                        if exit_status is 1:  # Check to make sure the job submitted
                            print "Job {0} failed to submit".format(qsub_command)
print "Done submitting jobs!"
