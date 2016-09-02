import subprocess
import sys

layer_set = [1,3]
if sys.argv[2] == 'MSR3D':
    layer_set = [1]
for num_words in [5,20,40,60,80,100]:
    for layer in [1]:#layer_set:
        for multi_ts in [1]:
            for l2 in [ 1e-4]:
                for l1 in 0:
                    for rs in range(5,10):
                        sbatch_command="sbatch -o ~/logfiles/slurm-%j.out /home-3/ltao4@jhu.edu/Code/moving_poselet/scripts/{}.sh {} {} {} {} {} {} {} {}".format(sys.argv[1],sys.argv[2],num_words,layer,sys.argv[3],multi_ts,l2,0.1*l1,rs)
                        print(sbatch_command)
                        exit_status = subprocess.call(sbatch_command, shell=True)
                        if exit_status is 1:  # Check to make sure the job submitted
                            print "Job {0} failed to submit".format(qsub_command)
print "Done submitting jobs!"
