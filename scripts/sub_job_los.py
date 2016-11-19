import subprocess
import sys

for dataset in ['Suturing', 'KnotTying', 'NeedlePassing']:

    for num_words in [150, 300]:
        for split in ['subject', 'supertrial']:
            for sub in subset:
                layer = 1
                sbatch_command="sbatch -o ~/logfiles/slurm-%j.out /home-3/ltao4@jhu.edu/Code/moving_poselet/scripts/job_los.sh {} {} {} {} {}".format(dataset,num_words,layer,split,sub)
                print(sbatch_command)
                exit_status = subprocess.call(sbatch_command, shell=True)
                if exit_status is 1:  # Check to make sure the job submitted:
                    print "Job {0} failed to submit".format(qsub_command)
print "Done submitting jobs!"
