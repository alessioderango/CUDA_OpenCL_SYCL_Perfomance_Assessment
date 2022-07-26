1) Rename the setenv_YOURHOSTNAME.txt file by changing the YOURHOSTNAME word with the output of the "echo $HOSTNAME" command

2) Edit the renamed file by updating the environmental variables according to your system configuration

3) Run the run_tests.sh script by the "sh run_tests.sh -gpu GPUNUMBER" command by replacing GPUNUMBER with the ID of the output od nvidia-smi of the A100 GPU

4) Run the run_tests_stresstest.sh script by the "sh run_tests_stresstest.sh -gpu GPUNUMBER" command by replacing GPUNUMBER with the ID of the output of nvidia-smi of the A100 GPU

5) Run the run_profile.sh script by the "sh run_profile.sh -gpu GPUNUMBER -block_size BLOCKSIZE" command by replacing GPUNUMBER with the ID of the output od nvidia-smi of the A100 GPU and BLOCKSIZE with the size of the block, we used a default 16 blocksize for our experiments (please use the same value)

6) Run the run_profile_with_halo.sh script by the "sh run_profile_with_halo.sh -gpu GPUNUMBER -block_size BLOCKSIZE" command by replacing GPUNUMBER with the ID of the output of nvidia-smi of the A100 GPU and BLOCKSIZE with the size of the block, we used a default 16 blocksize for our experiments (please use the same value)

7) Run the run_profile_without_halo.sh script by the "sh run_profile_without_halo.sh -gpu GPUNUMBER -block_size BLOCKSIZE" command by replacing GPUNUMBER with the ID of the output od nvidia-smi of the A100 GPU and BLOCKSIZE with the size of the block, we used a default 16 blocksize for our experiments (please use the same value)

8) Run the run_profile_stresstest.sh script by the "sh run_profile_stresstest.sh -gpu GPUNUMBER -block_size BLOCKSIZE" command by replacing GPUNUMBER with the ID of the output od nvidia-smi of the A100 GPU and BLOCKSIZE with the size of the block, we used a default 16 blocksize for our experiments (please use the same value)

9) Run the run_profile_stresstest_with_halo.sh script by the "sh run_profile_stresstest_with_halo.sh -gpu GPUNUMBER -block_size BLOCKSIZE" command by replacing GPUNUMBER with the ID of the output od nvidia-smi of the A100 GPU and BLOCKSIZE with the size of the block, we used a default 16 blocksize for our experiments (please use the same value)

10) Run the run_profile_stresstest_without_halo.sh script by the "sh run_profile_stresstest_without_halo.sh -gpu GPUNUMBER -block_size BLOCKSIZE" command by replacing GPUNUMBER with the ID of the output od nvidia-smi of the A100 GPU and BLOCKSIZE with the size of the block, we used a default 16 blocksize for our experiments (please use the same value)

At the end of each run, you will find different folders named "LogProfile_*" and "LogOutput_*".
These folders contain the results of the experiments, and we need to get them back.
Please create an archive with all the Log folders and send it to us.
