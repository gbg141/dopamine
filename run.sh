
#!/bin/bash
# Script to prepare the docker container to be run

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=7

cd /home/dopamine/
base=./logs

if [ ! -z $1 ]
then
    python3 -um dopamine.discrete_domains.train --base_dir=$base --debugger --gin_files='dopamine/agents/covariate_shift/configs/covariate_shift.gin' &
    tensorboard --logdir /home/dopamine/logs --host 0.0.0.0 --debugger_port 7000 &
else
    python3 -um dopamine.discrete_domains.train --base_dir=$base --gin_files='dopamine/agents/covariate_shift/configs/covariate_shift.gin' &
    tensorboard --logdir /home/dopamine/logs --host 0.0.0.0 &
fi