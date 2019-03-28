
#!/bin/bash
# Script to prepare the docker container to be run

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2

base=./logs
port=6006

if [ ! -z $1 ]
then
    python3 -um dopamine.discrete_domains.train --base_dir=$base --debugger --gin_files='dopamine/agents/covariate_shift/configs/covariate_shift.gin' &
    tensorboard --logdir $base --host 0.0.0.0 --port $port --debugger_port 7000 &
else
    python3 -um dopamine.discrete_domains.train --base_dir=$base --gin_files='dopamine/agents/covariate_shift/configs/covariate_shift.gin' &
    tensorboard --logdir $base --host 0.0.0.0 --port $port &
fi