
#!/bin/bash
# Script to prepare the docker container to be run

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=9

base=./logs/logexp055
host=0.0.0.0
port=1001
gin_files='dopamine/agents/covariate_shift/configs/covariate_shift.gin'

if [ ! -z $1 ]
then
    python3 -um dopamine.discrete_domains.train --base_dir=$base --debugger --gin_files=$gin_files &
    tensorboard --logdir $base --host $host --port $port --debugger_port 7000 &
else
    python3 -um dopamine.discrete_domains.train --base_dir=$base --gin_files=$gin_files &
    tensorboard --logdir $base --host $host --port $port &
fi