#!/bin/bash
NET=res101
DATASET=adas
BATCH_SIZE=1
CHECKPOINT=500
DISP_INTERVAL=2000
NUM_WORKERS=1
LR_RATE=0.01
vGPU=3

CHECKEPOCH=1
CHECKSESSION=1
SESSION=1

LR_DECAY_EPOCHS=3

# use anaconda python
echo "######### training ########"
CUDA_VISIBLE_DEVICES=${vGPU} python train_net.py adas_car --dataset ${DATASET} --net ${NET} \
                --session ${SESSION} --checksession ${CHECKSESSION} \
                --checkepoch ${CHECKEPOCH} --checkpoint ${CHECKPOINT} \
                --disp_interval ${DISP_INTERVAL} \
                --lr ${LR_RATE} \
                --bs ${BATCH_SIZE} \
                --lr_decay_step ${LR_DECAY_EPOCHS} \
                --r \
                --cag \
                --cuda \
                --lighthead 
