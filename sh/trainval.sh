#!/bin/bash
NET=res101
DATASET=adas
BATCH_SIZE=1
CHECKPOINT=1
DISP_INTERVAL=100
NUM_WORKERS=1
LR_RATE=0.001
vGPU=0

CHECKEPOCH=1
CHECKSESSION=1
SESSION=1

# use anaconda python
echo "######### training ########"
CUDA_VISIBLE_DEVICES=${vGPU} python train_net.py car_anchor4 --dataset ${DATASET} --net ${NET} \
                --session ${SESSION} --checksession ${CHECKSESSION} \
                --checkepoch ${CHECKEPOCH} --checkpoint ${CHECKPOINT} \
                --disp_interval ${DISP_INTERVAL} \
                --lr ${LR_RATE} \
                --bs ${BATCH_SIZE} \
                --r \
                --cuda \
                --cag \
                --lighthead 

#echo "#########" ${i} "evaluate ########"
#CUDA_VISIBLE_DEVICES=${vGPU} python test_net.py cag_anchor4 --dataset ${DATASET} --net ${NET}  \
#                --checksession ${SESSION} \
#                --checkepoch ${CHECKEPOCH} --checkpoint ${CHECKPOINT} \
#                --bs ${BATCH_SIZE} \
#                --flip \
#                --cuda \
#                --cag \
#                --lighthead 
