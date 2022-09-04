#!/usr/bin/env bash
source runner_helper.sh
echo $OPTIONS
preprocesseddir="/mnt/nfs/niddk/ACT_30HZ_CSV_preprocessed"
datadir="/mnt/nfs/niddk/ACT_30HZ_CSV"
eventdir="/mnt/nfs/niddk/ACT_EVENTS"
mkdir -p $preprocesseddir
source /etc/environment
export EXP_NAME="niddk_prep"
MAKE_CLIENT_LOG_DIR
echo "tail -f ${LOG_DIR}/${EXP_NAME}/"'$WORKER_NAME.log'
$PARALLEL_SSH_ALL "cd /mnt/nfs/DeepPostures/MSSE-2021; $DGL_PY pre_process_data.py --gt3x-dir "${datadir}"/"'${RANK_NUMBER}'" --pre-processed-dir "${preprocesseddir}"/"'${RANK_NUMBER}'" --activpal-dir "${eventdir}" --mp 40 --gzipped 2>&1 | tee -a ${LOG_DIR}/${EXP_NAME}/"'$WORKER_NAME.log'


