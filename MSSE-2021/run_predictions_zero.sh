#!/usr/bin/env bash
source runner_helper.sh
echo $OPTIONS
preprocesseddir="/mnt/nfs/niddk/ACT_30HZ_CSV_preprocessed_sampled"
predictionsdir="/mnt/nfs/niddk/ACT_PREDICTIONS_SAMPLED_ZERO"
mkdir -p $predictionsdir
source /etc/environment
export EXP_NAME="niddk_pred"
MAKE_CLIENT_LOG_DIR
echo "tail -f ${LOG_DIR}/${EXP_NAME}/"'$WORKER_NAME.log'
$PARALLEL_SSH_ALL "cd /mnt/nfs/DeepPostures/MSSE-2021; $DGL_PY make_predictions.py --predictions-dir "${predictionsdir}" --pre-processed-dir "${preprocesseddir}"/"'${RANK_NUMBER}'" --padding zero --output-label 2>&1 | tee -a ${LOG_DIR}/${EXP_NAME}/"'$WORKER_NAME.log'
