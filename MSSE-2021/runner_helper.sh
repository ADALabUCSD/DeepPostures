#!/usr/bin/env bash
# Copyright 2020 Yuhao Zhang and Arun Kumar. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
set -e
TIMESTAMP=${1:-`date "+%Y_%m_%d_%H_%M_%S"`}
EPOCHS=${2:-200}
SIZE=${3:-8}
OPTIONS=${4:-""}
SECONDS=0
HOSTS="/local/host_list"
HOSTS_ALL="/local/all_host_list"
master_ip="10.10.1.1"
LOG_DIR="/mnt/nfs/logs/run_logs/$TIMESTAMP"
MODEL_DIR="/mnt/nfs/models/$TIMESTAMP"
export START=0
export FINISH=0

mkdir -p $MODEL_DIR
mkdir -p $LOG_DIR
PARALLEL_SSH="parallel-ssh -i -h $HOSTS -t 0 -O StrictHostKeyChecking=no"
PARALLEL_SSH_ALL="parallel-ssh -h $HOSTS_ALL -t 0 -O StrictHostKeyChecking=no -o $LOG_DIR -e $LOG_DIR"

echo $SUB_LOG_DIR
echo $MODEL_DIR
echo "Clearing master sys cache"
free && sync && echo 3 |sudo tee /proc/sys/vm/drop_caches && free
echo "Clearing workers sys cache"
$PARALLEL_SSH 'free && sync && echo 3 |sudo tee /proc/sys/vm/drop_caches && free'

CLEAN_UP_DGL (){
    $PARALLEL_SSH 'pkill -f torch.distributed; pkill -f run_dgl.py'
    sleep 2
}

SHUTDOWN_SPARK (){
    echo "Shutting down master"
    bash /usr/local/spark/sbin/stop-master.sh && sleep 3
    echo "Shutting down workers"
    $PARALLEL_SSH "bash /usr/local/spark/sbin/stop-slave.sh"
}
START_SPARK (){
    echo "Starting master"
    bash /usr/local/spark/sbin/start-master.sh && sleep 3
    echo "Starting workers"
    $PARALLEL_SSH "bash /usr/local/spark/sbin/start-slave.sh $master_ip:7077" && sleep 6
}
RESTART_SPARK () {
    # echo "Restarting master"
    # bash /usr/local/spark/sbin/stop-master.sh && sleep 3
    # bash /usr/local/spark/sbin/start-master.sh && sleep 6
    # echo "Restarting workers"
    # $PARALLEL_SSH "bash /usr/local/spark/sbin/stop-slave.sh && bash /usr/local/spark/sbin/start-slave.sh $master_ip:7077" && sleep 15
    $SPARK_HOME/sbin/stop-all.sh
    $SPARK_HOME/sbin/start-all.sh
}
RESTART_CEREBRO () {
    echo "Restarting cerebro"
    $PARALLEL_SSH "bash /local/cerebro-greenplum/cerebro_gpdb/run_cerebro_worker.sh >/dev/null 2>&1 &" && sleep 10
}


PRINT_START () {
   echo "Running $EXP_NAME ..."
   echo "$EXP_NAME, Start time `date "+%Y-%m-%d %H:%M:%S"`"| tee -a $LOG_DIR/global.log
   export START=$SECONDS
} 
PRINT_END () {
   echo "$EXP_NAME, End time `date "+%Y-%m-%d %H:%M:%S"`"| tee -a $LOG_DIR/global.log
   export FINISH=$SECONDS
   elapsed=$(($FINISH - $START))
   echo "$EXP_NAME, EXP TIME $elapsed"| tee -a $LOG_DIR/global.log
   echo "$EXP_NAME, TOTAL EXECUTION (INCREMENTAL) $SECONDS"| tee -a $LOG_DIR/global.log
   sleep 30
}

MAKE_CLIENT_LOG_DIR () {
    SUB_LOG_DIR=$LOG_DIR/$EXP_NAME
    mkdir -p $SUB_LOG_DIR
}

RUN_EXP () {
    
   MAKE_CLIENT_LOG_DIR
   echo "$1" | tee -a ${SUB_LOG_DIR}/client.log
   PRINT_START
   set +e
   eval "$1 2>&1 | tee -a ${SUB_LOG_DIR}/client.log" && touch ${SUB_LOG_DIR}/__SUCCESS__
   set -e
   PRINT_END
}

SUCCESS_TAG () {
   touch $LOG_DIR/__SUCCESS__
}

