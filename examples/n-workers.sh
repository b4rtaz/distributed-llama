#!/bin/bash

# This script starts N workers from a single command. Mainly useful for testing and debugging.
# Usage:
#
# W=7 T=2 bash n-workers.sh start
# W=7 bash n-workers.sh stop
#
# Env vars:
# W - n workers
# T - n threads per worker

cd "$(dirname "$0")"

if [ -z "$W" ]; then
  W=3
fi
if [ -z "$T" ]; then
  T=1
fi

if [ "$1" == "start" ]; then
    for (( w = 0; w < $W ; w += 1 ));
    do
        PORT=$(expr 9999 - $w)
        PROC_ID=$(lsof -ti:$PORT)
        if [ -n "$PROC_ID" ]; then
            kill -9 $PROC_ID
            echo "Killed process $PROC_ID"
        fi

        mkdir -p dllama_worker_$w # macOs does not support -Logfile argument, so we place logs inside different directories
        cd dllama_worker_$w
        screen -d -L -S dllama_worker_$w -m ../../dllama worker --port $PORT --nthreads $T
        cd ..
        echo "Started worker $w on port $PORT"
    done

    sleep 2
elif [ "$1" == "stop" ]; then
    for (( w = 0; w < $W ; w += 1 ));
    do
        screen -S dllama_worker_$w -X quit
    done

    echo "Stopped $W workers"
else
    echo "Usage: $0 [start|stop]"
fi

echo "> screen -ls"
screen -ls
