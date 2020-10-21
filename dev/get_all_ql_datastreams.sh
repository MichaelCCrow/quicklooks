#!/bin/bash

da=/data/archive
if [ $# = 1 ]
then
    if [ -d "$da/$1" ]
    then
        ls "$da/$1" | grep -vE '.a1|.a0|.00' > "/home/quicklooks/sites/$1-all.txt"
    else >&2 echo "bad site code"
    fi
else >&2 echo "must provide site code as first arg"
fi
#ls /data/archive/sgp/ | grep -vE '.a1|.a0|.00'
