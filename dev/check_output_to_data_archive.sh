#!/bin/bash
da=/data/archive
if [[ $# = 1 && -d "$da/$1" ]]
then
    for i in $(find $1 -type f | xargs -I{} basename {} | grep -v '.icons' | cut -d '.' -f 1-3 | uniq)
    do
        ls -ltr $da/$1/$(echo $i | cut -d '.' -f 1-2)/$i*
        #find /data/archive/$1/*.{b0,b1,c0,c1,c2,s1,s2} -maxdepth 1 -type f -mtime -1 -ls | wc -l
    done
else echo 'valid site code must be first and only arg'
fi
