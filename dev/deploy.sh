#!/bin/bash
#dest=act_plotting_main.py
#[ $# = 1 ] && dest=$1
#scp act_plotting_main.py armprod-plot:$dest
ls
if [ -f act_plotting_main.py ]; then echo "its there"; else echo "nope"; exit; fi
scp act_plotting_main.py armprod-plot:

