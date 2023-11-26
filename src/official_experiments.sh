#!/bin/bash
for i in 2732 9845 3264 4859 9225 7891 4373 5874 6744 3468
do
   echo "Instance $i"
   #for j in 1 2
   #do
   #  echo "Run $j"
   python train.py data.params.instance=$i
   #done
done
