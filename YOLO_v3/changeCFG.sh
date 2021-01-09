#!/bin/bash
sed -i 's/batch=1/batch=64/' yolov3_testing.cfg
sed -i 's/subdivisions=1/subdivisions=16/' yolov3_testing.cfg
sed -i 's/max_batches = 500200/max_batches = 4000/' yolov3_testing.cfg
# We change the number of classes to 43
sed -i '610 s@classes=80@classes=43@' yolov3_testing.cfg
sed -i '696 s@classes=80@classes=43@' yolov3_testing.cfg
sed -i '783 s@classes=80@classes=43@' yolov3_testing.cfg
# We change filters to (classes+5)*3 = (43+5)*3 = 144
sed -i '603 s@filters=255@filters=144@' yolov3_testing.cfg
sed -i '689 s@filters=255@filters=144@' yolov3_testing.cfg
sed -i '776 s@filters=255@filters=144@' yolov3_testing.cfg
