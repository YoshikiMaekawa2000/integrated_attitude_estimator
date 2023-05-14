#!/bin/bash

image_name='integrated_attitude_estimator'
image_tag='noetic'

docker build -t $image_name:$image_tag .