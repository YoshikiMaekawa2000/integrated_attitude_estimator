#!/bin/bash
image_name="integrated_attitude_estimator"
tag_name="noetic"
script_dir=$(cd $(dirname $0); pwd)

docker run -it \
    --net="host" \
    --gpus all \
    --privileged \
    --shm-size=480g \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --name="integrated_attitude_estimator" \
    --volume="$script_dir/:/home/ros_catkin_ws/src/$image_name/" \
    --volume="/home/kawai/ssd_dir/:/home/ssd_dir/" \
    --volume="/fs/kawai/:/home/strage/" \
    $image_name:$tag_name \
    bash -c "source /opt/ros/noetic/setup.bash && cd  /home/ros_catkin_ws/ && catkin_make && source devel/setup.bash && bash"
