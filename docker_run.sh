#!/bin/bash
image_name="integrated_attitude_estimator"
tag_name="noetic"
script_dir=$(cd $(dirname $0); pwd)

# xhost +
docker run -it \
    --net=host \
    --gpus all \
    --privileged \
    --shm-size=16g \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --device=/dev/dri:/dev/dri \
    --name="integrated_attitude_estimator" \
    --volume="$script_dir/:/home/ros_catkin_ws/src/$image_name/" \
    --volume="/home/amsl/bagfiles/:/home/bagfiles/" \
    --volume="/media/amsl/96fde31e-3b9b-4160-8d8a-a4b913579ca21/:/home/strage/" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    $image_name:$tag_name \
    bash -c "source /opt/ros/noetic/setup.bash && cd  /home/ros_catkin_ws/ && catkin_make && source devel/setup.bash && bash"