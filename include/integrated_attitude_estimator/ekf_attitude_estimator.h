#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf/tf.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>

#include <fstream>
#include <sstream>
#include <random>

//Custom message
#include "integrated_attitude_estimator/EularAngle.h"


class EKFAttitudeEstimator{
    public:
        EKFAttitudeEstimator();
        ~EKFAttitudeEstimator();
        bool init_process();
        void gt_angle_callback(const integrated_attitude_estimator::EularAngle::ConstPtr& msg);
        void imu_callback(const sensor_msgs::Imu::ConstPtr& msg);
        integrated_attitude_estimator::EularAngle get_correct_angular_velocity(sensor_msgs::Imu imu_data);
        integrated_attitude_estimator::EularAngle get_imu_angle(sensor_msgs::Imu imu_data);
        void dnn_angle_callback(const integrated_attitude_estimator::EularAngle::ConstPtr& msg);
        void prior_process(integrated_attitude_estimator::EularAngle angular_velocity, double sigma);
        void posterior_process(integrated_attitude_estimator::EularAngle angle, double sigma);
        double randomize_value(double mean, double variance);

        void publish_angle();
        void save_csv(ros::Time time);

    private:
        ros::NodeHandle nh;
        ros::NodeHandle private_nh;

        int count = 0;
        ros::Time init_time;
        bool get_bag_data = false;

        ros::Subscriber imu_sub;
        ros::Subscriber angle_sub;
        ros::Subscriber gt_angle_sub;
        ros::Publisher ekf_angle_pub;

        tf2_ros::Buffer tfBuffer;
        tf2_ros::TransformListener tfListener;
        tf2_ros::TransformBroadcaster tfBroadcaster;

        std::string imu_topic_name = "/imu/data";
        std::string dnn_angle_topic_name = "/infer_angle";
        std::string gt_angle_topic_name = "/gt_correct_angle";
        std::string ekf_angle_topic_name = "/ekf_angle";

        const int robot_state_size = 3; // Roll Pitch Yaw

        // DNN params
        integrated_attitude_estimator::EularAngle dnn_angle;
        double sigma_dnn_angle = 0.01;

        // IMU params
        sensor_msgs::Imu imu_data;
        integrated_attitude_estimator::EularAngle angular_velocity;
        integrated_attitude_estimator::EularAngle imu_angle;
        ros::Time imu_prev_time;
        ros::Time imu_current_time;
        double imu_duration = 0.0;
        double sigma_imu_velocity = 0.01;
        double sigma_imu_angle = 0.01;

        //GT Angle Params
        integrated_attitude_estimator::EularAngle gt_angle;

        //Using Data
        bool use_imu_angle = true;
        bool use_imu_angular_velocity = true;
        bool use_dnn_angle = true;

        /*objects*/
		Eigen::VectorXd X;
		Eigen::MatrixXd P;
        integrated_attitude_estimator::EularAngle estimated_angle;

        //Saving Param
        bool save_as_csv = false;
        std::string csv_file_directory = "/home/strage/integrated_attitude_estimator_log/ResNet/";
        std::string csv_file_name = "ekf_attitude_estimator_log.csv";

        // Random
        bool do_randomize_angle = false;
        bool do_randomize_velocity = false;
        double standard_deviation_velocity = 0.03;
        double standard_deviation_angle = 0.1;

};