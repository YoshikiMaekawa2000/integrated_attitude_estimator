#include "integrated_attitude_estimator/ekf_attitude_estimator.h"

EKFAttitudeEstimator::EKFAttitudeEstimator():private_nh("~"), tfListener(tfBuffer){
    bool init_result = init_process();
    if(init_result==false){
        ROS_ERROR("Failed to initialize EKF attitude estimator");
        exit(1);
    }
    
    //Subscribers
    imu_sub = nh.subscribe(imu_topic_name, 1, &EKFAttitudeEstimator::imu_callback, this);
    angle_sub = nh.subscribe(dnn_angle_topic_name, 1, &EKFAttitudeEstimator::dnn_angle_callback, this);
    gt_angle_sub = nh.subscribe(gt_angle_topic_name, 1, &EKFAttitudeEstimator::gt_angle_callback, this);

    //Publishers
    ekf_angle_pub = nh.advertise<integrated_attitude_estimator::EularAngle>(ekf_angle_topic_name, 1);

    X = Eigen::VectorXd::Zero(robot_state_size);
	const double initial_sigma = 1.0e-100;
	P = initial_sigma*Eigen::MatrixXd::Identity(robot_state_size, robot_state_size);
}

bool EKFAttitudeEstimator::init_process(){
    bool result = true;

    double dparam = 0.0;
    bool bparam = true;
    std::string sparam;

    if(private_nh.getParam("use_imu_angle", bparam)){
        use_imu_angle = bparam;
    }
    if(private_nh.getParam("use_imu_angular_velocity", bparam)){
        use_imu_angular_velocity = bparam;
    }
    if(private_nh.getParam("use_dnn_angle", bparam)){
        use_dnn_angle = bparam;
    }

    if(private_nh.getParam("sigma_imu_velocity", dparam)){
        sigma_imu_velocity = dparam;
        if(sigma_imu_velocity < 0.0){
            ROS_ERROR("INVALID PARAMETER: sigma_imu_velocity");
            result = false;
        }
    }

    if(private_nh.getParam("sigma_imu_angle", dparam)){
        sigma_imu_angle = dparam;
        if(sigma_imu_angle < 0.0){
            ROS_ERROR("INVALID PARAMETER: sigma_imu_angle");
            result = false;
        }
    }

    if(private_nh.getParam("sigma_dnn_angle", dparam)){
        sigma_dnn_angle = dparam;
        if(sigma_dnn_angle < 0.0){
            ROS_ERROR("INVALID PARAMETER: sigma_dnn_angle");
            result = false;
        }
    }

    if(private_nh.getParam("save_as_csv", bparam)){
        save_as_csv = bparam;
    }

    if(private_nh.getParam("csv_file_directory", sparam)){
        csv_file_directory = sparam;
        if(csv_file_directory.length() <= 0){
            ROS_ERROR("INVALID PARAMETER: csv_file_directory");
            result = false;
        }
    }

    if(private_nh.getParam("csv_file_name", sparam)){
        csv_file_name = sparam;
        if(csv_file_name.length() <= 0){
            ROS_ERROR("INVALID PARAMETER: csv_file_name");
            result = false;
        }
    }

    if(private_nh.getParam("imu_topic_name", sparam)){
        imu_topic_name = sparam;
        if(imu_topic_name.length() <= 0){
            ROS_ERROR("INVALID PARAMETER: imu_topic_name");
            result = false;
        }
    }

    if(private_nh.getParam("dnn_angle_topic_name", sparam)){
        dnn_angle_topic_name = sparam;
        if(dnn_angle_topic_name.length() <= 0){
            ROS_ERROR("INVALID PARAMETER: dnn_angle_topic_name");
            result = false;
        }
    }

    if(private_nh.getParam("gt_angle_topic_name", sparam)){
        gt_angle_topic_name = sparam;
        if(gt_angle_topic_name.length() <= 0){
            ROS_ERROR("INVALID PARAMETER: gt_angle_topic_name");
            result = false;
        }
    }

    if(private_nh.getParam("ekf_angle_topic_name", sparam)){
        ekf_angle_topic_name = sparam;
        if(ekf_angle_topic_name.length() <= 0){
            ROS_ERROR("INVALID PARAMETER: ekf_angle_topic_name");
            result = false;
        }
    }

    if(private_nh.getParam("do_randomize_angle", bparam)){
        do_randomize_angle = bparam;
    }

    if(private_nh.getParam("do_randomize_velocity", bparam)){
        do_randomize_velocity = bparam;
    }

    if(private_nh.getParam("standard_deviation_angle", dparam)){
        standard_deviation_angle = dparam;
        if(standard_deviation_angle < 0.0){
            ROS_ERROR("INVALID PARAMETER: standard_deviation_angle");
            result = false;
        }
    }

    if(private_nh.getParam("standard_deviation_velocity", dparam)){
        standard_deviation_velocity = dparam;
        if(standard_deviation_velocity < 0.0){
            ROS_ERROR("INVALID PARAMETER: standard_deviation_velocity");
            result = false;
        }
    }

    return result;
}

EKFAttitudeEstimator::~EKFAttitudeEstimator(){}

void EKFAttitudeEstimator::gt_angle_callback(const integrated_attitude_estimator::EularAngle::ConstPtr& msg){
    gt_angle = *msg;
}

void EKFAttitudeEstimator::imu_callback(const sensor_msgs::Imu::ConstPtr& msg){
    imu_data = *msg;

    if(imu_data.orientation.x == 0.0 && imu_data.orientation.y == 0.0 && imu_data.orientation.z == 0.0 && imu_data.orientation.w == 0.0){
        //Do nothing
    }
    else{
        if(count == 0){
            init_time = ros::Time::now();
            get_bag_data = true;
            imu_prev_time = init_time;
            imu_current_time = init_time;
            imu_duration = (imu_current_time - imu_prev_time).toSec();
        }
        else{
            imu_prev_time = imu_current_time;
            imu_current_time = imu_data.header.stamp;
            imu_duration = (imu_current_time - imu_prev_time).toSec();
        }
        
        if(imu_duration > 0.5){
            imu_duration = 0.0;
        }
        
        if(use_imu_angular_velocity==true){
            angular_velocity = get_correct_angular_velocity(imu_data);
            prior_process(angular_velocity, sigma_imu_velocity);
        }

        if(use_imu_angle==true){
            imu_angle = get_imu_angle(imu_data);
            posterior_process(imu_angle, sigma_imu_angle);
        }

        if((use_imu_angular_velocity==true || use_imu_angle==true) && get_bag_data==true){
            publish_angle();
        }

        count += 1;
    }
}

double EKFAttitudeEstimator::randomize_value(double mean, double variance){
    double random_value = 0.0;
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());

    std::normal_distribution<float> dist(mean, variance);
    random_value = dist(engine);
    return random_value;
}

integrated_attitude_estimator::EularAngle EKFAttitudeEstimator::get_imu_angle(sensor_msgs::Imu imu_data){
    integrated_attitude_estimator::EularAngle angle;
    double imu_roll, imu_pitch, imu_yaw;
    tf::Quaternion q;
    quaternionMsgToTF(imu_data.orientation, q);
    tf::Matrix3x3(q).getRPY(imu_roll, imu_pitch, imu_yaw);

    angle.roll = imu_pitch * -1.0;
    angle.pitch = (imu_roll + M_PI/2.0) * -1.0;
    angle.yaw = imu_yaw;

    //std::cout << "angle.roll: " << angle.roll << std::endl;

    if(do_randomize_angle == true){
        angle.roll = randomize_value(angle.roll, standard_deviation_angle);
        angle.pitch = randomize_value(angle.pitch, standard_deviation_angle);
        angle.yaw = randomize_value(angle.yaw, standard_deviation_angle);
    }

    //std::cout << "angle.roll: " << angle.roll << std::endl;

    return angle;
}

integrated_attitude_estimator::EularAngle EKFAttitudeEstimator::get_correct_angular_velocity(sensor_msgs::Imu imu_data){
    integrated_attitude_estimator::EularAngle velocity;

    // TODO: Correct angular velocity
    velocity.roll = imu_data.angular_velocity.z;
    velocity.pitch = -1.0 * imu_data.angular_velocity.x;
    velocity.yaw = -1.0 * imu_data.angular_velocity.y;

    //std::cout << "velocity.roll: " << velocity.roll << std::endl;

    if(do_randomize_velocity == true){
        velocity.roll = randomize_value(velocity.roll, standard_deviation_velocity);
        velocity.pitch = randomize_value(velocity.pitch, standard_deviation_velocity);
        velocity.yaw = randomize_value(velocity.yaw, standard_deviation_velocity);
    }

    //std::cout << "velocity.roll: " << velocity.roll << std::endl;

    return velocity;
}

void EKFAttitudeEstimator::dnn_angle_callback(const integrated_attitude_estimator::EularAngle::ConstPtr& msg){
    dnn_angle = *msg;
    dnn_angle.roll = dnn_angle.roll;
    dnn_angle.pitch = dnn_angle.pitch;
    dnn_angle.yaw = dnn_angle.yaw;

    if(use_dnn_angle==true && get_bag_data==true){
        posterior_process(dnn_angle, sigma_dnn_angle);
        publish_angle();
    }
}

void EKFAttitudeEstimator::prior_process(integrated_attitude_estimator::EularAngle angular_velocity, double sigma){
    //printf("prior process\n");
    double roll = X(0);
    double pitch = X(1);
    double yaw = X(2); // 0.0

    double delta_r = angular_velocity.roll * imu_duration;
    double delta_p = angular_velocity.pitch * imu_duration;
    double delta_y = angular_velocity.yaw * imu_duration;
    Eigen::Vector3d Drpy = {delta_r, delta_p, delta_y};

    Eigen::Matrix3d Rot_rpy;	//normal rotation
	Rot_rpy <<	1,	sin(roll)*tan(pitch),	cos(roll)*tan(pitch),
			0,	cos(roll),		-sin(roll),
			0,	sin(roll)/cos(pitch),	cos(roll)/cos(pitch);

    Eigen::VectorXd F(X.size());
    // F <<	roll,
    //         pitch,
    //         yaw;
    
    F(0) = roll;
    F(1) = pitch;
    F(2) = yaw;
    
    F = F + Rot_rpy*Drpy;

    /*jF*/
	Eigen::MatrixXd jF = Eigen::MatrixXd::Zero(X.size(), X.size());
    jF(0, 0) = 1 + (cos(roll)*tan(pitch)*delta_p + sin(roll)*tan(pitch)*delta_y);
    jF(0, 1) = sin(roll)/cos(pitch)/cos(pitch)*delta_p + cos(roll)/cos(pitch)/cos(pitch)*delta_y;
    jF(0, 2) = 0.0;
    jF(1, 0) = -sin(roll)*delta_p + cos(roll)*delta_y;
    jF(1, 1) = 1.0;
    jF(1, 2) = 0.0;
    jF(2, 0) = cos(roll)/cos(pitch)*delta_p - sin(roll)/cos(pitch)*delta_y;
    jF(2, 1) = sin(roll)*sin(pitch)/cos(pitch)/cos(pitch)*delta_p + cos(roll)*sin(pitch)/cos(pitch)/cos(pitch)*delta_y;
    jF(2, 2) = 1.0;

    Eigen::MatrixXd Q = sigma*Eigen::MatrixXd::Identity(X.size(), X.size());

    /*Update*/
	X = F;
    //X(2) = 0.0;
	P = jF*P*jF.transpose() + Q;
}

void EKFAttitudeEstimator::posterior_process(integrated_attitude_estimator::EularAngle angle, double sigma){
    //printf("posterior process\n");

    Eigen::VectorXd Z(3);
	Z <<	angle.roll,
		angle.pitch,
		angle.yaw;

    Eigen::VectorXd Zp = X;
    Eigen::MatrixXd jH = Eigen::MatrixXd::Identity(Z.size(), X.size());
	Eigen::VectorXd Y = Z - Zp;
	Eigen::MatrixXd R = sigma*Eigen::MatrixXd::Identity(Z.size(), Z.size());
	Eigen::MatrixXd S = jH*P*jH.transpose() + R;
	Eigen::MatrixXd K = P*jH.transpose()*S.inverse();
	X = X + K*Y;

    //X(2) = 0.0;

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(X.size(), X.size());
	P = (I - K*jH)*P;
}

void EKFAttitudeEstimator::publish_angle(){
    estimated_angle.header.stamp = ros::Time::now();
    double inference_sec = (estimated_angle.header.stamp - init_time).toSec();

    estimated_angle.roll = X(0);
    estimated_angle.pitch = X(1);
    estimated_angle.yaw = 0.0;


    printf("----Estimated angle [deg]----\n");
    printf("Inference Time     : %6.3f\n", inference_sec);
    printf("\n");
    printf("Estimated Roll     : %6.3f\n", estimated_angle.roll*180.0/M_PI);
    printf("Estimated Pitch    : %6.3f\n", estimated_angle.pitch*180.0/M_PI);
    printf("\n");
    printf("DNN Estimated Roll : %6.3f\n", dnn_angle.roll*180.0/M_PI);
    printf("DNN Estimated Pitch: %6.3f\n", dnn_angle.pitch*180.0/M_PI);
    printf("\n");
    printf("IMU Roll           : %6.3f\n", imu_angle.roll*180.0/M_PI);
    printf("IMU Pitch          : %6.3f\n", imu_angle.pitch*180.0/M_PI);
    printf("\n");
    printf("Ground Truth Roll  : %6.3f\n", gt_angle.roll*180.0/M_PI);
    printf("Ground Truth Pitch : %6.3f\n", gt_angle.pitch*180.0/M_PI);
    printf("\n");
    printf("Diff Roll          : %6.3f\n", (estimated_angle.roll - gt_angle.roll)*180.0/M_PI);
    printf("Diff Pitch         : %6.3f\n", (estimated_angle.pitch - gt_angle.pitch)*180.0/M_PI);
    printf("-----------------------------\n");

    ekf_angle_pub.publish(estimated_angle);
    if(save_as_csv==true){
        save_csv(estimated_angle.header.stamp);
    }
}

void EKFAttitudeEstimator::save_csv(ros::Time time){
    std::string csv_path = csv_file_directory + csv_file_name;
    std::ofstream final_csvfile(csv_path, std::ios::app); //ios::app で追記モードで開ける
    double inference_sec = (time - init_time).toSec();

    std::string str_inference_time = std::to_string(inference_sec);

    std::string str_estimated_roll = std::to_string(estimated_angle.roll);
    std::string str_estimated_pitch = std::to_string(estimated_angle.pitch);
    std::string str_estimated_yaw = std::to_string(estimated_angle.yaw);

    std::string str_gt_roll = std::to_string(gt_angle.roll);
    std::string str_gt_pitch = std::to_string(gt_angle.pitch);
    std::string str_gt_yaw = std::to_string(gt_angle.yaw);

    final_csvfile << str_inference_time << ","
                    << str_estimated_roll << ","
                    << str_estimated_pitch << ","
                    << str_estimated_yaw << ","
                    << str_gt_roll << ","
                    << str_gt_pitch << ","
                    << str_gt_yaw << std::endl;
    
    final_csvfile.close();
}

int main(int argc, char **argv){
    ros::init(argc, argv, "ekf_attitude_estimator");
	std::cout << "EKF Attitude Estimator" << std::endl;
    EKFAttitudeEstimator ekf_attitude_estimator;

    ros::spin();

    return 0;
}