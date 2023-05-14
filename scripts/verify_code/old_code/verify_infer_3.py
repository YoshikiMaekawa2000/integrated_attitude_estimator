import yaml
import csv
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

class VerifyInfer:
    def __init__(self, CFG):
        self.CFG = CFG
        self.infer_log_top_path = CFG["infer_log_top_path"]
        self.infer_log_file_name = CFG["infer_log_file_name"]
        self.infer_log_path = os.path.join(self.infer_log_top_path, self.infer_log_file_name)

        self.start_seq = int(CFG['start_seq'])
        self.end_seq = int(CFG['end_seq'])
        self.step_seq = int(CFG['step_seq'])

        # tmp_result_csv = [roll, pitch, correct_roll, correct_pitch, diff_roll, diff_pitch]

        self.data_list = self.data_load()
        print("Data Length: ", len(self.data_list))

    def data_load(self):
        data_list = []
        
        with open(self.infer_log_path) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                data_list.append(row)

        return data_list

    def spin(self):
        print("Starting spin")

        correct_roll_list = []
        correct_pitch_list = []

        tmp_roll_list = []
        tmp_pitch_list = []

        diff_roll_list = []
        diff_pitch_list = []

        counter = 0
        counter_list = []

        inference_time_list = []

        for row in self.data_list:
            #print(row)
            tmp_roll_list.append(float(row[0]))
            tmp_pitch_list.append(float(row[1]))

            tmp_correct_roll = float(row[2])
            tmp_correct_pitch = float(row[3])

            tmp_diff_roll = float(row[2]) - float(row[0])
            tmp_diff_pitch = float(row[3]) - float(row[1])

            correct_roll_list.append(tmp_correct_roll)
            correct_pitch_list.append(tmp_correct_pitch)

            diff_roll_list.append(tmp_diff_roll)
            diff_pitch_list.append(tmp_diff_pitch)

            inference_time = float(row[6])
            inference_time_list.append(inference_time)

            counter_list.append(counter)
            counter += 1

        gt_roll_list = []
        gt_pitch_list = []

        roll_list = []
        pitch_list = []

        diff_roll = []
        diff_pitch = []

        fig_counter = []

        fig_inference_time = []

        for i in range(self.start_seq, self.end_seq, self.step_seq):
            #print(i)
            roll_list.append(tmp_roll_list[i])
            pitch_list.append(tmp_pitch_list[i])

            gt_roll_list.append(correct_roll_list[i])
            gt_pitch_list.append(correct_pitch_list[i])

            diff_roll.append(diff_roll_list[i])
            diff_pitch.append(diff_pitch_list[i])

            fig_inference_time.append(inference_time_list[i])

            fig_counter.append(i)

        fig = plt.figure()
        ax_roll = fig.add_subplot(2, 1, 1)
        ax_pitch = fig.add_subplot(2, 1, 2)

        c1,c2,c3,c4, c5 = "blue","green","red","black", "orange"

        #ax_roll.plot(fig_counter, diff_roll, c=c1, label="Diff Roll")
        ax_roll.plot(fig_inference_time, gt_roll_list, c=c2, label="GT Roll")
        ax_roll.plot(fig_inference_time, roll_list, c=c3, label="Inferenced Roll")
        
        #ax_pitch.plot(fig_counter, diff_pitch, c=c1, label="Diff Pitch")
        ax_pitch.plot(fig_inference_time, gt_pitch_list, c=c2, label="GT Pitch")
        ax_pitch.plot(fig_inference_time, pitch_list, c=c3, label="Inferenced Pitch")

        ax_roll.set_xlabel("Inference Time [sec]")
        ax_roll.set_ylabel("Roll [deg]")
        ax_roll.set_ylim(-30.0, 30.0)

        ax_pitch.set_xlabel("Inference Time [sec]")
        ax_pitch.set_ylabel("Pitch [deg]")
        ax_pitch.set_ylim(-30.0, 30.0)

        ax_roll.legend()
        ax_pitch.legend()

        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('./verify_infer_vit_2.py')

    parser.add_argument(
        '--verify_infer', '-vi',
        type=str,
        required=False,
        default='./verify_infer_3.yaml',
    )

    FLAGS, unparsed = parser.parse_known_args()

    #Load yaml file
    try:
        print("Opening yaml file %s", FLAGS.verify_infer)
        CFG = yaml.safe_load(open(FLAGS.verify_infer, 'r'))
    except Exception as e:
        print(e)
        print("Error yaml file %s", FLAGS.verify_infer)
        quit()

    verify_infer = VerifyInfer(CFG)
    verify_infer.spin()