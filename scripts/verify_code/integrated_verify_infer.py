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
        self.infer_log_base_name = CFG["infer_log_base_name"]
        self.infer_log_base_path = os.path.join(self.infer_log_top_path, self.infer_log_base_name)
        self.infer_log_csv_path = self.infer_log_base_path + ".csv"

        self.start_seq = int(CFG['start_seq'])
        # self.end_seq = int(CFG['end_seq'])
        # if self.end_seq < 0:
        #     print("Invalid end seq")
        #     exit()
        self.step_seq = int(CFG['step_seq'])

        self.fig_1_name = CFG["fig_1_name"]
        self.fig_2_name = CFG["fig_2_name"]

        self.end_time = int(CFG['end_time'])

        self.mae_roll = 0.0
        self.mae_pitch = 0.0

        # tmp_result_csv = [roll, pitch, correct_roll, correct_pitch, diff_roll, diff_pitch]

        self.data_list = self.data_load()
        print("Data Length: ", len(self.data_list))
        self.end_seq = len(self.data_list)-1

    def data_load(self):
        data_list = []
        
        with open(self.infer_log_csv_path) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                data_list.append(row)

        return data_list

    def save_scatter_graph(self):
        acc_roll_diff = 0.0
        acc_pitch_diff = 0.0
        acc_roll_variance = 0.0
        acc_pitch_variance = 0.0
        size_of_csv = float(len(self.data_list))

        roll_diff_list = []
        pitch_diff_list = []

        correct_roll_list = []
        correct_pitch_list = []

        roll_val_list = []
        pitch_val_list = []

        for row in self.data_list:
            tmp_roll_diff = abs(float(row[0]) - float(row[2]))
            tmp_pitch_diff = abs(float(row[1]) - float(row[3]))

            tmp_correct_roll = float(row[2])
            tmp_correct_pitch = float(row[3])

            roll_val = float(row[0])
            pitch_val = float(row[1])

            roll_diff_list.append(tmp_roll_diff)
            pitch_diff_list.append(tmp_pitch_diff)

            correct_roll_list.append(tmp_correct_roll)
            correct_pitch_list.append(tmp_correct_pitch)

            acc_roll_diff += tmp_roll_diff
            acc_pitch_diff += tmp_pitch_diff

            roll_val_list.append(roll_val)
            pitch_val_list.append(pitch_val)

        
        acc_roll_diff = acc_roll_diff/size_of_csv
        acc_pitch_diff = acc_pitch_diff/size_of_csv

        self.mae_roll = acc_roll_diff
        self.mae_pitch = acc_pitch_diff

        acc_roll_variance = acc_roll_variance/size_of_csv
        acc_pitch_variance = acc_pitch_variance/size_of_csv

        print("Average Roll Difference  :" + str(acc_roll_diff) + " [deg]")
        print("Average Pitch Difference :" + str(acc_pitch_diff) + " [deg]")
        # print("Average Roll Variance  :" + str(acc_roll_variance))
        # print("Average Pitch Variance :" + str(acc_pitch_variance))

        fig = plt.figure(figsize=(19.20, 10.80))
        c1,c2,c3,c4, c5 = "blue","green","red","black", "orange"    # 各プロットの色

        ax1 = fig.add_subplot(2, 3, 1)
        ax4 = fig.add_subplot(2, 3, 2)
        ax5 = fig.add_subplot(2, 3, 3)
        ax2 = fig.add_subplot(2, 3, 4)
        ax3 = fig.add_subplot(2, 3, 5)

        plt.suptitle("MAE of Roll: {:.3f}".format(self.mae_roll) + " [deg], MAE of Pitch: {:.3f}".format(self.mae_pitch) + " [deg]", fontsize=30)

        ax1.scatter(roll_diff_list, pitch_diff_list, color=c1, s=0.1)
        ax2.scatter(correct_roll_list, roll_diff_list, color=c2, s=0.1)
        ax3.scatter(correct_pitch_list, pitch_diff_list, color=c3, s=0.1)
        ax4.scatter(roll_val_list, roll_diff_list, color=c4, s=0.1)
        ax5.scatter(pitch_val_list, pitch_diff_list, color=c5, s=0.1)

        ax1.set_xlabel("Roll Diff")
        ax1.set_ylabel("Pitch Diff")
        ax1.set_xlim(0, 31)
        ax1.set_ylim(0, 31)

        ax2.set_xlabel("Roll GT")
        ax2.set_ylabel("Roll Diff")
        ax2.set_xlim(-31, 31)
        ax2.set_ylim(0, 31)

        ax3.set_xlabel("Pitch GT")
        ax3.set_ylabel("Pitch Diff")
        ax3.set_xlim(-31, 31)
        ax3.set_ylim(0, 31)

        ax4.set_xlabel("Roll Val")
        ax4.set_ylabel("Roll Diff")
        ax4.set_xlim(-31, 31)
        ax4.set_ylim(0, 31)

        ax5.set_xlabel("Pitch Val")
        ax5.set_ylabel("Pitch Diff")
        ax5.set_xlim(-31, 31)
        ax5.set_ylim(0, 31)

        fig_1_save_path = self.infer_log_base_path + self.fig_1_name + ".png"
        plt.savefig(fig_1_save_path)

        #plt.show()

    def save_seq_graph(self):
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

        fig_1 = plt.figure(figsize=(19.20, 10.80))
        ax_roll = fig_1.add_subplot(2, 1, 1)
        ax_pitch = fig_1.add_subplot(2, 1, 2)

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

        print("Last Inference Time: ", inference_time_list[-1], "[sec]")

        ax_roll.set_xlim(0, self.end_time)

        ax_roll.legend()
        ax_pitch.legend()

        plt.suptitle("MAE of Roll: {:.3f}".format(self.mae_roll) + " [deg], MAE of Pitch: {:.3f}".format(self.mae_pitch) + " [deg]", fontsize=30)

        fig_2_save_path = self.infer_log_base_path + self.fig_2_name + ".png"
        plt.savefig(fig_2_save_path)

        # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('./verify_infer_vit_2.py')

    parser.add_argument(
        '--verify_infer', '-vi',
        type=str,
        required=False,
        default='./integrated_verify_infer.yaml',
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
    verify_infer.save_scatter_graph()
    verify_infer.save_seq_graph()