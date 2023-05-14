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

        self.data_list, self.roll_variance, self.pitch_variance = self.data_load()

    def data_load(self):
        csv_path = os.path.join(self.infer_log_top_path, self.infer_log_file_name)
        data_list = []
        roll_val_list = []
        pitch_val_list = []
        
        with open(csv_path) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                data_list.append(row)
                roll_val_list.append(row[0])
                pitch_val_list.append(row[1])

        return data_list, roll_val_list, pitch_val_list

    def spin(self):
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

        acc_roll_variance = acc_roll_variance/size_of_csv
        acc_pitch_variance = acc_pitch_variance/size_of_csv

        print("Average Roll Difference  :" + str(acc_roll_diff) + " [deg]")
        print("Average Pitch Difference :" + str(acc_pitch_diff) + " [deg]")
        print("Average Roll Variance  :" + str(acc_roll_variance))
        print("Average Pitch Variance :" + str(acc_pitch_variance))

        fig = plt.figure()

        c1,c2,c3,c4, c5 = "blue","green","red","black", "orange"    # 各プロットの色

        ax1 = fig.add_subplot(2, 3, 1)
        ax4 = fig.add_subplot(2, 3, 2)
        ax5 = fig.add_subplot(2, 3, 3)
        ax2 = fig.add_subplot(2, 3, 4)
        ax3 = fig.add_subplot(2, 3, 5)

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

        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./verify_infer_vit.py")

    parser.add_argument(
        '--verify_infer', '-vi',
        type=str,
        required=False,
        default='./verify_infer_1.yaml',
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