import training
import random
import utils
import scale
import numpy as np
import sys
import math
from os import listdir
from os.path import isfile, join


class basecallTraining():
    train_dir = ""
    test_dir = ""
    current_file = ""
    current_file_length = 0
    current_lines = None
    current_input = None
    current_index_input = 0
    current_index_reference = 0
    current_bases_per_event_ratio = 1
    current_scale = 0
    current_scale_sd = 0
    current_shift = 0
    def __init__(self, train_dir, test_dir):
        if (train_dir.endswith("/")):
            basecallTraining.train_dir = train_dir
        else:
            basecallTraining.train_dir = train_dir+"/"
        if (test_dir.endswith("/")):
            basecallTraining.test_dir = test_dir
        else:
            basecallTraining.test_dir = test_dir+"/"

    @staticmethod
    def resetValues():
        basecallTraining.current_file = ""
        basecallTraining.current_file_length = 0
        basecallTraining.current_lines = None
        basecallTraining.current_input = None
        basecallTraining.current_index_input = 0
        basecallTraining.current_index_reference = utils.extend_size
        basecallTraining.current_bases_per_event_ratio = 1
        basecallTraining.current_scale = 0
        basecallTraining.current_scale_sd = 0
        basecallTraining.current_shift = 0

    @staticmethod
    def getFiles(path):
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        return onlyfiles

    @staticmethod
    def get_next_input():
        '''
            It reads an entire event file and its reference. When the file reaches the end it picks another one at random
            and it does the same reading process. The function returns the next event input of the current file,
            in a sliding window.
        '''
        if (basecallTraining.current_file == "" or (basecallTraining.current_file_length - basecallTraining.current_index_input) <= utils.batch_size):
            basecallTraining.resetValues()
            train_files = basecallTraining.getFiles(basecallTraining.train_dir)
            num = random.randrange(0,len(train_files))
            target_file = train_files[num]
            with open(basecallTraining.train_dir+target_file) as f:
                lines = f.readlines()
            basecallTraining.current_lines = lines
            basecallTraining.current_file = target_file
            basecallTraining.current_file_length = len(lines)
            fast5file = basecallTraining.current_lines[0].split()[4]
            basecallTraining.current_scale, basecallTraining.current_scale_sd, basecallTraining.current_shift = scale.get_scale_and_shift(fast5file, 1, "template")
            input = []
            for x in range(1,len(basecallTraining.current_lines)):
                input.append(basecallTraining.current_lines[x].split())
                mean = input[len(input)-1][0]
                stdv = input[len(input)-1][1]
                mean, stdv = scale.scale(mean,stdv,basecallTraining.current_scale,basecallTraining.current_scale_sd,basecallTraining.current_shift)
                input[len(input) - 1][0] = mean
                input[len(input) - 1][1] = stdv
            input.pop(0)
            basecallTraining.current_input = input
            basecallTraining.current_bases_per_event_ratio = float(float(len(basecallTraining.current_lines[0].split()[3]))/float(len(input)))
        ret = np.asarray(basecallTraining.current_input)[basecallTraining.current_index_input:(basecallTraining.current_index_input + utils.batch_size)]
        basecallTraining.current_index_input += 2
        return ret

    @staticmethod
    def get_next_reference_sequence():
        '''
            This calculates where the the next reference sequence is and returns it. It works like a sliding window.
        '''
        fromX = int(max(basecallTraining.current_index_reference - utils.extend_size, 0))
        toX = int(min(basecallTraining.current_index_reference + utils.extend_size + 4*max(utils.batch_size*utils.elements_size, utils.batch_size*basecallTraining.current_bases_per_event_ratio),len(basecallTraining.current_input)))

        if (toX - fromX % 2 != 1):
            toX += 1

        target_sequence = basecallTraining.current_lines[0].split()[3][fromX:toX]
        basecallTraining.current_index_reference += math.ceil(basecallTraining.current_bases_per_event_ratio)
        return target_sequence

    def get_test_input(self):
        train_files = basecallTraining.getFiles(basecallTraining.test_dir)
        num = random.randrange(0,len(train_files))
        target_file = train_files[num]
        with open(basecallTraining.test_dir+target_file) as f:
            lines = f.readlines()
        lines.pop(0) #take out first line
        input = []
        for l in lines:
            input.append(l.split())
        return np.asarray(input)

    def get_reference_input(self):
        train_files = basecallTraining.getFiles(basecallTraining.test_dir)
        num = random.randrange(0,len(train_files))
        target_file = train_files[num]
        with open(basecallTraining.test_dir+target_file) as f:
            lines = f.readlines()
        target_sequence = lines[0].split()[3]
        return target_sequence

    def startTraining(self):
        training.training(self.get_next_input,self.get_next_reference_sequence)
        return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Usage: python main DIRECTORY_FOR_TRAINING DIRECTORY_FOR_TEST"
    else:
        train_dir = sys.argv[1]
        test_dir = sys.argv[2]
        basecallTrainer = basecallTraining(train_dir, test_dir)
        basecallTrainer.startTraining()