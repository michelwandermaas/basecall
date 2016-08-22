import tensorflow as tf
import numpy as np
import random
import sys
import math

#variables

batch_size = 100
extend_size = 6
sequence_size = batch_size + (extend_size*2)
feature_size = 3
elements_size = 2
dictionary = ["A","C","G","T","-"]
alphabet_dict = {"A":0, "C":1, "G":2, "T":3, "-":4}
dictionary_no_gap = ["A","C","G","T"]
dictionary_size = dictionary.__len__()
lstm_hidden_size = feature_size/2
number_of_layers = 3
learning_rate = 0.01
training_steps = 10000
type = tf.float32

numAcc = 100 #every x iterations for accuracy calculation



#aligment variables
'''
    Since we do not care about evolutionary similarity, we will not use a scoring matrix.
    Instead, we will use the probabilities to weight the scores.
    identical_char_score *= average_output_probability(without gaps)
    non_identical_char_score *= 1-average_output_probability(without gaps)
    gap_opening_score *= average_gap_probability
'''
identical_char_score = 2
non_identical_char_score = 1
gap_opening_score = -4
gap_identical_score = 0

#functions

# it`s impossible to restore the exact same value...

# I can simply transform the one-hot-encoding probabilities to an array of W`s size, and it would be made of 0`s and 1`s
# that would make one weight go up and the others go down

'''
    This function will take size, calculate a normal distribution based on mean and stdv. Then it output the f(x), for
    size x`s between 0 and 1, in order, and equally spaced.
'''

mean_gaussian = 0.5
stdv_gaussian = 0.2
min_gaussian = 0.1

def gaussian(x, sigma, mu):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))

def get_gaussian_distr(size):
    x = np.zeros(size)
    step = float(1.0/float(size))
    for i in range(size):
        x[i] = gaussian(i*step, stdv_gaussian, mean_gaussian)
        if x[i] < min_gaussian:
            x[i] = min_gaussian
    return x

def get_sequences_identity(seq1,seq2):
    equal = 0.0
    for i in range(len(seq1)):
        if seq1[i] == seq2[i]:
            equal += 1
    return equal/float(len(seq1))


def get_prob_from_output_and_W(output, W): # not being used as of now
    '''
    :param output: output from the lstm layers, dimensions = [batch_size][feature_size]
    :param W: output weights, dimensions =  [element_size][dictionary_size][feature_size]
    :return: prob: dimensions = [batch_size][element_size][dictionary_size]
    '''
    '''

    '''
    prob = np.zeros((batch_size, elements_size, dictionary_size), dtype=np.float32)
    for index in range(batch_size):
        for element in range(elements_size):
            for word in range(dictionary_size):
                y = tf.mul(output[index, :], W[element, word, :])
                prob[index, element, word] = y
    return prob


def get_sequence_from_prob(probabilities):
    '''
    :param probabilities: dimensions = [batch_size][element_size][dictionary_size]
    :return: sequences: dimensions = [batch_size]
            output_avg_prob: average probability for the output base
            non_output_avg_prob: average probability for the other bases
    '''
    sequences = ""
    total_out_prob = 0
    for x in probabilities:
        for y in x:
            max = 0
            best_word = -1
            for z in range(dictionary_size):
                if y[z] > max:
                    best_word = z
                    max = y[z]
            sequences += dictionary[best_word]
            total_out_prob += max
    output_avg_prob = total_out_prob/(batch_size*elements_size*dictionary_size)
    return sequences, output_avg_prob, (1-output_avg_prob)

def score_match(A,B, match_score, mismatch_score, gap_score, gap_identical_score):
    if (A == "-" and B == "-"):
        return gap_identical_score
    elif (A==B):
        return match_score
    elif(A=='-' or B =='-'):
        return gap_score
    else:
        return mismatch_score

def get_prob_dict(prob):
    return_list = []
    for x in prob:
        dict = {}
        for y in range(len(x)):
            dict[dictionary[y]] = x[y]
        return_list.append(dict)
    return return_list

def probability_alignment(probabilities, reference_sequence):
    '''
    :param probabilities: list of a dictionary {"Base":log probabilities}
    :param reference_sequence: a string
    :return: aligned reference
    '''
    gauss = np.log(get_gaussian_distr(len(probabilities)))

    m, n = len(probabilities), len(reference_sequence)  # length of two sequences

    # Generate DP table and traceback path pointer matrix
    score = (np.zeros((m + 1, n + 1))).tolist()  # the DP table
    pointer = (np.zeros((m + 1, n + 1))).tolist()  # to store the traceback path

    max_score = 0  # initial maximum score in DP table
    # Calculate DP table and mark pointers
    max_i, max_j = 0, 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            score_diagonal = score[i - 1][j - 1] + math.log(probabilities[i - 1][reference_sequence[j - 1]]) + gauss[i-1]
            #score_left = score[i][j - 1] + math.log(probabilities[i - 1]["-"])
            score_left = score[i-1][j] + math.log(probabilities[i - 1]["-"]) + gauss[i-1]
            score[i][j] = min(score_left, score_diagonal)
            if score[i][j] == score_left:
                pointer[i][j] = 1  # 1 means trace up
            if score[i][j] == score_diagonal:
                pointer[i][j] = 3  # 3 means trace diagonal
            if score[i][j] <= max_score:
                max_i = i
                max_j = j
                max_score = score[i][j]

    align1, align2 = '', ''  # initial sequences

    i, j = max_i, max_j  # indices of path starting point

    #print pointer
    #sys.exit()

    # traceback, follow pointers
    while i > 0 and j > 0:
        if pointer[i][j] == 3:
            align2 += reference_sequence[j - 1]
            i -= 1
            j -= 1
        elif pointer[i][j] == 1:
            align2 += '-'
            i -= 1

    return align2[::-1]

def needle(seq1, seq2, match_score, mismatch_score, gap_score, gap_identical_score): #seq 1 equals output and seq2 equals reference, returns seq2 aligned
    '''
    :param seq1: output
    :param seq2: reference
    :param match_score:
    :param mismatch_score:
    :return: reference aligned
    '''
    '''
        This is a global alignment with no vertical transition. Also, there`s a weight to each score addition equivalent
        to the position of that alignment in the reference sequence. This weight is taken from a normal distribution,
        with a minimum weight, in a way that the position closest to the center of the sequence gets higher weights.
        The objective is to allow for the output to align itself to the any part of the sequence (even the overextensions
        on each side) but give it a push to align with the center.
    '''
    m, n = len(seq1), len(seq2)  # length of two sequences
    # Generate DP table and traceback path pointer matrix
    score = (np.zeros((m + 1, n + 1))).tolist()  # the DP table
    # Calculate DP table
    for i in range(0, m + 1):
        score[i][0] = gap_opening_score * i
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = score[i - 1][j - 1] + score_match(seq1[i - 1], seq2[j - 1], match_score, mismatch_score, gap_score, gap_identical_score)
            delete = score[i - 1][j] + gap_opening_score
            score[i][j] = max(match, delete)
    # Traceback and compute the alignment
    align1, align2 = '', ''
    i, j = m, n  # start from the bottom right cell
    while i > 0 and j > 0:  # end toching the top or the left edge
        score_current = score[i][j]
        score_diagonal = score[i - 1][j - 1]
        score_left = score[i - 1][j]
        if score_current == score_diagonal + score_match(seq1[i - 1], seq2[j - 1], match_score, mismatch_score, gap_score, gap_identical_score):
            align1 += seq1[i - 1]
            align2 += seq2[j - 1]
            i -= 1
            j -= 1
        elif score_current == score_left + gap_opening_score:
            align1 += seq1[i - 1]
            align2 += '-'
            i -= 1
        else:
            print "Error.\n"
            sys.exit(-1)
    # Finish tracing up to the top left cell
    while i > 0:
        align1 += seq1[i - 1]
        align2 += '-'
        i -= 1
    return align2[::-1]

def align(output_sequence, target_sequence, output_avg_prob, non_output_avg_prob): # TODO: see if I am getting the right alignment
    # this function returns a listo of alignments with the highest score. I will choose only the first, and I only care about the sequence itself, not the score
    ret = needle(output_sequence,target_sequence,identical_char_score*output_avg_prob,non_identical_char_score*non_output_avg_prob, gap_opening_score*non_output_avg_prob, gap_identical_score)
    return ret

def get_one_hot_encoding_prob(aligned_reference):
    '''
    :param aligned_reference: dimensions = [batch_size*element_size]
    :return: probabilities: dimensions = [batch_size][element_size][dictionary_size]
    '''
    probabilities = []
    count = 0
    probabilities.append([])
    for x in aligned_reference:
        if (count == elements_size):
            count = 0
            probabilities.append([])
        count += 1
        x_index = probabilities.__len__()-1
        probabilities[x_index].append([])
        for y in range(dictionary_size):
            if (dictionary[y]==x):
                probabilities[x_index][probabilities[x_index].__len__()-1].append(1.0)
            else:
                probabilities[x_index][probabilities[x_index].__len__()-1].append(0.0)
    return np.asarray(probabilities)

def get_argmax_encoding_prob(seq):
    sequences = []
    idx = 0
    for x in seq:
        if(idx == 0):
            sequences.append([])
            idx += 1
            sequences[len(sequences)-1].append(float(alphabet_dict[x]))
        else:
            idx = 0
            sequences[len(sequences)-1].append(float(alphabet_dict[x]))

    return np.asarray(sequences)

def get_next_input():
    return np.random.rand(batch_size, feature_size)

def get_next_reference_sequence():
    size = batch_size
    string = ""
    for i in range(size):
        string += dictionary_no_gap[random.randint(0,dictionary_size-2)]
    return string

def get_test_input():
    return np.random.rand(batch_size,feature_size)

def get_test_reference_input():
    size = batch_size
    string = ""
    for i in range(size):
        string += dictionary_no_gap[random.randint(0,dictionary_size-2)]
    return string