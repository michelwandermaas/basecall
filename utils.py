import tensorflow as tf
import numpy as np
import random
import sys

#variables

batch_size = 100
extend_size = 6
sequence_size = batch_size + (extend_size*2)
feature_size = 3
elements_size = 2
dictionary = ["A","C","G","T","-"]
dict = {"A":0,"C":1,"G":2,"T":3,"-":4}
dictionary_no_gap = ["A","C","G","T"]
dictionary_size = dictionary.__len__()
lstm_hidden_size = feature_size
number_of_layers = 3
learning_rate = 0.05
training_steps = 1000
type = tf.float32

numAcc = 2 #every x iterations for accuracy calculation




'''
    Rob mentioned normalising the data with a overall current mean parameter.
    Also he said something about these shift parameters.
'''


'''
    Alignment penalties: Multiply by probabilities
'''


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
gap_opening_score = -2
gap_identical_score = 0

#functions

# it`s impossible to restore the exact same value...

# I can simply transform the one-hot-encoding probabilities to an array of W`s size, and it would be made of 0`s and 1`s
# that would make one weight go up and the others go down
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
    #raise NotImplementedError

def score_match(A,B, match_score, mismatch_score, gap_score, gap_identical_score):
    if (A == "-" and B == "-"):
        return gap_identical_score
    elif (A==B):
        return match_score
    elif(A=='-' or B =='-'):
        return gap_score
    else:
        return mismatch_score

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
            sequences[len(sequences)-1].append(float(dict[x]))
        else:
            idx = 0
            sequences[len(sequences)-1].append(float(dict[x]))

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