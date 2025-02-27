import tensorflow as tf
import numpy as np
import random
import sys
import math

#variables

batch_size = 100
extend_size = batch_size/5
sequence_size = batch_size + (extend_size*2)
feature_size = 3
elements_size = 2
dictionary = ["A","C","G","T","-"]
alphabet_dict = {"A":0, "C":1, "G":2, "T":3, "-":4}
dictionary_no_gap = ["A","C","G","T"]
dictionary_size = dictionary.__len__()
lstm_hidden_size = 100
number_of_layers = 3
learning_rate = 0.01
training_steps = 10000
type = tf.float32

numAcc = 50 #every x iterations for accuracy calculation


mean_gaussian = 0.5
stdv_gaussian = 0.15
min_gaussian = 0.1

def gaussian(x, sigma, mu):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))

def get_gaussian_distr(size, sig_gaussian=mean_gaussian, mu_gaussian=mean_gaussian):
    '''
    :param size: size of the array returned
    :param sig_gaussian: sigma
    :param mu_gaussian: mu
    :return: np.array of size=size based on a gaussian distribution: f(x) with x=position in the array
    '''
    x = np.zeros(size)
    step = float(1.0/float(size))
    for i in range(size):
        x[i] = gaussian(i*step, sig_gaussian, mu_gaussian)
        if x[i] < min_gaussian:
            x[i] = min_gaussian
    return x

def get_sequences_identity(seq1,seq2):
    '''
    :return: the percentage of similarity between the two sequences
    '''
    equal = 0.0
    for i in range(len(seq1)):
        if seq1[i] == seq2[i]:
            equal += 1
    return equal/float(len(seq1))


def get_sequence_from_prob(probabilities):
    '''
    :param probabilities: dimensions = [batch_size][element_size][dictionary_size]
    :return: sequences: dimensions = [batch_size]
            output_avg_prob: average probability for the output base
            non_output_avg_prob: average probability for the other bases
    '''
    '''
        This is used to get the most probable sequence,by outputting the most probable base for each element in the event.
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

def get_prob_dict(prob):
    '''
        This is used to build a list of dictionaries from output probabilities in the dimensions:[batch_size*element_size].
        Each size dictionary will be in the form: key="Base" value:probability
    '''
    return_list = []
    new_prob = np.log(prob)
    for x in new_prob:
        dict = {}
        for y in range(len(x)):
            dict[dictionary[y]] = x[y]
        return_list.append(dict)
    return return_list

minimum_length_ref_seq = batch_size*elements_size #minimum size of the reference sequence used in the alignment

def align_by_prob(probabilities,reference_sequence):
    '''
    :param probabilities: output probabilities
    :param reference_sequence: reference sequence, probability larger than needed
    :return: best alignment found, or one that satisfies the minimum requirements
    '''
    '''
        The function receives a large reference sequence and aligns it to the probabilities.
        It starts with a small (minimum_length_ref_seq) sequence in the middle of the reference sequence, and it keeps
        inscreasing the size of the sequence to be used in the alignment until the average alignment probability is higher
        than a fixed constant or until it uses all the reference sequence.
    '''
    length = minimum_length_ref_seq
    best_alignment = ""
    middle_point = len(reference_sequence)/2
    avg_score = -sys.maxint
    best_score = -sys.maxint
    while length < len(reference_sequence) and avg_score < -1.6:
        start_point = middle_point - length/2
        end_point = middle_point + length/2
        alignment, score = deepnano_align(probabilities, reference_sequence[start_point:end_point])
        avg_score = float(float(score)/float(len(probabilities)))
        if avg_score > best_score:
            best_score = avg_score
            best_alignment = alignment
        length += 8

    return best_alignment

def deepnano_align(probabilities, reference_sequence):
    '''
        This alignment is based on deepnano`s approach for the alignment. It is a dynamic programming algorithm,
        it fills each cell of the array with the max score of one of the three possible cenarios:
            1. gap, gap
            2. gap, base
            3. base, base
        At the end it tracesback from the scores and figures the best alignment. It is not a global alignment, but it
        requires the alignment to use all the probabilities. Therefore the best alignment is somewhere in the last row,
        assuming rows are made out of probabilities and columns of the reference sequence.
    '''

    scores = np.zeros(((batch_size*elements_size)+1, len(reference_sequence)+1))
    scores = scores.tolist()
    alignments = np.chararray(((batch_size*elements_size)+1, len(reference_sequence)+1))
    alignments.fill("")
    alignments = alignments.tolist()
    for i in range(2, (batch_size*elements_size)+1):
        for j in range(1, len(reference_sequence)+1):
            if i > 2:
                best_score = scores[i-2][j] + probabilities[i-2]["-"] + probabilities[i-1]["-"]
                best_string = "--"
            else:
                best_score = scores[i-2][j-1] + probabilities[i-2]["-"] + probabilities[i-1][reference_sequence[j-1]]
                best_string = "-" + reference_sequence[j - 1]

            scoring = scores[i-2][j-1] + probabilities[i-2]["-"] + probabilities[i-1][reference_sequence[j-1]]
            if scoring > best_score:
                best_score = scoring
                best_string = "-"+reference_sequence[j-1]

            if j > 1:
                scoring = scores[i-2][j-2] + probabilities[i-2][reference_sequence[j-2]] + probabilities[i-1][reference_sequence[j-1]]
                if scoring > best_score:
                    best_score = scoring
                    best_string = reference_sequence[j-2] + reference_sequence[j-1]

            alignments[i][j] = best_string
            scores[i][j] = best_score

    best_pos = min(len(reference_sequence), minimum_length_ref_seq, batch_size*elements_size) - 1

    for j in range(best_pos, len(reference_sequence)):
        if (scores[(batch_size*elements_size)-1][j] > scores[(batch_size*elements_size)-1][best_pos]):
            best_pos = j

    ipos = (batch_size*elements_size)
    jpos = best_pos
    # One might want to force a global alignment with the following line
    #jpos = len(reference_sequence)

    final_alignment = ""

    while ipos > 0:
        string = alignments[ipos][jpos]
        final_alignment += string[::-1]
        if (string == ""):
            print ipos
            print jpos
            raise Exception
        if string[0] == "-" and string[1] == "-":
            ipos -= 2
        elif string[0] == "-" and string[1] != "-":
            ipos -= 2
            jpos -= 1
        elif string[0] != "-" and string[1] != "-":
            ipos -= 2
            jpos -= 2

    return final_alignment, scores[(batch_size*elements_size)][best_pos]

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


'''
    Functions that assign random values for the inputs of the training network.
    It is used to test if the network is properly working.
'''
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