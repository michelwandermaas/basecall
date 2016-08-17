import utils
import tensorflow as tf
import numpy as np
import sys

def training(get_next_input, get_next_reference_sequence, get_test_input = None, get_test_reference_input = None):

    sess = tf.Session()

    X = tf.placeholder(utils.type, [utils.batch_size, utils.feature_size])

    W = tf.Variable(tf.random_normal([utils.lstm_hidden_size*2, utils.elements_size * utils.dictionary_size], mean=1, stddev=0.3), name="W")

    target_probabilties = tf.placeholder(utils.type, [utils.batch_size * utils.elements_size, utils.dictionary_size])

    target_max = tf.placeholder(utils.type, [utils.batch_size * utils.elements_size])


    '''
    lstm = tf.nn.rnn_cell.BasicLSTMCell(utils.lstm_hidden_size)

    stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * utils.number_of_layers)

    initial_state = state = stacked_lstm.zero_state(utils.batch_size, utils.type)

    # Input X to the network and get the output and the state

    output, state = stacked_lstm(X, state)

    '''

    # so this one works differently. There`s a time notion in place, which I will probably have to think about later.
    # also, the output is different. there`s one output for the forward pass and one for the backward pass.
    #TODO: see if this is the way I want this to be, specially referring to the time_step, it looks incorrect

    #X = tf.split(0, utils.batch_size, X)

    with tf.variable_scope("fw1"):
        lstm_fw_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(utils.lstm_hidden_size, forget_bias=1.0, state_is_tuple=True)
    with tf.variable_scope("bw1"):
        lstm_bw_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(utils.lstm_hidden_size, forget_bias=1.0, state_is_tuple=True)

    with tf.variable_scope("fw2"):
        lstm_fw_cell_2 = tf.nn.rnn_cell.BasicLSTMCell(utils.lstm_hidden_size, forget_bias=1.0, state_is_tuple=True)
    with tf.variable_scope("bw2"):
        lstm_bw_cell_2 = tf.nn.rnn_cell.BasicLSTMCell(utils.lstm_hidden_size, forget_bias=1.0, state_is_tuple=True)

    with tf.variable_scope("fw3"):
        lstm_fw_cell_3 = tf.nn.rnn_cell.BasicLSTMCell(utils.lstm_hidden_size, forget_bias=1.0, state_is_tuple=True)
    with tf.variable_scope("bw3"):
        lstm_bw_cell_3 = tf.nn.rnn_cell.BasicLSTMCell(utils.lstm_hidden_size, forget_bias=1.0, state_is_tuple=True)

    with tf.variable_scope("layer1"):
        outputs_1, _, _ = tf.nn.bidirectional_rnn(lstm_fw_cell_1, lstm_bw_cell_1, [X], dtype=utils.type)

    with tf.variable_scope("layer2"):
        outputs_2, _, _ = tf.nn.bidirectional_rnn(lstm_fw_cell_2, lstm_bw_cell_2, outputs_1, dtype=utils.type)

    with tf.variable_scope("layer3"):
        output, _, _ = tf.nn.bidirectional_rnn(lstm_fw_cell_3, lstm_bw_cell_3, outputs_2, dtype=utils.type)

    output = tf.concat(0, output)
    #tf.scalar_summary("out", output)

    pred = tf.matmul(output,W)
    pred_reshaped = tf.reshape(pred, (utils.batch_size * utils.elements_size, utils.dictionary_size))
    prediction = tf.nn.softmax(pred_reshaped)
    #tf.scalar_summary("scaled_pred", pred)


    # approach I tried
    '''

    maxed_out = tf.to_float(tf.argmax(pred_reshaped, 1))

    max = tf.reshape(maxed_out, (utils.batch_size, utils.elements_size))
    '''

    # instead of calculating this, I will calculate the difference between the target_W and the current W
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(pred_reshaped, target_probabilties)

    #tf.scalar_summary("cross_entropy", cross_entropy)

    cost = tf.reduce_mean(cross_entropy)
    # approach I tried
    '''
    cost = tf.reduce_mean(tf.sub(target_max,maxed_out))
    '''
    tf.scalar_summary("cost", cost)

    merged = tf.merge_all_summaries()

    trainWriter = tf.train.SummaryWriter("./train", sess.graph)

    optimizer = tf.train.GradientDescentOptimizer(utils.learning_rate).minimize(cost)

    sess.run(tf.initialize_all_variables())

    for i in range(utils.training_steps):
        my_input = get_next_input()
        reference_sequence = get_next_reference_sequence()
        # out, w = sess.run([output, W], feed_dict={X:my_input})

        # I want to optimize W so my probabilities get right. I need to find the W that would make my probability equal to the reference.
        # All these calculations will produce a target_W that I will use to calculate the gradient in relation to current_W.
        # At the same time, now that I think about it, perhaps I should include more variables to optimize.
        prob, predict, out = sess.run([prediction, pred, output], feed_dict={X:my_input})

        prob_reshaped = np.reshape(prob, (utils.batch_size, utils.elements_size, utils.dictionary_size))

        output_sequence, output_avg_prob, non_output_avg_prob = utils.get_sequence_from_prob(prob_reshaped)
        aligned_reference = utils.align(output_sequence, reference_sequence, output_avg_prob, non_output_avg_prob)

        probabilities_dict = utils.get_prob_dict(prob)

        aligned_reference = utils.probability_alignment(probabilities_dict, reference_sequence)
        #print probabilities_dict

        target_prob = utils.get_one_hot_encoding_prob(aligned_reference)

        target_prob = target_prob.reshape(utils.batch_size * utils.elements_size, utils.dictionary_size)
        # approach I tried
        '''
        target_maxed = utils.get_argmax_encoding_prob(aligned_reference)
        '''


        opt, summary = sess.run([optimizer, merged], feed_dict={X: my_input, target_probabilties: target_prob})
        if (i % utils.numAcc):
            #print prob_reshaped
            #print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
            print output_sequence
            print "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
            print reference_sequence
            print "//////////////////////////////////////"
            print aligned_reference
            print "llllllllllllllllllllllllllllllllllllll"
            print "Training Accuracy: "+str(utils.get_sequences_identity(output_sequence, aligned_reference)) + "\n"
        trainWriter.add_summary(summary,i)
        #print prob
        #print cost
        #print cross_entropy

    trainWriter.close()

if __name__ == "__main__":
    training(utils.get_next_input, utils.get_next_reference_sequence)