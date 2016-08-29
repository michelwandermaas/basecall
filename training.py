import utils
import tensorflow as tf
import numpy as np
import sys

def training(get_next_input, get_next_reference_sequence, get_test_input = None, get_test_reference_input = None):
    '''
        Network layout:

            1. Input: events features, reference sequence
            2. 3 layers of bidirectional LSTMs with hidden_size=utils.lstm_hidden_size
            3. Multiply output by variable W
            4. Align the output probabilities with the reference sequence
            5. Get one hot encoding probabilities for the reference alignment
            6. Calculate cost=(reference_prob - output_prob)*gaussian_distribution.
                This way getting the probabilities right in the middle events is prioritized.
            7. Minimize the loss (applying gradients)
    '''

    sess = tf.Session()

    X = tf.placeholder(utils.type, [utils.batch_size, utils.feature_size])

    W = tf.Variable(tf.random_normal([utils.lstm_hidden_size*2, utils.elements_size * utils.dictionary_size], mean=1, stddev=0.3), name="W")

    target_probabilties = tf.placeholder(utils.type, [utils.batch_size * utils.elements_size, utils.dictionary_size])

    gaussian = tf.placeholder(utils.type, [utils.batch_size * utils.elements_size, utils.dictionary_size])

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

    pred = tf.matmul(output,W)

    pred_reshaped = tf.reshape(pred, (utils.batch_size * utils.elements_size, utils.dictionary_size))

    prediction = tf.nn.softmax(pred_reshaped)

    loss = tf.abs(tf.sub(prediction, target_probabilties) * gaussian)

    cost = tf.reduce_mean(loss)

    tf.scalar_summary("cost", cost)

    merged = tf.merge_all_summaries()

    trainWriter = tf.train.SummaryWriter("./training_tensorboard", sess.graph)

    optimizer = tf.train.GradientDescentOptimizer(utils.learning_rate).minimize(cost)

    sess.run(tf.initialize_all_variables())

    aux_gauss = utils.get_gaussian_distr(utils.batch_size*utils.elements_size)

    extended_gauss = np.zeros((utils.batch_size * utils.elements_size, utils.dictionary_size))

    for i in range(utils.batch_size * utils.elements_size):
        for j in range(utils.dictionary_size):
            extended_gauss[i][j] = aux_gauss[i]

    for i in range(utils.training_steps):
        my_input = get_next_input()
        reference_sequence = get_next_reference_sequence()

        prob, predict, out = sess.run([prediction, pred, output], feed_dict={X:my_input})

        prob_reshaped = np.reshape(prob, (utils.batch_size, utils.elements_size, utils.dictionary_size))

        probabilities_dict = utils.get_prob_dict(prob)

        aligned_reference = utils.align_by_prob(probabilities_dict, reference_sequence)

        target_prob = utils.get_one_hot_encoding_prob(aligned_reference)

        target_prob = target_prob.reshape(utils.batch_size * utils.elements_size, utils.dictionary_size)

        opt, summary = sess.run([optimizer, merged], feed_dict={X: my_input, target_probabilties: target_prob, gaussian: extended_gauss})
        if (i % utils.numAcc == 0):
            '''
                Calculate accuracy and print
            '''
            output_sequence, output_avg_prob, non_output_avg_prob = utils.get_sequence_from_prob(prob_reshaped)
            print "Iteration "+str(i)
            print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
            print output_sequence
            #print "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
            #print reference_sequence
            print "//////////////////////////////////////"
            print aligned_reference
            print "llllllllllllllllllllllllllllllllllllll"
            print "Training Accuracy: "+str(utils.get_sequences_identity(output_sequence, aligned_reference)) + "\n"
        trainWriter.add_summary(summary,i)

    trainWriter.close()

if __name__ == "__main__":
    training(utils.get_next_input, utils.get_next_reference_sequence)