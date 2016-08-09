import utils
import tensorflow as tf
import numpy as np

def training(get_next_input, get_next_reference_sequence, get_test_input = None, get_test_reference_input = None):

    sess = tf.Session()

    X = tf.placeholder(utils.type, [None, utils.feature_size])

    W = tf.Variable(tf.random_normal([utils.feature_size, utils.elements_size * utils.dictionary_size]), name="W")

    target_probabilties = tf.placeholder(utils.type, [utils.batch_size * utils.elements_size, utils.dictionary_size])

    lstm = tf.nn.rnn_cell.BasicLSTMCell(utils.lstm_hidden_size)

    stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * utils.number_of_layers)

    initial_state = state = stacked_lstm.zero_state(utils.batch_size, utils.type)

    # Input X to the network and get the output and the state

    output, state = stacked_lstm(X, state)

    #tf.scalar_summary("out", output)

    pred = tf.matmul(output,W)
    pred = tf.reshape(pred, (utils.batch_size * utils.elements_size, utils.dictionary_size))
    prediction = tf.nn.softmax(pred)
    #tf.scalar_summary("scaled_pred", pred)

    '''
        Calculations
    '''

    # instead of calculating this, I will calculate the difference between the target_W and the current W
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(pred, target_probabilties)

    #tf.scalar_summary("cross_entropy", cross_entropy)

    cost = tf.reduce_mean(cross_entropy)

    tf.scalar_summary("cost", cost)

    merged = tf.merge_all_summaries()

    trainWriter = tf.train.SummaryWriter("./train", sess.graph)

    optimizer = tf.train.GradientDescentOptimizer(utils.learning_rate).minimize(cost)

    sess.run(tf.initialize_all_variables())

    count = 0

    for i in range(utils.training_steps):
        count += 1
        getAcc = False
        if count == utils.numAcc:
            getAcc = True
            count = 0
        my_input = get_next_input()
        reference_sequence = get_next_reference_sequence()
        # out, w = sess.run([output, W], feed_dict={X:my_input})

        # I want to optimize W so my probabilities get right. I need to find the W that would make my probability equal to the reference.
        # All these calculations will produce a target_W that I will use to calculate the gradient in relation to current_W.
        # At the same time, now that I think about it, perhaps I should include more variables to optimize.
        prob = sess.run(prediction, feed_dict={X:my_input})

        prob = np.reshape(prob, (utils.batch_size, utils.elements_size, utils.dictionary_size))

        output_sequence = utils.get_sequence_from_prob(prob)
        aligned_reference = utils.align(output_sequence, reference_sequence)
        target_prob = utils.get_one_hot_encoding_prob(aligned_reference)

        prob = prob.reshape(utils.batch_size, utils.elements_size * utils.dictionary_size)
        target_prob = target_prob.reshape(utils.batch_size * utils.elements_size, utils.dictionary_size)

        opt, summary = sess.run([optimizer, merged], feed_dict={X: my_input, target_probabilties: target_prob})
        if (getAcc):
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