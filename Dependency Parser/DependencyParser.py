import collections
import tensorflow as tf
import numpy as np
import pickle
import math
from progressbar import ProgressBar

from DependencyTree import DependencyTree
from ParsingSystem import ParsingSystem
from Configuration import Configuration
import Config
import Util



'''
========================================================================================
It is only one file with multiple models I tried. Each has header and can be run by 
commenting current model and uncommenting the model we wanat to run
========================================================================================
'''

class DependencyParserModel(object):

    def __init__(self, graph, embedding_array, Config):

        self.build_graph(graph, embedding_array, Config)

    def build_graph(self, graph, embedding_array, Config):
        

        with graph.as_default():
            '''
            ===================================================================
            Default Configuration  as mentioned in paper
            ===================================================================
            '''
            self.embeddings = tf.Variable(embedding_array, dtype=tf.float32)
            self.train_inputs = tf.placeholder(tf.int32, shape=[None, Config.n_Tokens], name="train_inputs")
            

            self.train_labels = tf.placeholder(tf.float32, shape=[None, parsing_system.numTransitions()],
                                               name="train_labels")
            

            self.test_inputs = tf.placeholder(tf.int32, shape=[Config.n_Tokens], name="test_inputs")

            w_i = tf.Variable(
                tf.random_normal(shape=[Config.hidden_size, Config.n_Tokens * Config.embedding_size],
                                 stddev=0.1))  

            w_o = tf.Variable(tf.random_normal
                                         (shape=[parsing_system.numTransitions(), Config.hidden_size], stddev=0.1))

            b_i = tf.Variable(tf.random_normal(shape=[Config.hidden_size], stddev=0.1))


            e1 = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            e = tf.reshape(e1, [-1, Config.n_Tokens * Config.embedding_size])

    
            # Forward Pass
            self.prediction = self.forward_pass(e, w_i, b_i, w_o)
           


            #To remove -1 from labels
            v = tf.greater(self.train_labels, -1.0)
            v1 = tf.where(v)
            self.v1 = tf.where(v)
            c=tf.cast(v,self.train_labels.dtype)

            v1 = tf.reshape(v1, shape=[-1])
            self.t_l = self.train_labels * c

            strain_labels = tf.gather(self.train_labels, v1)
            
            #Loss Function

            self.loss = (tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.prediction, labels=(tf.argmax(self.t_l, 1)))) +
                         (Config.lam/2) * tf.nn.l2_loss(e) +
                         (Config.lam/2) * tf.nn.l2_loss(w_o) +
                         (Config.lam/2) * tf.nn.l2_loss(w_i) +
                         (Config.lam/2) * tf.nn.l2_loss(b_i)
                         )
            
            '''
            ===================================================================
            Different Non Linearities (Graph same as default only change in forward pass)
            ===================================================================
            with graph.as_default():
            
            self.embeddings = tf.Variable(embedding_array, dtype=tf.float32)
            self.train_inputs = tf.placeholder(tf.int32, shape=[None, Config.n_Tokens], name="train_inputs")
            

            self.train_labels = tf.placeholder(tf.float32, shape=[None, parsing_system.numTransitions()],
                                               name="train_labels")
            

            self.test_inputs = tf.placeholder(tf.int32, shape=[Config.n_Tokens], name="test_inputs")

            w_i = tf.Variable(
                tf.random_normal(shape=[Config.hidden_size, Config.n_Tokens * Config.embedding_size],
                                 stddev=0.1))  

            w_o = tf.Variable(tf.random_normal
                                         (shape=[parsing_system.numTransitions(), Config.hidden_size], stddev=0.1))

            b_i = tf.Variable(tf.random_normal(shape=[Config.hidden_size], stddev=0.1))


            e1 = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            e = tf.reshape(e1, [-1, Config.n_Tokens * Config.embedding_size])

    
            # Forward Pass for non linearity
            self.prediction = self.forward_nonlinearity(e, w_i, b_i, w_o)
           


            #To remove -1 from labels
            v = tf.greater(self.train_labels, -1.0)
            v1 = tf.where(v)
            self.v1 = tf.where(v)
            c=tf.cast(v,self.train_labels.dtype)

            v1 = tf.reshape(v1, shape=[-1])
            self.t_l = self.train_labels * c

            strain_labels = tf.gather(self.train_labels, v1)
            
            #Loss Function

            self.loss = (tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.prediction, labels=(tf.argmax(self.t_l, 1)))) +
                         (Config.lam/2) * tf.nn.l2_loss(e) +
                         (Config.lam/2) * tf.nn.l2_loss(w_o) +
                         (Config.lam/2) * tf.nn.l2_loss(w_i) +
                         (Config.lam/2) * tf.nn.l2_loss(b_i)
                         )
            
            optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
            grads = optimizer.compute_gradients(self.loss)
            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            
            #To try unclipped gradients, comment and uncomment relevant lines below
            
            self.app = optimizer.apply_gradients(clipped_grads)
            #self.app=optimizer.apply_gradients(grads)
            
            
            
            # Only prediction for test data
            test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            test_embed = tf.reshape(test_embed, [1, -1])
            self.test_pred = self.forward_nonlinearity(test_embed, w_i, b_i, w_o)

            # Initialization
            self.init = tf.global_variables_initializer()
            
            
            '''
          
            
            '''
            ===================================================================
            Best Model
            ===================================================================
            self.embeddings = tf.Variable(embedding_array, dtype=tf.float32)
            self.train_inputs = tf.placeholder(tf.int32, shape=[None, Config.n_Tokens], name="train_inputs")
            # self.train_inputs=x

            self.train_labels = tf.placeholder(tf.float32, shape=[None, parsing_system.numTransitions()],
                                               name="train_labels")
            # self.train_labels=x1

            self.test_inputs = tf.placeholder(tf.int32, shape=[Config.n_Tokens], name="test_inputs")



            w_i = tf.Variable(
                tf.random_normal(shape=[Config.hidden_size, Config.n_Tokens * Config.embedding_size],
                                 stddev=0.1))  
            w_o = tf.Variable(tf.random_normal
                                         (shape=[parsing_system.numTransitions(), Config.hidden_size], stddev=0.1))
            w_i1 = tf.Variable(tf.random_normal(shape=[Config.hidden_size, Config.hidden_size], stddev=0.1))

            b_i = tf.Variable(tf.random_normal(shape=[Config.hidden_size], stddev=0.1))
            b_i1 = tf.Variable(tf.random_normal(shape=[Config.hidden_size], stddev=0.1))


            e1 = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            e = tf.reshape(embed1, [-1, Config.n_Tokens * Config.embedding_size])


            # calling the forward_pass


            self.prediction = self.forward_two(e, w_i, w_i1, b_i,b_i1, w_o)



            #creating a filter to remove all non feasible targets
            v = tf.greater(self.train_labels, -1.0)
            v1 = tf.where(v)
            self.v1 = tf.where(v)
            casted = tf.cast(v, self.train_labels.dtype)

            v1 = tf.reshape(v1, shape=[-1])
            self.t_l = self.train_labels * casted

            strain_labels = tf.gather(self.train_labels, v1)
            
           #Loss Function

            self.loss = (tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.prediction, labels=(tf.argmax(self.t_l, 1)))) +
                         (Config.lam / 2) * tf.nn.l2_loss(e) +
                         (Config.lam / 2) * tf.nn.l2_loss(w_o) +
                         (Config.lam / 2) * tf.nn.l2_loss(w_i1) +

                         (Config.lam / 2) * tf.nn.l2_loss(w_i) +
                         (Config.lam / 2) * tf.nn.l2_loss(b_i) +

                         (Config.lam / 2) * tf.nn.l2_loss(b_i1)
                         )
            
            
            
            optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
            grads = optimizer.compute_gradients(self.loss)
            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            
            #To try unclipped gradients, comment and uncomment relevant lines below
            
            self.app = optimizer.apply_gradients(clipped_grads)
            #self.app=optimizer.apply_gradients(grads)
            
            
            
            # prediction test data
            test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            test_embed = tf.reshape(test_embed, [1, -1])
            self.test_pred = self.forward_two(test_embed, w_i,w_i1, b_i,b_i1, w_o)

            # Initialization
            self.init = tf.global_variables_initializer()
            
            ''' 
          
            '''
            ===================================================================
            Two Hidden Layer
            ===================================================================
            self.embeddings = tf.Variable(embedding_array, dtype=tf.float32)
            self.train_inputs = tf.placeholder(tf.int32, shape=[None, Config.n_Tokens], name="train_inputs")
            # self.train_inputs=x

            self.train_labels = tf.placeholder(tf.float32, shape=[None, parsing_system.numTransitions()],
                                               name="train_labels")
            # self.train_labels=x1

            self.test_inputs = tf.placeholder(tf.int32, shape=[Config.n_Tokens], name="test_inputs")



            w_i = tf.Variable(
                tf.random_normal(shape=[Config.hidden_size, Config.n_Tokens * Config.embedding_size],
                                 stddev=0.1))  
            w_o = tf.Variable(tf.random_normal
                                         (shape=[parsing_system.numTransitions(), Config.hidden_size], stddev=0.1))
            w_i1 = tf.Variable(tf.random_normal(shape=[Config.hidden_size, Config.hidden_size], stddev=0.1))

            b_i = tf.Variable(tf.random_normal(shape=[Config.hidden_size], stddev=0.1))
            b_i1 = tf.Variable(tf.random_normal(shape=[Config.hidden_size], stddev=0.1))


            e1 = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            e = tf.reshape(embed1, [-1, Config.n_Tokens * Config.embedding_size])


            # forward_pass


            self.prediction = self.forward_two(e, w_i, w_i1, b_i,b_i1, w_o)



            #to remove -1 labels
            v = tf.greater(self.train_labels, -1.0)
            v1 = tf.where(v)
            self.v1 = tf.where(v)
            casted = tf.cast(v, self.train_labels.dtype)

            v1 = tf.reshape(v1, shape=[-1])
            self.t_l = self.train_labels * casted

            strain_labels = tf.gather(self.train_labels, v1)
            
           #Loss Function

            self.loss = (tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.prediction, labels=(tf.argmax(self.t_l, 1)))) +
                         (Config.lam / 2) * tf.nn.l2_loss(e) +
                         (Config.lam / 2) * tf.nn.l2_loss(w_o) +
                         (Config.lam / 2) * tf.nn.l2_loss(w_i1) +

                         (Config.lam / 2) * tf.nn.l2_loss(w_i) +
                         (Config.lam / 2) * tf.nn.l2_loss(b_i) +

                         (Config.lam / 2) * tf.nn.l2_loss(b_i1)
                         )
            
            
            
            optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
            grads = optimizer.compute_gradients(self.loss)
            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            
            #To try unclipped gradients, comment and uncomment relevant lines below
            
            self.app = optimizer.apply_gradients(clipped_grads)
            #self.app=optimizer.apply_gradients(grads)
            
            
            
            # Only prediction for test data
            test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            test_embed = tf.reshape(test_embed, [1, -1])
            self.test_pred = self.forward_two(test_embed, w_i,w_i1, b_i,b_i1, w_o)

            # Initialization
            self.init = tf.global_variables_initializer()
            
            '''
            '''
            ===================================================================
            Three Hidden Layer
            ===================================================================
            self.embeddings = tf.Variable(embedding_array, dtype=tf.float32)
            self.train_inputs = tf.placeholder(tf.int32, shape=[None, Config.n_Tokens], name="train_inputs")
            

            self.train_labels = tf.placeholder(tf.float32, shape=[None, parsing_system.numTransitions()],
                                               name="train_labels")

            self.test_inputs = tf.placeholder(tf.int32, shape=[Config.n_Tokens], name="test_inputs")


            w_i = tf.Variable(
                tf.random_normal(shape=[Config.hidden_size, Config.n_Tokens * Config.embedding_size],
                                 stddev=0.1))  # r the tutorial document on Piazza
            w_o = tf.Variable(tf.random_normal
                                         (shape=[parsing_system.numTransitions(), Config.hidden_size], stddev=0.1))
            w_i1 = tf.Variable(tf.random_normal(shape=[Config.hidden_size, Config.hidden_size], stddev=0.1))
            w_i2 = tf.Variable(tf.random_normal
                                         (shape=[Config.hidden_size, Config.hidden_size], stddev=0.1))

            b_i = tf.Variable(tf.random_normal(shape=[Config.hidden_size], stddev=0.1))
            b_i1 = tf.Variable(tf.random_normal(shape=[Config.hidden_size], stddev=0.1))
            b_i2 = tf.Variable(tf.random_normal(shape=[Config.hidden_size], stddev=0.1))


            e1 = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            e = tf.reshape(e1, [-1, Config.n_Tokens * Config.embedding_size])


            # calling the forward_pass

            self.prediction=self.forward_pass_threehiddenlayers(e,w_i,w_i1,w_i2,b_i,b_i1,b_i2,w_o)

            v = tf.greater(self.train_labels, -1.0)
            v1 = tf.where(v)
            self.v1 = tf.where(v)
            casted = tf.cast(v, self.train_labels.dtype)

            v1 = tf.reshape(v1, shape=[-1])
            self.t_l = self.train_labels * casted

            strain_labels = tf.gather(self.train_labels, v1)
            # self.train_labels=strain_labels
    


            self.loss = (tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.prediction, labels=(tf.argmax(self.t_l, 1)))) +
                         Config.lam / 2 * tf.nn.l2_loss(e) +
                         Config.lam / 2 * tf.nn.l2_loss(w_o) +
                         Config.lam / 2 * tf.nn.l2_loss(w_i1) +
            lam*tf.nn.l2_loss(weights_input2)+tf.nn.l2_loss(b_i2)+

                         Config.lam / 2 * tf.nn.l2_loss(w_i) +
                         Config.lam / 2 * tf.nn.l2_loss(b_i) +

                         Config.lam / 2 * tf.nn.l2_loss(b_i1)
                         )
            
            optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
            grads = optimizer.compute_gradients(self.loss)
            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            
            #To try unclipped gradients, comment and uncomment relevant lines below
            
            self.app = optimizer.apply_gradients(clipped_grads)
            #self.app=optimizer.apply_gradients(grads)
            
            
            
            # Only prediction for test data
            test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            test_embed = tf.reshape(test_embed, [1, -1])
            self.test_pred = self.forward_two(test_embed, w_i,w_i1,w_i2,b_i,b_i1,b_i2,w_o)

            # Initialization
            self.init = tf.global_variables_initializer()
            
            '''
            '''
            ===================================================================
            Parallel Pass 
            ===================================================================
            self.embeddings = tf.Variable(embedding_array, dtype=tf.float32)
            self.train_inputs = tf.placeholder(tf.int32, shape=[None, Config.n_Tokens], name="train_inputs")
            

            self.train_labels = tf.placeholder(tf.float32, shape=[None, parsing_system.numTransitions()],
                                               name="train_labels")
           

            self.test_inputs = tf.placeholder(tf.int32, shape=[Config.n_Tokens], name="test_inputs")


            w_o = tf.Variable(tf.random_normal
                                         (shape=[parsing_system.numTransitions(), Config.hidden_size], stddev=0.1))
           
            #defining variables for parallel layers
            w_i_w=tf.Variable(tf.random_normal(shape=[Config.hidden_size,((Config.n_Tokens/3)+2)*Config.embedding_size], stddev=0.1)) #as per the tutorial document on Piazza
            w_i_p=tf.Variable(tf.random_normal(shape=[Config.hidden_size,((Config.n_Tokens/3)+2)*Config.embedding_size], stddev=0.1))
            w_i_l=tf.Variable(tf.random_normal(shape=[Config.hidden_size,((Config.n_Tokens/3)-4)*Config.embedding_size], stddev=0.1)) #as per the tutorial document on Piazza
            b_i=tf.Variable(tf.random_normal(shape=[Config.hidden_size],stddev=0.1))
            
            self.word1=tf.nn.embedding_lookup(self.embeddings,self.train_inputs[:,0:(Config.n_Tokens/3)+2])
            self.pos1=tf.nn.embedding_lookup(self.embeddings,self.train_inputs[:,((Config.n_Tokens/3)+2):(2*((Config.n_Tokens/3)+2))])
            self.label1=tf.nn.embedding_lookup(self.embeddings,self.train_inputs[:,2*((Config.n_Tokens/3)+2):])

            self.word=tf.reshape(self.word1,shape=[-1,Config.embedding_size*((Config.n_Tokens/3)+2)])
            self.pos=tf.reshape(self.pos1,  shape=[-1,Config.embedding_size*((Config.n_Tokens/3)+2)])
            self.label=tf.reshape(self.label1,shape=[-1,Config.embedding_size*((Config.n_Tokens/3)-4)])


            # forward_pass

            self.prediction=self.forward_parallel(w_i_w,self.word,w_i_p,self.pos,w_i_l,self.label,b_i,w_o)

            v = tf.greater(self.train_labels, -1.0)
            v1 = tf.where(v)
            self.v1 = tf.where(v)
            casted = tf.cast(vector, self.train_labels.dtype)

            v1 = tf.reshape(v1, shape=[-1])
            self.t_l = self.train_labels * casted

            strain_labels = tf.gather(self.train_labels, vector1)
            
            self.loss = (tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.prediction, labels=(tf.argmax(self.t_l, 1)))) +

                         Config.lam/2 * tf.nn.l2_loss(w_o) +
                         Config.lam/2 * tf.nn.l2_loss(w_i_l) +
                         Config.lam/2 * tf.nn.l2_loss(w_i_p) +
                         Config.lam/2 * tf.nn.l2_loss(w_i_w) +
                         Config.lam/2 * tf.nn.l2_loss(b_i)+
                         Config.lam/2*tf.nn.l2_loss(self.label)+
                         Config.lam/2*tf.nn.l2_loss(self.word)+
                         Config.lam/2*tf.nn.l2_loss(self.pos)

                         )
            
            
            
            optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
            grads = optimizer.compute_gradients(self.loss)
            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            self.app = optimizer.apply_gradients(clipped_grads)

            # For test data, we only need to get its prediction

            test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            test_embed = tf.reshape(test_embed, [1, -1])

            test_word1 = tf.nn.embedding_lookup(self.embeddings, self.test_inputs[:((Config.n_Tokens / 3) + 2)])
            test_pos1 = tf.nn.embedding_lookup(self.embeddings, self.test_inputs[((Config.n_Tokens / 3) + 2):(
                        2 * ((Config.n_Tokens / 3) + 2))])
            test_label1 = tf.nn.embedding_lookup(self.embeddings, self.test_inputs[2 * ((Config.n_Tokens / 3) + 2):])
            test_word = tf.reshape(test_word1, shape=[1,-1])
            test_pos= tf.reshape(test_pos1, shape=[1,-1])
            test_label = tf.reshape(test_label1, shape=[1,-1])


            self.test_pred=self.forward_parallel(w_i_w,test_word,
                                                        w_i_p, test_pos, w_i_l,test_label,
                                        b_i, w_o)

            # intializer
            self.init = tf.global_variables_initializer()
            
            '''
            
                   


            optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
            grads = optimizer.compute_gradients(self.loss)
            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            
            ''' To try unclipped gradients, comment and uncomment relevant lines below'''
            
            self.app = optimizer.apply_gradients(clipped_grads)
            #self.app=optimizer.apply_gradients(grads)
            
            
            
            # Only prediction for test data
            test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            test_embed = tf.reshape(test_embed, [1, -1])
            self.test_pred = self.forward_pass(test_embed, w_i, b_i, w_o)

            # Initialization
            self.init = tf.global_variables_initializer()

    def train(self, sess, num_steps):
        """

        :param sess:
        :param num_steps:
        :return:
        """
        self.init.run()
        print "Initailized"

        average_loss = 0
        for step in range(num_steps):
            start = (step * Config.batch_size) % len(trainFeats)
            end = ((step + 1) * Config.batch_size) % len(trainFeats)
            if end < start:
                start -= end
                end = len(trainFeats)
            batch_inputs, batch_labels = trainFeats[start:end], trainLabels[start:end]

            feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}

            _, loss_val = sess.run([self.app, self.loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % Config.display_step == 0:
                if step > 0:
                    average_loss /= Config.display_step
                print "Average loss at step ", step, ": ", average_loss
                average_loss = 0
            if step % Config.validation_step == 0 and step != 0:
                print "\nTesting on dev set at step ", step
                predTrees = []
                for sent in devSents:
                    numTrans = parsing_system.numTransitions()

                    c = parsing_system.initialConfiguration(sent)
                    while not parsing_system.isTerminal(c):
                        feat = getFeatures(c)
                        pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                        optScore = -float('inf')
                        optTrans = ""

                        for j in range(numTrans):
                            if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                                optScore = pred[0, j]
                                optTrans = parsing_system.transitions[j]

                        c = parsing_system.apply(c, optTrans)

                    predTrees.append(c.tree)
                result = parsing_system.evaluate(devSents, predTrees, devTrees)
                print result

        print "Train Finished."

    def evaluate(self, sess, testSents):
        """

        :param sess:
        :return:
        """

        print "Starting to predict on test set"
        predTrees = []
        for sent in testSents:
            numTrans = parsing_system.numTransitions()

            c = parsing_system.initialConfiguration(sent)
            while not parsing_system.isTerminal(c):
                # feat = getFeatureArray(c)
                feat = getFeatures(c)
                pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                optScore = -float('inf')
                optTrans = ""

                for j in range(numTrans):
                    if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                        optScore = pred[0, j]
                        optTrans = parsing_system.transitions[j]

                c = parsing_system.apply(c, optTrans)

            predTrees.append(c.tree)
        print "Saved the test results."
        Util.writeConll('result_test.conll', testSents, predTrees)


    def forward_pass(self, embed, weights_input, biases_input,weights_output):
        """

        :param embed:
        :param weights:
        :param biases:
        :return:
        """
        """
        =======================================================

        Implement the forwrad pass described in
        "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

        =======================================================
        """
        # For Default Model
        hidden_input = tf.add(tf.matmul(embed, tf.transpose(weights_input)), biases_input)
        non_linear = tf.pow(hidden_input, 3)
        hidden_output = tf.matmul(non_linear, tf.transpose(weights_output))
        return hidden_output
    
    '''  
    =======================================================
    Different Non linearities 
    =======================================================
        def forward_nonlinearity(self, embed, weights_input, biases_input, weights_output):
        """

        :param embed:
        :param weights:
        :param biases:
        :return:
        """


        hidden_input = tf.add(tf.matmul(embed, tf.transpose(weights_input)), biases_input)

        #Please comment uncomment relevant part
        #Default is assigned as Tanh

        non_linear=tf.tanh(hidden_input)
        #non_linear=tf.nn.relu(hidden_input)
        #non_linear=tf.sigmoid(hidden_input)
        #non_linear = tf.pow(hidden_input, 3)
        

        hidden_output = tf.matmul(non_linear, tf.transpose(weights_output))
        return hidden_output
    '''
    '''
    =======================================================
    Two hidden layer 
    =======================================================
    def forward_two(self, embed, weights_input1, weights_input2, biases_input1, bias_input2,
                                     weights_output):
        hidden_input1 = tf.add(tf.matmul(embed, tf.transpose(weights_input1)), biases_input1)
        non_linear1 = tf.pow(hidden_input1, 3)
        hidden_input2 = tf.add(tf.matmul(non_linear1, weights_input2), bias_input2)
        non_linear2 = tf.pow(hidden_input2, 3)
        hidden_output = tf.matmul(non_linear2, tf.transpose(weights_output))
        return hidden_ouput
    '''
    '''
    =======================================================
    Three hidden layer 
    =======================================================
    def forward_three(self, embed, weights_input1, weights_input2, weights_input3, biases_input1,
                                       bias_input2, bias_input3, weights_output):
        hidden_input1 = tf.add(tf.matmul(embed, tf.transpose(weights_input1)), biases_input1)
        non_linear1 = tf.pow(hidden_input1, 3)
        hidden_input2 = tf.add(tf.matmul(non_linear1, weights_input2), bias_input2)
        non_linear2 = tf.pow(hidden_input2, 3)
        hidden_input3 = tf.add(tf.matmul(non_linear2, weights_input3), bias_input3)
        hidden_output = tf.matmul(hidden_input3, tf.transpose(weights_output))
        return hidden_output
    '''
    '''
    =======================================================
    Forward pass parallel layers
    =======================================================
    def forward_parallel(self, weights_input_word, word, weights_input_pos, pos, weights_input_label, label,
                                biases_input, weights_output):
        hidden_input1 = tf.add(tf.matmul(word, tf.transpose(weights_input_word)), biases_input)
        non_linear1 = tf.pow(hidden_input1, 3)
        hidden_input2 = tf.add(tf.matmul(pos, tf.transpose(weights_input_pos)), biases_input)
        non_linear2 = tf.pow(hidden_input2, 3)
        hidden_input3 = tf.add(tf.matmul(label, tf.transpose(weights_input_label)), biases_input)
        non_linear3 = tf.pow(hidden_input3, 3)
        hidden_output = tf.matmul((tf.add(non_linear1, tf.add(non_linear2, non_linear3))), tf.transpose(weights_output))
        return hidden_output    
    '''
    
def genDictionaries(sents, trees):
    word = []
    pos = []
    label = []
    for s in sents:
        for token in s:
            word.append(token['word'])
            pos.append(token['POS'])

    rootLabel = None
    for tree in trees:
        for k in range(1, tree.n + 1):
            if tree.getHead(k) == 0:
                rootLabel = tree.getLabel(k)
            else:
                label.append(tree.getLabel(k))

    if rootLabel in label:
        label.remove(rootLabel)

    index = 0
    wordCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    wordCount.extend(collections.Counter(word))
    for word in wordCount:
        wordDict[word] = index
        index += 1

    posCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    posCount.extend(collections.Counter(pos))
    for pos in posCount:
        posDict[pos] = index
        index += 1

    labelCount = [Config.NULL, rootLabel]
    labelCount.extend(collections.Counter(label))
    for label in labelCount:
        labelDict[label] = index
        index += 1

    return wordDict, posDict, labelDict


def getWordID(s):
    if s in wordDict:
        return wordDict[s]
    else:
        return wordDict[Config.UNKNOWN]


def getPosID(s):
    if s in posDict:
        return posDict[s]
    else:
        return posDict[Config.UNKNOWN]


def getLabelID(s):
    if s in labelDict:
        return labelDict[s]
    else:
        return labelDict[Config.UNKNOWN]


def getFeatures(c):

    """
    =================================================================

    Implement feature extraction described in
    "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    =================================================================
    """
    w = []
    p = []
    l = []

    b_i = [c.getBuffer(0), c.getBuffer(1), c.getBuffer(2)]
    s_i = [c.getStack(0), c.getStack(1), c.getStack(2)]
    

    b_t_3 = []
    s_t_3 = []
    
    
    for i in b_i:
        b_t_3.append(c.getWord(i))
        b_t_3.append(c.getPOS(i))
    for i in s_i:
        s_t_3.append(c.getWord(i))
        s_t_3.append(c.getPOS(i))

    for i in range(len(s_t_3)):
        if i % 2 == 0:
            w.append(getWordID(s_t_3[i]))
            w.append(getWordID(b_t_3[i]))
        else:
            p.append(getPosID(s_t_3[i]))
            p.append(getPosID(b_t_3[i]))

    s_i = [c.getStack(0), c.getStack(1)]

    for index in s_i:
        left_child1 = c.getLeftChild(index, 1)
        w.append(getWordID(c.getWord(left_child1)))
        p.append(getPosID(c.getPOS(left_child1)))
        l.append(getLabelID(c.getLabel(left_child1)))

        right_child1 = c.getRightChild(index, 1)
        w.append(getWordID(c.getWord(right_child1)))
        p.append(getPosID(c.getPOS(right_child1)))
        l.append(getLabelID(c.getLabel(right_child1)))

        left_child2 = c.getLeftChild(index, 2)
        w.append(getWordID(c.getWord(left_child2)))
        p.append(getPosID(c.getPOS(left_child2)))
        l.append(getLabelID(c.getLabel(left_child2)))

        right_child2 = c.getRightChild(index, 2)
        w.append(getWordID(c.getWord(right_child2)))
        p.append(getPosID(c.getPOS(right_child2)))
        l.append(getLabelID(c.getLabel(right_child2)))

        lc_left_child1 = c.getLeftChild(left_child1, 1)
        w.append(getWordID(c.getWord(lc_left_child1)))
        p.append(getPosID(c.getPOS(lc_left_child1)))
        l.append(getLabelID(c.getLabel(lc_left_child1)))

        rc_right_child1 = c.getLeftChild(right_child1, 1)
        w.append(getWordID(c.getWord(rc_right_child1)))
        p.append(getPosID(c.getPOS(rc_right_child1)))
        l.append(getLabelID(c.getLabel(rc_right_child1)))

    return w + p + l


def genTrainExamples(sents, trees):
    numTrans = parsing_system.numTransitions()

    features = []
    labels = []
    pbar = ProgressBar()
    for i in pbar(range(len(sents))):
        if trees[i].isProjective():
            c = parsing_system.initialConfiguration(sents[i])

            while not parsing_system.isTerminal(c):
                oracle = parsing_system.getOracle(c, trees[i])
                feat = getFeatures(c)
                label = []
                for j in range(numTrans):
                    t = parsing_system.transitions[j]
                    if t == oracle:
                        label.append(1.)
                    elif parsing_system.canApply(c, t):
                        label.append(0.)
                    else:
                        label.append(-1.)

                if 1.0 not in label:
                    print i, label
                features.append(feat)
                labels.append(label)
                c = parsing_system.apply(c, oracle)
    return features, labels


def load_embeddings(filename, wordDict, posDict, labelDict):
    dictionary, word_embeds = pickle.load(open(filename, 'rb'))

    embedding_array = np.zeros((len(wordDict) + len(posDict) + len(labelDict), Config.embedding_size))
    knownWords = wordDict.keys()
    foundEmbed = 0
    for i in range(len(embedding_array)):
        index = -1
        if i < len(knownWords):
            w = knownWords[i]
            if w in dictionary:
                index = dictionary[w]
            elif w.lower() in dictionary:
                index = dictionary[w.lower()]
        if index >= 0:
            foundEmbed += 1
            embedding_array[i] = word_embeds[index]
        else:
            embedding_array[i] = np.random.rand(Config.embedding_size) * 0.02 - 0.01
    print "Found embeddings: ", foundEmbed, "/", len(knownWords)

    return embedding_array


if __name__ == '__main__':

    wordDict = {}
    posDict = {}
    labelDict = {}
    parsing_system = None

    trainSents, trainTrees = Util.loadConll('train.conll')
    devSents, devTrees = Util.loadConll('dev.conll')
    testSents, _ = Util.loadConll('test.conll')
    genDictionaries(trainSents, trainTrees)

    embedding_filename = 'word2vec.model'

    embedding_array = load_embeddings(embedding_filename, wordDict, posDict, labelDict)

    labelInfo = []
    for idx in np.argsort(labelDict.values()):
        labelInfo.append(labelDict.keys()[idx])
    parsing_system = ParsingSystem(labelInfo[1:])
    print parsing_system.rootLabel

    print "Generating Traning Examples"
    trainFeats, trainLabels = genTrainExamples(trainSents, trainTrees)
    print "Done."

    # Build the graph model
    graph = tf.Graph()
    model = DependencyParserModel(graph, embedding_array, Config)

    num_steps = Config.max_iter
    with tf.Session(graph=graph) as sess:

        model.train(sess, num_steps)

        model.evaluate(sess, testSents)

