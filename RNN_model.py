#coding=utf-8
import tensorflow as tf
import numpy as np
from data_helper import build_glove_dic


log_dir = '/home/anch/tmp_trainable_keep_0p5_hidden_100_swap_lstm'

class TextRNN(object):
    """
    A RNN model for quora question pair problem
    """
 
    def __init__(self,

      ## architecture hyper-parameters
      rnn_type = "rnn", ## or lstm or gru ?
      nonlinear_type = "sigmoid", ## or relu ? temp use sigmoid.
      l2_reg_lambda=0.0,
      number_units=64,
      embedding_trainable=True, 
      ##train-related-parameters
      batch_size=64   
      
    ## hyper-parameters finish
    ):
        #placeholders for input, output and dropout
        ##input_s1,input_s2 输入ID而不是word_embedding
        self.input_s1 = tf.placeholder(tf.int32,[batch_size, None],name="s1")
        self.input_s2 = tf.placeholder(tf.int32,[batch_size, None],name="s2")
        ## Quora question pair label , 1 for same meaning, 0 otherwise
        self.input_y  = tf.placeholder(tf.int64,[batch_size],name="label")
        ## based on tensor-flow rnn model, we should know advance the sequence length of the sentence
        self.input_seqlen1 = tf.placeholder(tf.int32, [batch_size],name="seqlen1")        
        self.input_seqlen2 = tf.placeholder(tf.int32, [batch_size],name="seqlen2")  
        ## prevent overfitting
        self.dropout_keep_prob = tf.placeholder(tf.float32,name="dropout_keep_prob")
        ###placeholder finish
        
        ##hyper-parameters
        self.rnn_type       = rnn_type
        self.nonlinear_type  = nonlinear_type

        self.batch_size     = batch_size
        self.number_units   = number_units
        self.embedding_trainable = embedding_trainable   
        self.l2_reg_lambda  = l2_reg_lambda
        #Keeping track of l2 regularization loss(optional)
        self.l2_loss = tf.constant(0.0)

	self.num_classes = 2
        ##anch add for one_hot
	self.y_one_hot = tf.one_hot(self.input_y, self.num_classes)  
        
        self.init_weight()
	#print "rnn model object: self.init_weight() finish"
        self.add_encoder()
	#print "rnn model object: self.add_encoder() finish"
        #self.add_dropout()
	#print "rnn model object: self.add_dropout() finish"
        self.add_final_state()
	#print "rnn model object: self.add_final_state() finish"
 	self.add_loss()      
        
    def init_weight(self):
        ##Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            _, self.word_embedding = build_glove_dic()
            self.embedding_size = self.word_embedding.shape[1]
            self.W = tf.get_variable(name='word_embedding', shape = self.word_embedding.shape,dtype=tf.float32,
            initializer=tf.constant_initializer(self.word_embedding), trainable=self.embedding_trainable)

            ## s1,s2 的形状是[batch_size, sequence_length, embedding_size]
            self.s1 = tf.nn.embedding_lookup(self.W, self.input_s1)
	    print self.s1

            self.s2 = tf.nn.embedding_lookup(self.W, self.input_s2)
            self.x1 = self.s1
            self.x2 = self.s2
            
            
            
    def add_encoder(self):
	#print "enter encoder"
        if self.rnn_type=="rnn":
            cell = tf.contrib.rnn.BasicRNNCell(self.number_units)
        elif self.rnn_type=="lstm":
            cell = tf.contrib.rnn.BasicLSTMCell(self.number_units)
        elif self.rnn_type=="gru": ##needs to check
            cell = tf.nn.rnn_cell.GRUCell(self.number_units)

        #print "in add_encoer:cell generated"
	## for variable reuse ....
        self.init_state0 = tf.get_variable('init_state', [1,self.number_units],
                                      initializer = tf.constant_initializer(0.0))
	#print "ini_state0 is", self.init_state0.shape

        self.init_state = tf.tile(self.init_state0, [self.batch_size,1])
	#print "ini_state second is", self.init_state.shape
        
        self.rnn_outputs1, _ = tf.nn.dynamic_rnn(cell, self.x1, sequence_length = self.input_seqlen1, initial_state=self.init_state)
        self.rnn_outputs2, _ = tf.nn.dynamic_rnn(cell, self.x2, sequence_length = self.input_seqlen2, initial_state=self.init_state)

       
    #def add_dropout(self):
    #    #add droptout, as the model otherwise quickly overfits . really ??
    #    self.rnn_outputs1 = tf.nn.dropout(self.rnn_outputs1, self.dropout_keep_prob)
    #    self.rnn_outputs2 = tf.nn.dropout(self.rnn_outputs2, self.dropout_keep_prob)
       
    def add_final_state(self):
	#print "tf.range is",tf.range(self.batch_size)
	print "tf.shape(self.rnn_outputs1)[1] is",tf.shape(self.rnn_outputs1    )[1]

        self.idx1 = tf.range(self.batch_size)* tf.shape(self.rnn_outputs1)[1] + (self.input_seqlen1-1)
        self.idx2 = tf.range(self.batch_size)* tf.shape(self.rnn_outputs2)[1] + (self.input_seqlen2-1)
       
 	## just for test purpose begin
	self.Reshape_1 = tf.reshape(self.rnn_outputs1, [-1, self.number_units]) 
	self.Reshape_2 = tf.reshape(self.rnn_outputs2, [-1, self.number_units]) 
	## just for test purpose end

        self.last_rnn_output1 = tf.gather(tf.reshape(self.rnn_outputs1, [-1, self.number_units]),self.idx1)
        self.last_rnn_output2 = tf.gather(tf.reshape(self.rnn_outputs2, [-1, self.number_units]),self.idx2)
        
    def add_loss(self):
        ## caculte "difference" between encoder output of sentense1 and sentense2
        ## caculate the norm1 distance    
        self.diff     = self.last_rnn_output1 - self.last_rnn_output2          ##shape [batch_size, num_units]

        self.diff_abs = tf.abs(self.diff)                                 ##shape [batch_size, num_units]
        #self.diff_abs = self.diff                                 ##shape [batch_size, num_units]

	## r1.* r2
	self.multi_t = tf.multiply(self.last_rnn_output1, self.last_rnn_output2)
	self.multi   = tf.reduce_sum(self.multi_t, axis = 1)
	print "self.multi = ", self.multi
	self.multi = tf.expand_dims(self.multi, 1) ##add a dimension at the end axis
	print "after expand dim, self.multi is", self.multi

        #self.features_before_drop = self.diff_abs
	#print "self.diff_abs = ", self.diff_abs
	#print "self.last_rnn_output1", self.last_rnn_output1
	#print "self.last_rnn_output2", self.last_rnn_output2

        #self.features_before_drop = tf.concat(1,[self.diff_abs, self.last_rnn_output1, self.last_rnn_output2])
        self.features_concat = tf.concat([self.last_rnn_output1, self.last_rnn_output2, self.diff_abs, self.multi], axis=1)

	#self.features_concat_2 = tf.concat([self.features_concat_1,self.last_rnn_output2],axis=1)

	#self.features_concat_3 = tf.concat([self.features_concat_2, self.multi],axis=1)

	#use 4-features concat or not? Does it matter?
	self.features_before_drop = self.features_concat
	#self.features_before_drop = self.diff_abs


	self.features =  tf.nn.dropout(self.features_before_drop, self.dropout_keep_prob)

 	with tf.name_scope("after_concat_layer"):
	    W = tf.get_variable(
	     "W",
	     shape = [self.number_units*3+1, self.num_classes],
	     #shape = [self.number_units, self.num_classes],
	     initializer= tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]),name="b")
	    ## do not use l2 regularize ?
  	    self.l2_loss += tf.nn.l2_loss(W)
	    self.l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.features,W,b,name="scores")
	    self.predictions = tf.argmax(self.scores,1,name="predictions")

	##Caculate mean cross-entropy loss
	with tf.name_scope("loss"):
	    self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels = self.y_one_hot)
	    #self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.scores, labels = self.input_y)
	    self.loss   = tf.reduce_mean(self.losses) + self.l2_reg_lambda * self.l2_loss
## for visialization
	    tf.summary.scalar('cross_entropy_loss', self.loss) 
	    		
	with tf.name_scope("output_layer"):
	    correct_predictions = tf.equal(self.predictions, tf.argmax(self.y_one_hot,1))
	    self.accuracy	= tf.reduce_mean(tf.cast(correct_predictions,"float"), name = "accuracy")
## for visialization
	    tf.summary.scalar('Accuracy', self.accuracy)

	    print "self.predictions = ", self.predictions
	    print "self.input_y = ", self.input_y
	    self.TP =tf.reduce_sum( (self.predictions) * (self.input_y) )
	    print "self.TP is ", self.TP
            self.TP_FP = tf.reduce_sum(self.predictions)
	    print "self.TP_FP = ", self.TP_FP
	  
	    self.precision = tf.divide(   tf.cast(self.TP, tf.float32) , tf.cast(self.TP_FP,tf.float32)   )
	    print "self.precision is", self.precision 
## for visialization
	    tf.summary.scalar('Precision', self.precision)

	    self.FN = tf.reduce_sum ( (self.input_y) * (1- self.predictions) )
	    self.TP_FN = self.TP + self.FN
	    self.recall = tf.divide(  tf.cast(self.TP, tf.float32), tf.cast(self.TP_FN, tf.float32) )

# for visialization
	    tf.summary.scalar('Recall', self.recall)

# summaries合并
	    self.merged = tf.summary.merge_all()

# 写到指定的磁盘路径中
	    self.train_writer = tf.summary.FileWriter(log_dir + '/train')
	    self.test_writer = tf.summary.FileWriter(log_dir + '/test')

	      
	    #self.
        #self.losses =  [-self.input_y * tf.log(self.diff_exp) - (1-self.input_y) * tf.log(1-self.diff_exp+very_small)]     ## shape [batch_size]
	

#print "self.losses = ", tf.run(self.losses)
	
        #self.loss = tf.reduce_sum(self.losses)/(self.batch_size*2)                       ## shape [1,]
        #self.loss = tf.reduce_sum(self.losses)                    ## shape [1,]
	
        ## just for test-purpose of the wrapper of this file, NO Actual Ustage
	##self.val  = tf.Variable(initial_value=2.4)
	##self.loss += self.val


#        self.diff_abs_sum = tf.reduce_sum(self.diff_abs, axis=1)             ##shape [batch_size]
#        
#        ## squeeze the norm1 distance between (0,1)
#        self.diff_exp = tf.exp(-self.diff_abs_sum) ##shape [batch_size], 
#        
#        ## automatically learn the "threshold" 
#        ##"use this nolinear to map exp(-||x1-x2||) (L1 norm diff) to probability")
#        with tf.name_scope("threshold"):
#            self.thre_W = tf.Variable(name="W", initial_value = 1.0)
#            self.thre_b = tf.Variable(name="b", initial_value = tf.log(0.5))
#
#
#            ####self.wx_plus_b = self.diff_exp * self.thre_W + self.thre_b               ## shape [batch_size]
#            self.wx_plus_b = self.diff_exp + self.thre_b               ## shape [batch_size]
#
#
#        ##apply sigmoid OR relu ??
#        if (self.nonlinear_type == "sigmoid"):
#            self.prob = 1/(1+tf.exp(-1.0 * self.wx_plus_b)) ## shape[batch_size]
#        elif self.nonlinear_type == "relu":
#            self.prob = maximum(0,self.wx_plus_b) ## ?
#            
#        ## use logistic regression (softmax) cost
#        ## if y=1, prob = prob
#        ## if y=0, prob = 1-prob
#	print "self.input_y.shape is",self.input_y.shape
#	print "prob.shape is", self.prob.shape
#
#        ## anch add for accuracy:2018-02-04
#        self.comp = self.diff_exp - self.input_y
#        self.comp = tf.abs(self.comp)
#
#        self.resu = 0.5 - self.comp
#        self.resu = tf.sign(self.resu)
#        self.accuracy = tf.reduce_mean(tf.cast(self.resu, tf.float32))
#
#
#        self.losses =  [-self.input_y * tf.log(self.prob) - (1-self.input_y) * tf.log(1-self.prob)]     ## shape [batch_size]
##print "self.losses = ", tf.run(self.losses)
#	
#        self.loss = tf.reduce_sum(self.losses)                       ## shape [1,]
	
        ## just for test-purpose of the wrapper of this file, NO Actual Ustage
	##self.val  = tf.Variable(initial_value=2.4)
	##self.loss += self.val
