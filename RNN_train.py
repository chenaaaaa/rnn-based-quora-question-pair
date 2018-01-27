 #coding=utf-8
#! /usr/bin/env python

import datetime
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

import data_helper
from RNN_model import TextRNN

##Parameters
#===========================================================
#第一个是参数名称，第二个参数是默认值，第三个是参数描述

# Data loading params
tf.flags.DEFINE_float("train_sample_percentage", 0.75, "Percentage of the training data to use for validation")

tf.flags.DEFINE_string("data_file","train_small.csv","Data source")
#tf.flags.DEFINE_string("data_file","train.csv","Data source")

## Model Hyperparamters
tf.flags.DEFINE_string("rnn_type", "rnn", "use rnn or lstm or gru ")
#tf.flags.DEFINE_string("rnn_type", "lstm", "use rnn or lstm or gru ")

tf.flags.DEFINE_string("nonlinear_type", "sigmoid", "use this nonlinear to map exp(-||x1-x2||) to probability") 
tf.flags.DEFINE_integer("l2_reg_lambda", 0.0, " should consider whether to use this")
tf.flags.DEFINE_integer("number_units", 9, "rnn internal neural cells (h) numbers")
tf.flags.DEFINE_boolean("embedding_trainable", False, "whether word2vec embedding is trainable")
tf.flags.DEFINE_float("dropout_keep_prob",0.5,"Dropout keep probability (default:0.5)")


## no other rnn specific ?? 


## Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default : 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default:200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default:100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps(default:100)")
tf.flags.DEFINE_integer("num_checkpoints",5,"Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================

# Load data
print("Loading training data...")

s1, s2, label, seqlen1, seqlen2 = data_helper.read_data_sets(FLAGS.data_file)
print "in RNN_train.py read_data_sets finish"
#label = np.asarray([y] for y in label)
#print "label.type is", type(label)
sample_num = len(label)
train_end = int(sample_num * FLAGS.train_sample_percentage)

print "sample_num=",sample_num
print "train_end=",train_end

## Split train/test set
## TO_DO : This is very crude, should use cross-validation

s1_train, s1_dev =                s1[:train_end], s1[train_end:]
s2_train, s2_dev =                s2[:train_end], s2[train_end:]
y_train, y_dev   =             label[:train_end], label[train_end:]
seqlen1_train, seqlen1_dev = seqlen1[:train_end], seqlen1[train_end:]
seqlen2_train, seqlen2_dev = seqlen2[:train_end], seqlen1[train_end:]

print("Train/Dev split : {:d}/ {:d}".format(len(y_train), len(y_dev)))

## Training
#====================================================================


with tf.Graph().as_default():
   session_conf = tf.ConfigProto(
       allow_soft_placement=FLAGS.allow_soft_placement,
       log_device_placement=FLAGS.log_device_placement)
   sess = tf.Session(config=session_conf)
   with sess.as_default():
       ## create a rnn object.
   
       rnn = TextRNN(
               rnn_type        = FLAGS.rnn_type,
               nonlinear_type  = FLAGS.nonlinear_type,
               l2_reg_lambda    = FLAGS.l2_reg_lambda,
               number_units    = FLAGS.number_units,
               embedding_trainable = FLAGS.embedding_trainable,
               batch_size      = FLAGS.batch_size)
       print "A rnn class generated" 
       # Define Training procedure
       global_step = tf.Variable(0, name = "global_step", trainable=False)
       optimizer = tf.train.AdamOptimizer(1e-4/5)
       grads_and_vars = optimizer.compute_gradients(rnn.loss)
       train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
       
       #Keep track of gradient values and sparsity(optional)
       grad_summaries = []
       for g, v in grads_and_vars:
           if g is not None:
               grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
               sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
               grad_summaries.append(grad_hist_summary)
               grad_summaries.append(sparsity_summary)
       grad_summaries_merged = tf.summary.merge(grad_summaries)

       # Output directory for models and summaries
       timestamp = str(int(time.time()))
       out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
       print("Writing to {}\n".format(out_dir))
       # Summaries for loss and pearson
       loss_summary = tf.summary.scalar("loss", rnn.loss)
       #acc_summary = tf.summary.scalar("pearson", rnn.pearson)
       # Train Summaries
	#train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
       train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
       train_summary_dir = os.path.join(out_dir, "summaries", "train")
       train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

       # Dev summaries
#dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
       dev_summary_op = tf.summary.merge([loss_summary])
       dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
       dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

       # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
       checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
       checkpoint_prefix = os.path.join(checkpoint_dir, "model")
       if not os.path.exists(checkpoint_dir):
           os.makedirs(checkpoint_dir)
       saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

       # Initialize all variables
       sess.run(tf.global_variables_initializer())
       sess.run(tf.local_variables_initializer())
           
       ##def train_step(s1, s2, y) if use CNN, seqlen1/2 is not needed        
       def train_step(s1,s2,y, seqlen1, seqlen2):
           """
           A single training steps
           """
           feed_dict = {
               rnn.input_s1: s1,
               rnn.input_s2: s2,
               rnn.input_y : y,
               rnn.input_seqlen1 : seqlen1,
               rnn.input_seqlen2 : seqlen2,
               rnn.dropout_keep_prob:FLAGS.dropout_keep_prob
           }
           #_, step, summaries, loss , pearson
           _, step, summaries, loss = sess.run([train_op, global_step, train_summary_op, rnn.loss],feed_dict)

##Just for test purpose begin

 	   #self_x1      = sess.run([rnn.x1],feed_dict)
 	   #self_x2      = sess.run([rnn.x2],feed_dict)
	   #self_rnn_output1 = sess.run([rnn.rnn_outputs1],feed_dict)
           #print("self_rnn_output1={}".format(self_rnn_output1))

	   #self_rnn_output2 = sess.run([rnn.rnn_outputs2],feed_dict)
           #print("self_rnn_output2={}".format(self_rnn_output2))

	   #self_idx1	   = sess.run([rnn.idx1],feed_dict)
	   #print("self_idx1={}".format(self_idx1))


	   #self_idx2	   = sess.run([rnn.idx2],feed_dict)
	   #print("self_idx2={}".format(self_idx2))

	   #self_idx2	   = sess.run([rnn.idx2],feed_dict)
	   #self_reshape1   = sess.run([rnn.Reshape_1],feed_dict)
	   #self_reshape2   = sess.run([rnn.Reshape_2],feed_dict)
	   #print("self_reshape1 ={}".format(self_reshape1))
	   #print("self_reshape2 ={}".format(self_reshape2))
           ##print("self_rnn_output2={}".format(self_rnn_output2))

	   #self_rnn_final1 = sess.run([rnn.last_rnn_output1],feed_dict)
           #print("self_rnn_final1={}".format(self_rnn_final1))


	   #self_rnn_final2 = sess.run([rnn.last_rnn_output2],feed_dict)
           #print("self_rnn_final2={}".format(self_rnn_final2))

           #print("self_rnn_final2={}".format(self_rnn_final2))
           #print("self_x1 = {}".format(self_x1))
           #print("self_x2 = {}".format(self_x2))
           #print("self_rnn_diff_abs={}".format(self_rnn_diff_abs))
           #print("self_rnn_diff_abs_sum={}".format(self_rnn_diff_ab    s_sum))
           #print("self_rnn_diff_exp={}".format(self_rnn_diff_exp))
           #print("self_rnn_wx_plus_b={}".format(self_rnn_wx_plus_b)    )
           #print("self_prob = {}".format(self_prob))
           #print("self_losses={}".format(self_losses))
           #print("self_loss={}".format(self_loss))
           ##print("self_val={}".format(self_val))
           #print sess.run(rnn.loss)


	   #self_rnn_diff_abs = sess.run([rnn.diff_abs],feed_dict)
	   #self_rnn_diff_abs_sum = sess.run([rnn.diff_abs_sum],feed_dict)
	   #self_rnn_diff_exp 	= sess.run([rnn.diff_exp],feed_dict)
	   #self_rnn_wx_plus_b = sess.run([rnn.wx_plus_b],feed_dict)
	   #self_prob	      = sess.run([rnn.prob],feed_dict)
	   #self_losses	      = sess.run([rnn.losses],feed_dict)
	   #self_loss	      = sess.run([rnn.loss], feed_dict)
	   self_thre_W	      = sess.run([rnn.thre_W],feed_dict)
	   self_thre_b	      = sess.run([rnn.thre_b],feed_dict)
           print("self_thre_W={}".format(self_thre_W))
           print("self_thre_b={}".format(self_thre_b))
	   ##self_val	      = sess.run([rnn.val],feed_dict) 
##Just for test purpose end






###this is the main run step
           _, step, summaries, loss = sess.run([train_op, global_step, train_summary_op, rnn.loss],feed_dict)
           time_str = datetime.datetime.now().isoformat()
#           #print("{}: step {}, loss {:g}, pearson {:g}".format(time_str, step, loss, pearson))
           print("{}: step {}, train loss {:g}".format(time_str, step, loss))
           train_summary_writer.add_summary(summaries, step)       
### main run step finish

       def dev_step(s1, s2, y,seqlen1, seqlen2, writer=None):
           """
           Evaluates model on a dev set
           """
           feed_dict = {
               rnn.input_s1: s1,
               rnn.input_s2: s2,
               rnn.input_y : y,
               rnn.input_seqlen1 : seqlen1,
               rnn.input_seqlen2 : seqlen2 ,  
               rnn.dropout_keep_prob:1.0
           }
           
           ##step, summaries, loss, pearson = sess.run(
#[global_step, dev_summary_op, cnn.loss, cnn.pearson],
#feed_dict)
           
           step, summaries, loss = sess.run(
               [global_step, dev_summary_op, rnn.loss], feed_dict)
           time_str = datetime.datetime.now().isoformat()
           #print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, pearson))
           print("{}: step {}, dev loss {:g}".format(time_str, step, loss))
           if writer:
               writer.add_summary(summaries, step)

       ## Generate batches
       STS_train = data_helper.dataset(s1 = s1_train, s2=s2_train, label= y_train,\
		       		       seqlen1 = seqlen1_train,seqlen2 = seqlen2_train
		       			)
       # Training loop. For each batch...
       
       for i in range(40000):
	   #print "this is the key round"
	   print "batch =",i
           ## next_batch needs modify fo rnn.
           batch_train = STS_train.next_batch(FLAGS.batch_size)
           
           ###NOTICE.
           ### HERE we should run the "one step" of the train.
           #print "batch_train[0] = ", batch_train[0]
           #print "batch_train[1] = ", batch_train[1]
           #print "batch_train[2] = ", batch_train[2]
           #print "batch_train[3] = ", batch_train[3]
           #print "batch_train[4] = ", batch_train[4]
           train_step(batch_train[0], batch_train[1], batch_train[2], batch_train[3], batch_train[4])
           current_step = tf.train.global_step(sess, global_step)
           if current_step % FLAGS.evaluate_every == 0:
               print("\nEvaluation:")
               dev_step(s1_dev[:FLAGS.batch_size], s2_dev[:FLAGS.batch_size], y_dev[:FLAGS.batch_size], seqlen1_dev[:FLAGS.batch_size], seqlen2_dev[:FLAGS.batch_size],writer=dev_summary_writer)
               print("")
           if current_step % FLAGS.checkpoint_every == 0:
               path = saver.save(sess, checkpoint_prefix, global_step=current_step)
               print("Saved model checkpoint to {}\n".format(path))

            
	   print "----all the batch has run finish----"
                
