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

# Nerual network architecture

tf.flags.DEFINE_string("RUN_MODEL", "two-feature-substrate-trainable-swap", "model used")
tf.flags.DEFINE_float("train_sample_percentage", 0.75, "Percentage of the training data to use for validation")

#tf.flags.DEFINE_string("data_file","train_small.csv","Data source")
#tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default : 64)")
#tf.flags.DEFINE_integer("train_size", 49152, "train size")

tf.flags.DEFINE_string("data_file","train.csv","Data source")
#tf.flags.DEFINE_integer("batch_size", 2048, "Batch Size (default : 64)")
#tf.flags.DEFINE_integer("batch_size", 1536, "Batch Size (default : 64)")
tf.flags.DEFINE_integer("batch_size", 1024, "Batch Size (default : 64)")

tf.flags.DEFINE_integer("train_size", 307200, "train size")



tf.flags.DEFINE_string("test_file","/home/anch/quora_data/test.csv","Test Data source")

## Model Hyperparamters
#tf.flags.DEFINE_string("rnn_type", "rnn", "use rnn or lstm or gru ")
#tf.flags.DEFINE_string("rnn_type", "gru", "use rnn or lstm or gru ")
tf.flags.DEFINE_string("rnn_type", "lstm", "use rnn or lstm or gru ")

tf.flags.DEFINE_string("nonlinear_type", "sigmoid", "use this nonlinear to map exp(-||x1-x2||) to probability") 
tf.flags.DEFINE_integer("l2_reg_lambda", 0.0, " should consider whether to use this")
tf.flags.DEFINE_integer("number_units", 100, "rnn internal neural cells (h) numbers")
tf.flags.DEFINE_boolean("embedding_trainable", True, "whether word2vec embedding is trainable")

#tf.flags.DEFINE_float("dropout_keep_prob",1.0,"Dropout keep probability (default:0.5)")
tf.flags.DEFINE_float("dropout_keep_prob",0.5,"Dropout keep probability (default:0.5)")



## Training parameters
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default:200)")
tf.flags.DEFINE_integer("evaluate_every", 4, "Evaluate model on dev set after this many steps (default:100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps(default:100)")
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
#print ("Loading test data...")
#s1_test, s2_test, seqlen1_test, seqlen2_test = data_helper.read_test_sets(FLAGS.test_file)

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
       optimizer = tf.train.AdamOptimizer(0.01)
       grads_and_vars = optimizer.compute_gradients(rnn.loss)
       train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
       
#       #Keep track of gradient values and sparsity(optional)
#       grad_summaries = []
#       for g, v in grads_and_vars:
#           if g is not None:
#               grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
#               sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
#               grad_summaries.append(grad_hist_summary)
#               grad_summaries.append(sparsity_summary)
#       grad_summaries_merged = tf.summary.merge(grad_summaries)

       # Output directory for models and summaries
#       timestamp = str(int(time.time()))
#       out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
#       print("Writing to {}\n".format(out_dir))
#       # Summaries for loss and pearson
#       loss_summary = tf.summary.scalar("train_loss", rnn.loss)
#       #acc_summary = tf.summary.scalar("pearson", rnn.pearson)
#       # Train Summaries
#	#train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
#       train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
#       train_summary_dir = os.path.join(out_dir, "summaries", "train")
#       train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
#
       # Dev summaries
#dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
#       dev_summary_op = tf.summary.merge([loss_summary])
#       dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
#       dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
#
       # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
#       checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
#       checkpoint_prefix = os.path.join(checkpoint_dir, "model")
#       if not os.path.exists(checkpoint_dir):
#           os.makedirs(checkpoint_dir)
#       saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
#
       # Initialize all variables
       sess.run(tf.global_variables_initializer())
       sess.run(tf.local_variables_initializer())
           
       ##def train_step(s1, s2, y) if use CNN, seqlen1/2 is not needed        
       def train_step(s1,s2,y, seqlen1, seqlen2,i ):
           """
           A single training steps
           """
	   #print "train_step round ", i
           feed_dict = {
               rnn.input_s1: s1,
               rnn.input_s2: s2,
               rnn.input_y : y,
               rnn.input_seqlen1 : seqlen1,
               rnn.input_seqlen2 : seqlen2,
               rnn.dropout_keep_prob:FLAGS.dropout_keep_prob
           }

           #self_accu, self_precision , self_recall, self_loss     = sess.run([rnn.accuracy, rnn.precision, rnn.recall, rnn.loss], feed_dict)
 	   #print "precision = {}".format(self_precision),  "    recall = {}".format(self_recall)



	   #train_accuracy  = tf.summary.scalar('train accuracy', self_accu)
	   #train_precision = tf.summary.scalar('train precision', self_precision)
	   #train_recall	   = tf.summary.scalar('train recall', self_recall)
	   #train_loss	   = tf.summary.scalar('train loss', self_loss)
           #train_summary_op = tf.summary.merge([train_loss, train_accuracy, train_recall, train_precision])


###this is the main run step
## train_op is used for optimization
           run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
           run_metadata = tf.RunMetadata()
           _, step, loss,summary  = sess.run([train_op, global_step, rnn.loss, rnn.merged],
					feed_dict= feed_dict, 
					options=run_options,
					run_metadata=run_metadata
)
           #rnn.train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
           rnn.train_writer.add_summary(summary, i )

           time_str = datetime.datetime.now().isoformat()
#           #print("{}: step {}, loss {:g}, pearson {:g}".format(time_str, step, loss, pearson))
           #print("{}: step {}, train loss {:g}".format(time_str, step, loss))
           #rnn.train_summary_writer.add_summary(summaries, step)       
### main run step finish

#       def dev_step(s1, s2, y,seqlen1, seqlen2, writer=None):
#           """
#           Evaluates model on a dev set
#           """
#           feed_dict = {
#               rnn.input_s1: s1,
#               rnn.input_s2: s2,
#               rnn.input_y : y,
#               rnn.input_seqlen1 : seqlen1,
#               rnn.input_seqlen2 : seqlen2 ,  
#               rnn.dropout_keep_prob:1.0
#           }
#           
#           ##step, summaries, loss, pearson = sess.run(
##[global_step, dev_summary_op, cnn.loss, cnn.pearson],
##feed_dict)
#           
#           step, summaries, loss = sess.run(
#               [global_step, dev_summary_op, rnn.loss], feed_dict)
#           time_str = datetime.datetime.now().isoformat()
#           #print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, pearson))
#           print("     {}: step {}, dev loss {:g}".format(time_str, step, loss))
#           if writer:
#               writer.add_summary(summaries, step)
#
#           self_accu      = sess.run([rnn.accuracy],feed_dict)
#           print ("         test accuracy = {}".format(self_accu))


       ## Generate batches
       STS_train = data_helper.dataset(s1 = s1_train, s2=s2_train, label= y_train,\
		       		       seqlen1 = seqlen1_train,seqlen2 = seqlen2_train
		       			)
##Following is the main loop
       
       # Training loop. For each batch...
       
       for i in range(40000):
           epoch_percentage = float( FLAGS.batch_size/ FLAGS.train_size * i * 1.0) 
	   if (i % 300 == 0): 
               print "architec is", FLAGS.RUN_MODEL
	       print "batch_i(th) =",i
	       print "batch_size = ", FLAGS.batch_size
	       print "data_file = ", FLAGS.data_file
	       print "neural-network type is ", FLAGS.rnn_type
	       print "keep-drop-prob = ", FLAGS.dropout_keep_prob
               ## next_batch needs modify fo rnn.
           batch_train = STS_train.next_batch(FLAGS.batch_size)
           
           ###NOTICE.
           ### HERE we should run the "one step" of the train.
           #print "batch_train[0] = ", batch_train[0]
           #print "batch_train[1] = ", batch_train[1]
           #print "batch_train[2] = ", batch_train[2]
           #print "batch_train[3] = ", batch_train[3]
           #print "batch_train[4] = ", batch_train[4]
##this is the train optimize
	   #print "Train :"
           #print "at mini-batch ", i
           train_step(batch_train[0], batch_train[1], batch_train[2], batch_train[3], batch_train[4], i )
           current_step = tf.train.global_step(sess, global_step)


	   num_dev = len(y_dev)/ FLAGS.batch_size
	   #print "num_dev = ", num_dev


##this is validation
           if i % FLAGS.evaluate_every == 0:
	       batsize = FLAGS.batch_size
               #print("\nEvaluation:")
	       accu_dev_list 		= []
	       precision_dev_list 	= []
	       recall_dev_list 		= []
               loss_dev_list 		= []
	       del accu_dev_list[:]
	       for j in range(num_dev):
                    #print "dev round = ",j
                    feed_dict = { 
                        rnn.input_s1: s1_dev           [j*batsize: (j+1) * batsize], 
                        rnn.input_s2: s2_dev           [j*batsize: (j+1) * batsize], 
                        rnn.input_y : y_dev            [j*batsize: (j+1) * batsize],
                        rnn.input_seqlen1 : seqlen1_dev[j*batsize: (j+1) * batsize],
                        rnn.input_seqlen2 : seqlen2_dev[j*batsize: (j+1) * batsize],   
                        rnn.dropout_keep_prob:1.0
                    }
             
                    ##step, summaries, loss, pearson = sess.run(
         #[global_step, dev_summary_op, cnn.loss, cnn.pearson],
         #feed_dict)
            
                    step, dev_loss, dev_accu, dev_preci, dev_recall, dev_summary  = sess.run([global_step, rnn.loss, rnn.accuracy, rnn.precision, rnn.recall, rnn.merged], feed_dict)

	
		    accu_dev_list.append(dev_accu)
                    precision_dev_list.append(dev_preci)
                    recall_dev_list.append(dev_recall)
                    loss_dev_list.append(dev_loss) 
	       #print "dev at  mini-batch ", i
	       dev_average_accuracy  = tf.reduce_mean(accu_dev_list)
	       dev_average_precision = tf.reduce_mean(precision_dev_list)
	       dev_average_recall    = tf.reduce_mean(recall_dev_list)
	       dev_average_loss      = tf.reduce_mean(loss_dev_list)

	       dev_loss_summary 	= tf.summary.scalar('dev average loss', dev_average_loss)
	       dev_accuracy_summary 	= tf.summary.scalar('dev average accuracy', dev_average_accuracy)
	       dev_recall_summary 	= tf.summary.scalar('dev average recall', dev_average_recall)
	       dev_precision_summary 	= tf.summary.scalar('dev average precision', dev_average_precision)
	       print "DEV loss: %.4f ; accuracy: %.4f ; recall: %.4f ; precison: %.4f ; DEV@ %d "%(\
	       sess.run(dev_average_loss),\
               sess.run(dev_average_accuracy),\
	       sess.run(dev_average_recall),\
               sess.run(dev_average_precision),\
	       i)


	       #dev_summary_merge = tf.summary.merge_all()  
	       #dev_summary_merge  	= tf.summary.merge(dev_loss_summary)
	       dev_summary_merge  	= tf.summary.merge([dev_loss_summary, dev_accuracy_summary, dev_recall_summary, dev_precision_summary])
	       dev_summary_merge_run    = sess.run(dev_summary_merge)

	       #rnn.test_writer.add_summary(dev_summary_merge_run, i )	       
##
#       loss_summary = tf.summary.scalar("train_loss", rnn.loss)
#       #acc_summary = tf.summary.scalar("pearson", rnn.pearson)
#       # Train Summaries
#       #train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
#       train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
#       train_summary_dir = os.path.join(out_dir, "summaries", "train")
#       train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

#               print "average accuracy  = ", sess.run(tf.reduce_mean(acc_dev_list))
#               print "average precision = ", sess.run(tf.reduce_mean(precision_dev_list))
#               print "average recall    = ", sess.run(tf.reduce_mean(recall_dev_list))
#
#           if i % FLAGS.checkpoint_every == 0:
#               path = saver.save(sess, checkpoint_prefix, global_step=current_step)
#               print("Saved model checkpoint to {}\n".format(path))

            
	   #print "----all the batch has run finish----"
                
