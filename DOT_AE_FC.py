# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:54:23 2020

@author: Yun Zou
"""

import tensorflow as tf
import numpy as np
import sys
import PPS,Born
import os
import scipy.io as io
import matplotlib.pyplot as plt


'''
##simulation
simu_r,simu_t,batch_ys_a,depth_a = PPS.dataprocess()

## extract parameters
simu_depth=depth_a[:,4]
simu_radius=depth_a[:,0]
simu_mua0 = depth_a[:,3]
simu_musp0 = depth_a[:,2]
n1 = len(simu_r)
simu_intra = simu_r

## get perturbation
simu_r_real = simu_r[:,:126]
simu_r_img = simu_r[:,126:252]
simu_r_complex = simu_r_real+1j*simu_r_img
simu_t_real = simu_t[:,:126]
simu_t_img = simu_t[:,126:252]
simu_t_complex = simu_t_real+1j*simu_t_img
simu_pert = (simu_t_complex-simu_r_complex[:int(n1),:])/simu_r_complex[:int(n1),:]
simu_pert = np.concatenate((np.real(simu_pert),np.imag(simu_pert)),1)


simu_Born,simu_kernel,simu_vz,simu_d1 = Born.Born_weights(simu_intra,simu_depth,simu_radius,simu_mua0,simu_musp0)

simu_pert,simu_image =  simu_pert,simu_image
print('Finish Loading Simu Data')
'''

## load phantom data
'''
p_r, p_t, p_image,p_paras = PPS.phantom_data()

p_depth=p_paras[:,5]
p_radius=p_paras[:,0]
p_mua0 = p_paras[:,4]
p_musp0 = p_paras[:,3]
n2 = len(p_r)
p_intra = np.array(p_r[:,:]).reshape((int(n2),252))

p_Born, p_kernel, p_vz, p_d1 = Born.Born_weights(p_intra,p_depth,p_radius,p_mua0,p_musp0 )

p_r_real = p_r[:,:126]
p_r_img = p_r[:,126:252]
p_r_complex = p_r_real+1j*p_r_img
p_t_real = p_t[:,:126]
p_t_img = p_t[:,126:252]
p_t_complex = p_t_real+1j*p_t_img
p_pert = (p_t_complex-p_r_complex[:int(n2),:])/p_r_complex[:int(n2),:]
p_pert = np.concatenate((np.real(p_pert),np.imag(p_pert)),1)
print('Finish Loading Phantom Data')
#delete_index = [12,13,14,15,18,51,52]
#delete_index = [18]
'''


'''
##clinical data load Washu2 system
clinical_r, clinical_t, clinical_image,clinical_paras = PPS.test_data()

clinical_depth=clinical_paras[:,5]
clinical_radius=clinical_paras[:,0]
clinical_mua0 = clinical_paras[:,4]
clinical_musp0 = clinical_paras[:,3]
n3 = len(clinical_r)
clinical_intra = np.array(clinical_r[:,:]).reshape((int(n3/1),252))

clinical_Born,clinical_kernel, clinical_vz, clinical_d1 = Born.Born_weights(clinical_intra,clinical_depth,clinical_radius,clinical_mua0,clinical_musp0)
clinical_r_real = clinical_r[:,:126]
clinical_r_img = clinical_r[:,126:252]
clinical_r_complex = clinical_r_real+1j*clinical_r_img
clinical_t_real = clinical_t[:,:126]
clinical_t_img = clinical_t[:,126:252]
clinical_t_complex = clinical_t_real+1j*clinical_t_img
clinical_pert = (clinical_t_complex-clinical_r_complex[:,:])/clinical_r_complex[:,:]
clinical_pert = np.concatenate((np.real(clinical_pert),np.imag(clinical_pert)),1)
'''

# load patient data UCONN system
p_pert, p_Born, p_image, p_kernel, p_vz, p_d1=PPS.load_patient()

#load shuying data
#p_pert, p_Born, p_image, p_kernel, p_vz, p_d1=PPS.load_shuying()

training_epochs = 100
batch_size = 200
display_step = 1


## raw data of DOT 
pert = tf.compat.v1.placeholder(tf.float32, [None,14*9*2], name='pert')
## Born weights
born_weights = tf.compat.v1.placeholder(tf.float32, [None,16*16*7,14*9*2], name='born_weights')
## ground truth
dmua = tf.compat.v1.placeholder(tf.float32, [None,16*16*7], name='dmua')
## anatomical information
ker = tf.compat.v1.placeholder(tf.float32, [None,16,16,7], name='kernel')


keep_prob = tf.compat.v1.placeholder(tf.float32)



def random_batch(x2_train, y_train,z_train,w_train, batch_size):
    rnd_indices = np.random.randint(0, len(x2_train), batch_size)    
    x2_batch = x2_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    z_batch = z_train[rnd_indices]  
    w_batch = w_train[rnd_indices] 
    return x2_batch, y_batch,z_batch,w_batch




initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
#initializer = tf.constant_initializer(1e-3)
weights1 = tf.Variable(initializer([252,128]), name='weights1')
b1 = tf.Variable(tf.constant(0.1, shape=[128],dtype=tf.float32), name="b1")
weights2 = tf.Variable(initializer([128, 128]), name='weights2')
b2 = tf.Variable(tf.constant(0.1, shape=[128],dtype=tf.float32), name="b2")
weights3 = tf.Variable(initializer([128, 1792]), name='weights3')
b3 = tf.Variable(tf.constant(0.1, shape=[1792],dtype=tf.float32), name="b3")
## pretrain inverse model

l1 = tf.matmul(pert, weights1)+b1
l1 = tf.nn.relu(l1)

l2 = tf.matmul(l1, weights2)+b2
l2 = tf.nn.relu(l2)

img = (tf.matmul(l2, weights3)+b3) #img


img_recon_reshape = tf.reshape(img,[-1,16*16*7,1])
images = tf.reshape(img,[-1,16,16,7])


img_tranpose1 = tf.transpose(tf.transpose(tf.transpose(tf.reshape(img_recon_reshape,(-1,16,16,7)),perm = [0,3,2,1]),perm = [0,2,1,3]),perm = [0,1,3,2])
img_tranpose = tf.reshape(img_tranpose1,(-1,16*16*7,1))

pert_recon = tf.matmul(tf.transpose(born_weights,perm = [0,2,1]), img_tranpose*0.25*0.25*0.5)
pert_recon = tf.reshape(pert_recon,[-1,252])


## encoder forward model
weights11 = tf.Variable(initializer([1792,256]), name='weights11')
b11 = tf.Variable(tf.constant(0.1, shape=[256],dtype=tf.float32), name="b11")
weights12 = tf.Variable(initializer([256, 128]), name='weights12')
b12 = tf.Variable(tf.constant(0.1, shape=[128],dtype=tf.float32), name="b12")
weights13 = tf.Variable(initializer([128, 252]), name='weights13')
b13 = tf.Variable(tf.constant(0.1, shape=[252],dtype=tf.float32), name="b13")


img_r = tf.reshape(dmua,(-1,1792)) 
l3 = tf.matmul(img_r, weights11)+b11
l3 = tf.nn.relu(l3)

l4 = tf.matmul(l3, weights12)+b12
l4 = tf.nn.relu(l4)

m_re = tf.matmul(l4, weights13)+b13

## fine-tune combine model

l1_f = tf.matmul(pert, weights1)+b1
l1_f = tf.nn.relu(l1_f)

l2_f = tf.matmul(l1_f, weights2)+b2
l2_f = tf.nn.relu(l2_f)

img_f = tf.matmul(l2_f, weights3)+b3 #img

img_reshape_f = tf.reshape(img_f,(-1,16,16,7)) 
img_tranpose1_f = tf.transpose(tf.transpose(tf.transpose(tf.reshape(img_reshape_f,(-1,16,16,7)),perm = [0,3,2,1]),perm = [0,2,1,3]),perm = [0,1,3,2])
img_tranpose1_f = tf.reshape(img_tranpose1_f,(-1,16*16*7,1))

pert_recon1 = tf.matmul(tf.transpose(born_weights,perm = [0,2,1]), img_tranpose1_f*0.25*0.25*0.5)
pert_recon1 = tf.reshape(pert_recon1,[-1,252])

images1 = tf.reshape(img_f,[-1,16,16,7])

l3_f = tf.matmul(img_f, weights11)+b11
l3_f = tf.nn.relu(l3_f)

l4_f = tf.matmul(l3_f, weights12)+b12
l4_f = tf.nn.relu(l4_f)

m_re_f = tf.matmul(l4_f, weights13)+b13

reg_l1 = tf.contrib.layers.apply_regularization(tf.contrib.layers.l1_regularizer(1e-6), tf.compat.v1.trainable_variables())
reg_l2 = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), tf.compat.v1.trainable_variables())

cost_mua = tf.reduce_mean(tf.abs(1e0*(img - dmua)))# + 1e0*tf.reduce_mean(tf.abs(1e0*(pert_recon - pert))) + 1e-1*tf.reduce_mean(tf.image.total_variation(images*kernel))

cost_de = tf.reduce_mean(tf.abs(1e0*(m_re - pert)))

cost_finetune = 10e-1*tf.reduce_mean(tf.abs(1e0*(m_re_f - pert)))#+1e-0*tf.reduce_mean(tf.abs(1e0*(pert_recon1 - pert)))
cost_finetune1 = 10e-1*tf.reduce_mean(tf.abs(1e0*(pert_recon1 - pert)))+ 1e-5*tf.reduce_mean(tf.image.total_variation(tf.abs(images1*ker))) + 1e-2*tf.reduce_mean(tf.abs(img_f))
cost_finetune2 = 1e0*tf.reduce_mean(tf.abs(1e0*(pert_recon1 - pert)))


optimizer = tf.compat.v1.train.AdamOptimizer(2e-4).minimize(10e-1*cost_mua)
optimizer_de = tf.compat.v1.train.AdamOptimizer(2e-4).minimize(10e-1*cost_de)
optimizer1 = tf.compat.v1.train.AdamOptimizer(5e-5).minimize(10e-1*cost_finetune)
optimizer2 = tf.compat.v1.train.AdamOptimizer(5e-5).minimize(10e-1*cost_finetune1)
saver = tf.compat.v1.train.Saver(tf.compat.v1.trainable_variables())



train_op = tf.group(optimizer, optimizer_de) #, optimizer_b2
train_op1 = tf.group(optimizer, optimizer_de,optimizer1,optimizer2)
train_op2 = tf.group(optimizer1,optimizer2)



c=[0]*training_epochs
c_val=[0]*training_epochs


with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    saver.restore(sess, 'model_irregular_simu/model.ckpt')
    #total_batch = int(len(p_pert)/batch_size*2)
    total_batch = 1
    cc = 0
    # Training cycle.
    for epoch in range(training_epochs):
    #for epoch in range(3000,training_epochs):
        if epoch < 5000:
        # Loop over all batches
            for i in range(total_batch):
                '''
                meas_train1, batch_image_train1, born_batch1,kernel_batch1 = random_batch(simu_pert,simu_image,simu_Born,simu_kernel,simu_vz, simu_d1,  60)  # max(x) = 1, min(x) = 0
                meas_train2, batch_image_train2, born_batch2,kernel_batch2 = random_batch(simu_pert_ud,simu_image_ud,simu_Born_ud,simu_kernel_ud ,simu_ud_vz, simu_ud_d1,  30)
                meas_train3, batch_image_train3, born_batch3,kernel_batch3 = random_batch(simu_pert_lr,simu_image_lr,simu_Born_lr,simu_kernel_lr,simu_lr_vz, simu_lr_d1,  30)
                meas_train = np.concatenate((meas_train1,meas_train2,meas_train3,phantom_pert))
                batch_image_train = np.concatenate((batch_image_train1,batch_image_train2,batch_image_train3,phantom_image))
                born_batch = np.concatenate((born_batch1,born_batch2,born_batch3,phantom_Born))
                kernel_batch = np.concatenate((kernel_batch1,kernel_batch2,kernel_batch3,phantom_kernel))
                '''
                # Run optimization op (backprop) and cost op (to get loss value)

                _,c[epoch] = sess.run([train_op2,cost_finetune], feed_dict={pert: p_pert,dmua:p_image, born_weights:p_Born,ker:p_kernel, keep_prob:1.0})
                is_train = False
                c_val[epoch] =sess.run(cost_mua, feed_dict={pert: p_pert,dmua:p_image, born_weights:p_Born,ker:p_kernel, keep_prob:1.0})
                is_train = True

            
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "%.9f" %(c[epoch]))
            print("Epoch:", '%04d' % (epoch+1), "val_cost=", "%.9f" %(c_val[epoch]))
 
    print("Optimization Finished!")
    #saver.save(sess,save_path='/home/whitaker/Desktop/DOT_ML/code_20210915/model/model_irregular_simu/model.ckpt')

    is_train = False

    test_image_show = p_image.reshape(p_pert.shape[0],256,7)
    test_image_show1 = p_image.reshape(p_pert.shape[0],256,7)
    out = sess.run([pert_recon1,images1],  feed_dict={pert: p_pert,dmua:p_image, born_weights:p_Born,ker:p_kernel, keep_prob:1.0})
    out1 = sess.run([pert_recon1,images1], feed_dict={pert: p_pert,dmua:p_image, born_weights:p_Born,ker:p_kernel, keep_prob:1.0})

    output = out[1].reshape((-1,256,7))# - np.array(minvalue).reshape(len(minvalue),1,1)
    output1 = out1[1].reshape((-1,256,7))# - np.array(minvalue).reshape(len(minvalue),1,1)
    output_t = output[:,:,5].reshape((-1,16,16))

plt.plot(c[:200])
plt.plot(c_val[:200],color='r')
plt.figure()
plt.plot(c[200:])
plt.plot(c_val[200:],color='r')

#p_compare = np.concatenate((p_pert[0].reshape(1,252),out[0][0].reshape(1,252)),0).transpose()
#io.savemat('p32.mat', {'mua':output})