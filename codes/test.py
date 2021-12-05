import tensorflow as tf
import os
import numpy as np
a0 = [[[10,10,10,10],[20,20,20,20],[30,30,30,30]],[[40,40,40,40],[4,4,4,4],[41,41,41,41]]]
a = np.array(a0)
a0_mask = [[1,1,0],[1,0,0]]
trg_mask = np.reshape(a0_mask,[2,3,1])
print(a0)
print(a)
print(a0_mask)
print(trg_mask)
len_gen =[2,1]
a_mask = a * trg_mask
a_mask1 = np.reshape(np.sum(a_mask, 1),[2,4])# shape: [batch_size,1, 700]   
print(a_mask)
print(a_mask1)
avg = np.transpose(np.divide(np.transpose(a_mask1),len_gen ))# shape: [batch_size, 700] 
print(avg)
input()
os.environ["CUDA_VISIBLE_DEVICES"]="0"

input0 = tf.constant([[[1, 2], [3,3]], [[4, 5], [6,6]], [[7, 8], [9,9]]])
input1 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
input2 = tf.constant([[[2]], [[2]], [[2]]])
#input3 = tf.constant([[2,2], [2,2], [2,2]])
input4 = tf.constant([[2,2], [20,20], [200,200]])

input5 = tf.constant([[1,1,1]])
input3= tf.constant([[[2,2]],[[20,20]],[[200,200]]])
input33= tf.constant([[[1],[0]],[[0],[1]],[[0],[0]]])
input34= tf.constant([[1,1],[0,1],[0,0]])
a=[input1,input3]
#res= input5+input4
ii= [[1,0,0,0,0],[1,1,1,1,1],[1,1,0,0,0]]
jj= [[2,2,2,2,2],[2,2,2,2,2],[2,2,2,2,2]]

#res= input4 *[[1,0,0],[1,1,1],[1,1,0]]
#res2 = input4*[[1],[0],[2]]
init_state_g = tf.ones([2,3, 4])
init_state_g_2 = tf.ones([2,1, 4])

res4 = tf.tile(input2, [1,1,1])
res5 = tf.tile(input2, [1,2,1])
res6 = tf.concat(input1,1)
res7 = tf.expand_dims(input1,1)
#res8 = tf.expand_dims(input3,1)
#res9 = tf.concat([res7,res8],1)
#res10 =init_state_g +init_state_g_2
#b =tf.convert_to_tensor(a)
#c = tf.reshape(b,[3,2,3])
de = tf.transpose(input0,  perm = [1, 0, 2])
elems = (np.array([1, 2, 3]), np.array([-1, 1, -1]))
alternate = tf.map_fn(lambda x: x[0] * x[1], elems, dtype=tf.int64)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
	#print(sess.run(tf.reshape(i, [4,3])))

	print(sess.run(tf.nn.softmax([0,0.5,5])))

	'''
	print('tf.math.divide(input4, [3,2,5]))', sess.run(tf.transpose(tf.math.divide(tf.transpose(input4), [3,2,5]))))
	print('tf.math.divide(input3, [3,2,1,4])', sess.run(tf.math.divide(input3, [3,2])))

	print('reduce_sum0', sess.run(tf.reduce_sum(input3,0)))
	print('reduce_sum1', sess.run(tf.reduce_sum(input3,1)))
	print('input3*input33',sess.run(input3*input33))
	print('input3*input34',sess.run(input3*input33))
	print(sess.run( input4*tf.reshape([[1],[0],[2]],[-1])))
	print(sess.run(res2))
	print(sess.run(res))
	print(sess.run(res2))
	print('flatten',sess.run( tf.reshape([[1],[0],[2]],[-1])))
	print(sess.run( input4*tf.reshape([[1],[0],[2]],[-1])))
	w_kol=[]
	w_kol2=[]
	for i in range(3):
		w=[]
		w_kol2.append([1.0] * (2) + [0.0] * (5-2))
		for i in range(5):
		
			w.append([1.0])
		for i in range (20-5):
			w.append([0])
		w_kol.append(w)
	print(w_kol2)
	x= tf.reshape(w_kol2, [3,5])
	x1= tf.reshape(ii, [3,5])
	x2= tf.reshape(jj, [3,5])
	print(sess.run(x))
	print(x.shape)
	print(sess.run(x2*x1))
	print(sess.run(x1*x2))
	


	print(sess.run(tf.reshape(w_kol,[3, 20])))
	print(sess.run(tf.reshape(w_kol,[3, 20,1])))

	print(init_state_g.shape)
	print(sess.run(init_state_g))
	print(init_state_g_2.shape)
	print(sess.run(init_state_g_2))
	print(res10.shape)
	print(sess.run(res10))
	print(sess.run(init_state_g_2 +init_state_g))

	#print(sess.run(res2))
	print(sess.run(res4))
	print(111111111111111111111111,input2.shape)
	print(sess.run(res5))
	print(111111111111111111111111,res4.shape)
	print(111111111111111111111111,res5.shape)
	print(111111111111111111111111,input1.shape)
	print(111111111111111111111111,input3.shape)
	print(sess.run(input1[:,0]))
	print(input3[:,0].shape)
	print(sess.run(res6))
	print(res6.shape)


	#print(sess.run(res7))
	#print(res7.shape)# 1,3,3

	#print(sess.run(res8))
	#print(res8.shape)#3,1,3
	print(sess.run(res9))
	print(res9.shape)
	print('a',type(a),a)
	print('b',type(b),b)
	print(sess.run(b))
	print(999999999999999,sess.run(c), type (c))
	print(sess.run(input0), input0.shape)
	print(sess.run(de))
	print(de.shape)
	print(sess.run(alternate))
	print(sess.run(input0))
	print(input0.shape)
	print(sess.run(tf.gather(input0[:,0,:],0)))
	'''








