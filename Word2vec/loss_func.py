import tensorflow as tf

def cross_entropy_loss(inputs, true_w):

  
    A = tf.log(tf.exp(tf.matmul(tf.transpose(true_w), inputs)))
    print("A shape")
    print(A.shape)
    B = tf.log(tf.reduce_sum(tf.exp(tf.matmul(tf.transpose(true_w), inputs)), axis=1, keepdims=True))
    print(B.shape)
    C=tf.subtract(B, A)
    print(C.shape)
    return tf.subtract(B, A)

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
 
    labels=tf.reshape(labels,[-1])
    uo=tf.nn.embedding_lookup(weights,labels)
    uo_bias=tf.nn.embedding_lookup(biases,labels)
    
    neg_samples=tf.nn.embedding_lookup(weights,sample)
    neg_samples_bias=tf.nn.embedding_lookup(biases,sample)
    
    uo_unigram=tf.gather(unigram_prob,labels)
    uo_unigram_neg_samples=tf.gather(unigram_prob,sample)
    i=tf.size(uo_unigram_neg_samples,out_type=float)
    
    Temp1= tf.add(tf.diag_part(tf.matmul(inputs,tf.transpose(uo))),uo_bias)
    Temp2=tf.log((tf.scalar_mul(i,uo_unigram)))
    Temp3=tf.log(tf.add(tf.sigmoid(tf.subtract(Temp1,Temp2)),10**-10))
 
    Temp4 = tf.add((tf.matmul(inputs, tf.transpose(neg_samples))),neg_samples_bias)
    Temp5=tf.log((tf.scalar_mul(i,uo_unigram_neg_samples)))
    Temp6=(tf.sigmoid(tf.subtract(Temp4,Temp5)))
    #print(Temp6)
    Temp7=tf.reduce_sum(tf.log(tf.add(tf.subtract(1.0,Temp6),10**-10)),axis=1)
    #print(Temp7)
    return tf.negative(tf.add(Temp3,Temp7))
    
    
