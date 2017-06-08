'''
Created on 16 may. 2017

@author: morgan
'''

import tensorflow as tf
import numpy as np

import javabridge
import bioformats
import matplotlib.pyplot as plt
import xml.etree.ElementTree
from xml.etree import cElementTree as ElementTree


def hspiral(image,blur):
       
    
    #NX=1506
    #NY=1506
    #NZ=9
    shape=image.shape
    NX=shape[0]
    NY=shape[1]
    NZ=shape[2]
       
    MX=blur.shape[0]
    MY=blur.shape[1]
    MZ=blur.shape[2]
    
    HXX=np.zeros((5,5,5))
    HXX[2,2,1]=1.0
    HXX[2,2,2]=-2.0
    HXX[2,2,3]=1.0
    
    HXY=np.zeros((5,5,5))
    
    HXY[2,1,1]=1.0
    HXY[2,1,3]=-1.0
    HXY[2,3,1]=-1.0
    HXY[2,3,3]=1.0
    
    HXZ=np.zeros((5,5,5))
    
    HXZ[1,2,1]=1.0
    HXZ[1,2,3]=-1.0
    HXZ[3,2,1]=-1.0
    HXZ[3,2,3]=1.0
    
    
    HYY=np.zeros((5,5,5))
    
    HYY[2,1,2]=1.0
    HYY[2,2,2]=-2.0
    HYY[2,3,2]=1.0
    
    HYZ=np.zeros((5,5,5))
    
    HYZ[1,1,2]=1.0
    HYZ[1,3,2]=-1.0
    HYZ[3,1,2]=-1.0
    HYZ[3,3,2]=1.0
    
    HZZ=np.zeros((5,5,5))
    HZZ[1,2,2]=1.0
    HZZ[2,2,2]=-2.0
    HZZ[3,2,2]=1.0
    
    
    HH=np.zeros((5,5,5,1,6),np.float32)
    
    HH[:,:,:,0,0]=HXX
    HH[:,:,:,0,1]=HXY
    HH[:,:,:,0,2]=HXZ
    HH[:,:,:,0,3]=HYY
    HH[:,:,:,0,4]=HYZ
    HH[:,:,:,0,5]=HZZ
    
    blur= blur.reshape((MX,MY,MZ,1,1))
   
   
    image=image.reshape((1,NX,NY,NZ,1))
    
    Y=tf.placeholder(shape=[1,NX,NY,NZ,1],dtype=tf.float32,name="Y")
    HHbank=tf.placeholder(shape=[5,5,5,1,6],dtype=tf.float32,name="HH")
    K=tf.placeholder(shape=[MX,MY,MZ,1,1],dtype=tf.float32,name="K")
    
    alpha_x=tf.constant(1.0)
    
    
    #logX=tf.Variable(np.log(image+0.01),name="X")
    logX=tf.Variable(tf.zeros(shape=[1,NX,NY,NZ,1],dtype=tf.float32),name="X")
    
    #logX=tf.Variable(tf.zeros(shape=[1,XX,YY,ZZ,1],name="mu_X"))
    X=tf.exp(logX)
    
    KX=tf.nn.conv3d(X,K,padding="SAME",strides=[1,1,1,1,1])
    HKX=tf.nn.conv3d(KX,HHbank,strides=[1,1,1,1,1],padding="SAME",name="laplacianH")
    
    HXX=HKX[:,:,:,:,0]
    HXY=HKX[:,:,:,:,1]
    HXZ=HKX[:,:,:,:,2]
    HYY=HKX[:,:,:,:,3]
    HYZ=HKX[:,:,:,:,4]
    HZZ=HKX[:,:,:,:,5]
    
    
    HHH=tf.stack([tf.stack([HXX,HXY,HXZ],axis=4),tf.stack([HXY,HYY,HYZ],axis=4),tf.stack([HXZ,HYZ,HZZ],axis=4)],axis=5)
    
    evals,_ = tf.self_adjoint_eig(HHH)
    
    hsregloss=  -0.5*alpha_x*tf.reduce_sum(tf.abs(evals))
    
    squaredlaplacianregloss=  -0.5*alpha_x*tf.reduce_sum(tf.abs(HXX)+tf.abs(HYY)+tf.abs(HZZ))
    
    poissonloss= tf.reduce_sum(-KX + Y*tf.log(KX) )
        
    obj1 = squaredlaplacianregloss + poissonloss
    
    obj2 = hsregloss + poissonloss
    
    #optim = tf.train.AdagradOptimizer(learning_rate=0.1)
    #optim_opt = optim.minimize(-obj)
    
    optim1 = tf.contrib.opt.ScipyOptimizerInterface(-obj1,options={'maxiter' : 5,'disp' : True },method="L-BFGS-B")
    optim2 = tf.contrib.opt.ScipyOptimizerInterface(-obj2,options={'maxiter' : 100,'disp' : True },method="L-BFGS-B")
    
    
    init_op = tf.global_variables_initializer()
    
    
    
    
    data={Y: image,
          K: blur,
          HHbank : HH
          }
    
    
    with tf.Session() as session:
    
        session.run(init_op)
    
        [Xval] = session.run([X],feed_dict=data)
        optim1.minimize(session,feed_dict=data)
        optim2.minimize(session,feed_dict=data)
        [Xval,evalsvalue] = session.run([X,evals],feed_dict=data)
        
        objness = np.abs(evalsvalue[:,:,:,:,0])+np.abs(evalsvalue[:,:,:,:,1]) +np.abs(evalsvalue[:,:,:,:,2])
        '''
        for it in range(1000):
            [Xval,_,objval,reglossval,poissonlossval]=session.run([X,optim_opt,obj,regloss,poissonloss],feed_dict=data)
            print objval
            print reglossval
            print poissonlossval
        '''
    
        
        bioformats.write_image(pathname='output.ome.tif',pixels=np.reshape(Xval,(XX,YY,ZZ)),pixel_type="float",c=0,t=0,size_c=1,size_z=ZZ,size_t=1)
        bioformats.write_image(pathname='objness.ome.tif',pixels=np.reshape(objness,(XX,YY,ZZ)),pixel_type="float",t=0,size_c=3,size_z=ZZ,size_t=1)
    
    javabridge.kill_vm()
    '''
    
    
    
    
            
    with tf.Session() as session:
        session.run(init_op)
        
        
        
    '''        
    
    '''
    
        
    
    
            
    
    
'''