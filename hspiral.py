'''
Created on 16 may. 2017

@author: morgan
'''

import tensorflow as tf
import numpy as np

class HSPIRAL:
    
    def __init__(self,imageshape,blurshape):
          
        #NX=1506
        #NY=1506
        #NZ=9
    
        self.NX=imageshape[0]
        self.NY=imageshape[1]
        self.NZ=imageshape[2]
       
        self.MX=blurshape[0]
        self.MY=blurshape[1]
        self.MZ=blurshape[2]
    
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
    
    
        self.HH=np.zeros((5,5,5,1,6),np.float32)
        
        self.HH[:,:,:,0,0]=HXX
        self.HH[:,:,:,0,1]=HXY
        self.HH[:,:,:,0,2]=HXZ
        self.HH[:,:,:,0,3]=HYY
        self.HH[:,:,:,0,4]=HYZ
        self.HH[:,:,:,0,5]=HZZ
        
        
    
        self.Y=tf.placeholder(shape=[1,self.NX,self.NY,self.NZ,1],dtype=tf.float32,name="Y")
        self.HHbank=tf.placeholder(shape=[5,5,5,1,6],dtype=tf.float32,name="HH")
        self.K=tf.placeholder(shape=[self.MX,self.MY,self.MZ,1,1],dtype=tf.float32,name="K")
    
        alpha_x=tf.constant(2e-4)
    
    
        #logX=tf.Variable(np.log(image+0.01),name="X")
        logX=tf.Variable(tf.zeros(shape=[1,self.NX,self.NY,self.NZ,1],dtype=tf.float32),name="X")
    
        #logX=tf.Variable(tf.zeros(shape=[1,XX,YY,ZZ,1],name="mu_X"))
        self.X=tf.exp(logX)
    
        KX=tf.nn.conv3d(self.X,self.K,padding="SAME",strides=[1,1,1,1,1])
        HKX=tf.nn.conv3d(KX,self.HHbank,strides=[1,1,1,1,1],padding="SAME",name="laplacianH")
    
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
    
        poissonloss= tf.reduce_sum(-KX + self.Y*tf.log(KX) )
        
        obj1 = squaredlaplacianregloss + poissonloss
    
        obj2 = hsregloss + poissonloss
    
        #optim = tf.train.AdagradOptimizer(learning_rate=0.1)
        #optim_opt = optim.minimize(-obj)
    
        self.optim1 = tf.contrib.opt.ScipyOptimizerInterface(-obj1,options={'maxiter' : 5,'disp' : True },method="L-BFGS-B")
        self.optim2 = tf.contrib.opt.ScipyOptimizerInterface(-obj2,options={'maxiter' : 100,'disp' : True },method="L-BFGS-B")
    
    
        self.init_op = tf.global_variables_initializer()
           
        
    def run(self,image,blur,session):

        blur= blur.reshape((self.MX,self.MY,self.MZ,1,1))
        image=image.reshape((1,self.NX,self.NY,self.NZ,1))
        
        data={self.Y: image,
              self.K: blur,
              self.HHbank : self.HH
              }
    
    
        
    
        session.run(self.init_op)
    
        self.optim1.minimize(session,feed_dict=data)
        self.optim2.minimize(session,feed_dict=data)
        [Xval] = session.run([self.X],feed_dict=data)
            
        return Xval.reshape((self.NX,self.NY,self.NZ))
