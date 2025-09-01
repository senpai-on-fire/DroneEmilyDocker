import numpy as np
import pandas as pd
import csv
import os
import pdb
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'  # expose GPU 0

import tensorflow.compat.v1 as tf
import ltc_model as ltc
from ctrnn_model import CTRNN, NODE, CTGRU
import argparse

numID = 1

def cut_in_sequences(x,y,seq_len,inc=1):

    sequences_x = []
    sequences_y = []

    for s in range(0,x.shape[0] - seq_len,inc):
        start = s
        end = start+seq_len
        sequences_x.append(x[start:end])
        sequences_y.append(y[start:end])

    return np.stack(sequences_x,axis=1),np.stack(sequences_y,axis=1)

#@tf.function(input_signature=[tf.TensorSpec(shape=None,dtype=tf.float32)])
#def compare(x):
#    a = np.float32(np.array([10000.0]))
#    stableEps = tf.constant(a)
#    result = tf.math.greater(x,stableEps)
#    return result

Nloop = 0;


class Custom_CE_Loss(tf.keras.losses.Loss):
    def __init__(self,labels,logits,insulin,meal,basal,initIsc1,initIsc2,initIp,BW,u2ss):
        self.y_true2 = labels
        self.y_pred2 = logits
        self.y_ins = insulin/5
        self.y_meal = meal/5
        self.y_basal = basal
        self.y_initIsc1 = initIsc1
        self.y_initIsc2 = initIsc2
        self.y_initIp = initIp
        self.BW = BW
        self.u2ss = u2ss
        #self.yDummy = tf.tile(self.y_pred2,241)

    def lossFV2(self):
        ############################# Parameters ##################################
        maxChange = 95;
        kempt = (1 + (0.5 - self.y_pred2[:, :, 0]) * maxChange / 100) * 0.18
        kabs = (1 + (0.5 - self.y_pred2[:, :, 1]) * maxChange / 100) * 0.012
        #kempt = ((self.y_pred2[:, :, 0]) * maxChange / 100) * 0.18
        #kabs = ((self.y_pred2[:, :, 1]) * maxChange / 100) * 0.012
        f = 0.9
        Gb = (1 + (0.5 - self.y_pred2[:, :, 2]) * maxChange / 100) * 119.13
        SG = (1 + (0.5 - self.y_pred2[:, :, 3]) * maxChange / 100) * 0.025
        #Gb = ((self.y_pred2[:, :, 2]) * maxChange / 100) * 119.13
        #SG = ((self.y_pred2[:, :, 3]) * maxChange / 100) * 0.025        
        Vg = 1.45
        p2 = (1 + (0.5 - self.y_pred2[:, :, 4]) * maxChange / 100) * 0.012
        SI = (1 + (0.5 - self.y_pred2[:, :, 5]) * maxChange / 100) * 0.001035 / Vg
        Ipb = (1 + (0.5 - self.y_pred2[:, :, 6]) * maxChange / 100)
        #p2 = ((self.y_pred2[:, :, 4]) * maxChange / 100) * 0.012
        #SI = ((self.y_pred2[:, :, 5]) * maxChange / 100) * 0.001035 / Vg
        #Ipb = ((self.y_pred2[:, :, 6]) * maxChange / 100)
        alpha = 7
        kd = (1 + (0.5 - self.y_pred2[:, :, 7]) * maxChange / 100) * 0.026
        beta = tf.floor((1 + (0.5 - self.y_pred2[:, :, 9]) * maxChange / 100) * 15)
        #kd = ((self.y_pred2[:, :, 7]) * maxChange / 100) * 0.026
        #beta = tf.floor(((self.y_pred2[:, :, 9]) * maxChange / 100) * 15)
        Vi = 0.126
        ka2 = (1 + (0.5 - self.y_pred2[:, :, 8]) * maxChange / 100) * 0.014
        #ka2 = ((self.y_pred2[:, :, 8]) * maxChange / 100) * 0.014
        ke = 0.127
        bolusD = tf.floor((1 + (0.5 - self.y_pred2[:, :, 10]) * maxChange / 100) * 5)
        #bolusD = tf.floor(((self.y_pred2[:, :, 10]) * maxChange / 100) * 5)
        r2 = 0.8124

        ###########################################################################
        # print("true2 shape: {}".format(self.y_true2))
        GVal = self.y_true2[:, :, 0]
        # print("Gval shape is {}".format(GVal))
        pmVal = tf.zeros(tf.shape(GVal), dtype=tf.dtypes.float32)

        Isc1Val = self.y_basal[:,:,0]/(kd*Vi)

        Isc2Val = kd*tf.math.divide_no_nan(Isc1Val,ka2)

        IpVal = tf.math.multiply(ka2, Isc2Val)/ke

        #Isc1Val = self.y_initIsc1[:, :, 0]/kd 

        #Isc2Val = tf.math.divide_no_nan(self.y_initIsc2[:, :, 0],ka2)

        #IpVal = self.y_initIp[:, :, 0]/ke
        Qsto1Val = tf.zeros(tf.shape(GVal), dtype=tf.dtypes.float32)
        # print("Qsto1Val shape is {}".format(Qsto1Val))
        Ipb = tf.zeros(tf.shape(GVal), dtype=tf.dtypes.float32)
        Qsto2Val = tf.zeros(tf.shape(GVal), dtype=tf.dtypes.float32)
        QgutVal = tf.zeros(tf.shape(GVal), dtype=tf.dtypes.float32)
        RatVal = tf.zeros(tf.shape(GVal), dtype=tf.dtypes.float32)
        insVal = tf.zeros(tf.shape(GVal), dtype=tf.dtypes.float32)
        xVal = tf.zeros(tf.shape(GVal), dtype=tf.dtypes.float32)
        gVal = self.y_true2[:, :, 0]
        y_ins = tf.zeros(tf.shape(GVal), dtype=tf.dtypes.float32)
        meal = tf.zeros(tf.shape(GVal), dtype=tf.dtypes.float32)

        GVal = tf.expand_dims(GVal, 2)
        Isc1Val = tf.expand_dims(Isc1Val, 2)
        Isc2Val = tf.expand_dims(Isc2Val, 2)
        IpVal = tf.expand_dims(IpVal, 2)
        xVal = tf.expand_dims(xVal, 2)
        Qsto1Val = tf.expand_dims(Qsto1Val, 2)
        Qsto2Val = tf.expand_dims(Qsto2Val, 2)
        QgutVal = tf.expand_dims(QgutVal, 2)
        RatVal = tf.expand_dims(RatVal, 2)
        gVal = tf.expand_dims(gVal, 2)
        y_ins = tf.expand_dims(y_ins, 2)
        meal = tf.expand_dims(meal, 2)

        limitLoop = 144
        tau = 5
        a = np.float32(np.array([100000.0]))
        stableEps = tf.constant(a)

        #for i in range(2,bolusD):
        #   dummy_y_ins = tf.zeros(tf.shape(pmVal),dtype=tf.dtypes.float32)
        #   dummy_y_ins = tf.expand_dims(dummy_y_ins,2)
        #   y_ins = tf.concat([y_ins,dummy_y_ins],2)
        #   y_ins = self.y_ins

        #for i in range(bolusD,limitLoop):
        #   dummy_y_ins = self.y_ins[:,:,i-bolusD+1]
        #   dummy_y_ins = tf.expand_dims(dummy_y_ins,2)
        #   y_ins = tf.concat([y_ins, dummy_y_ins], 2)

        #for i in range(2, beta):
        #   dummy_meal = tf.zeros(tf.shape(pmVal), dtype=tf.dtypes.float32)
        #   dummy_meal = tf.expand_dims(dummy_meal, 2)
        #   meal = tf.concat([meal, dummy_y_ins], 2)

        #for i in range(beta, limitLoop):
        #   dummy_meal = self.y_meal[:, :, i - bolusD + 1]
        #   dummy_meal = tf.expand_dims(dummy_meal, 2)
        #   y_meal = tf.concat([y_meal, dummy_meal], 2)
        #pdb.set_trace()
        eps = 25
        #breakpoint()
        for i in range(1, limitLoop, 1):
            ka1 = 0.0
            # print("Ipb is: {}".format(Ipb))
            # kd_reshaped = tf.broadcast_to(kd, tf.shape(self.u2ss))
            #Ipb = tf.math.divide_no_nan(tf.math.multiply(kd / ke, self.u2ss[:, :, i - 1]), kd)
            Ipb = IpVal[:,:,0]
            shape1 = tf.shape(GVal[:, :, i - 1])

            D = tf.where(tf.logical_and(GVal[:, :, i - 1] >= 60.0, GVal[:, :, i - 1] < 119.13), tf.fill(shape1, 1.0),
                         tf.fill(shape1, 0.0))
            E = tf.where(GVal[:, :, i - 1] < 60.0, tf.fill(shape1, 1.0), tf.fill(shape1, 0.0))
            safe_GVal = tf.maximum(GVal, eps)
            risk = 1.0+tf.math.multiply(10 * tf.math.square(
                tf.math.pow(tf.math.log(safe_GVal[:, :, i - 1]) , r2) - np.power(np.log(119.13), r2)),
                                    D) + 10 * np.power(np.power(np.log(60), r2) - np.power(np.log(119.13), r2), 2) * E
            if i == 1:
                risk2 = risk
            risk = tf.abs(risk)    
            
            #risk = tf.debugging.check_numerics(risk, "risk")
            dummyIsc1 = Isc1Val[:, :, i - 1] + tau * (tf.math.add(-tf.math.multiply(kd, Isc1Val[:, :, i - 1]), (
                tf.math.add(self.y_basal[:, :, i - 1], self.y_ins[:, :, i - 1])) / Vi))
            
            #dummyIsc1 = tf.debugging.check_numerics(dummyIsc1, "dummyIsc1")
            # kd_reshaped = tf.broadcast_to(kd, tf.shape(Isc1Val[:, :, 0]))
            # dummyIsc1 = Isc1Val[:, :, i - 1] + tau * (-tf.math.multiply(kd_reshaped, Isc1Val[:, :, i - 1]))
            dummyIsc2 = Isc2Val[:, :, i - 1] + tau * (
                        tf.math.multiply(kd, Isc1Val[:, :, i - 1]) - tf.math.multiply(ka2, Isc2Val[:, :, i - 1]))
            #dummyIsc2 = tf.debugging.check_numerics(dummyIsc2, "dummyIsc2")            
            dummyIp = IpVal[:, :, i - 1] + tau * (tf.math.multiply(ka2, Isc2Val[:, :, i - 1]) - ke * IpVal[:, :, i - 1])
            #dummyIp = tf.debugging.check_numerics(dummyIp, "dummyIp")
            dummyQsto1 = Qsto1Val[:, :, i - 1] + tau * (
                        -tf.math.multiply(kempt, Qsto1Val[:, :, i - 1]) + self.y_meal[:, :, i - 1])
            #dummyQsto1 = tf.debugging.check_numerics(dummyQsto1, "dummyQsto1")            
            #dummyQsto2 = Qsto2Val[:, :, i - 1] + tau * (
            #            tf.math.multiply(kempt, Qsto1Val[:, :, i - 1]) - tf.math.multiply(kempt, Qsto2Val[:, :, i - 1]))


            dummyQsto2 = Qsto2Val[:, :, i - 1] + tau * (
                        tf.math.multiply(kempt, dummyQsto1) - tf.math.multiply(kempt, Qsto2Val[:, :, i - 1]))

            #dummyQsto2 = tf.debugging.check_numerics(dummyQsto2, "dummyQsto2")            
            #dummyQgut = QgutVal[:, :, i - 1] + tau * (
            #            tf.math.multiply(kempt, Qsto2Val[:, :, i - 1]) - tf.math.multiply(kabs, QgutVal[:, :, i - 1]))


            dummyQgut = QgutVal[:, :, i - 1] + tau * (
                        tf.math.multiply(kempt, dummyQsto2) - tf.math.multiply(kabs, dummyQsto2))

            #dummyQgut = tf.debugging.check_numerics(dummyQgut, "dummyQgut")            
            #RatVal = f * tf.math.multiply(kabs, QgutVal[:, :, i - 1])

            RatVal = f * tf.math.multiply(kabs, dummyQgut)


            #print("risk is {}".format(risk))
            dummyXVal = xVal[:, :, i - 1] + tau * (
                        -tf.math.multiply(p2, xVal[:, :, i - 1]) - tf.math.multiply(SI, (IpVal[:, :, i - 1] - Ipb)))
            #dummyXVal = tf.debugging.check_numerics(dummyXVal, "dummyXVal")   
            
            raw_mix = (-(SG + risk * xVal[:, :, i-1]) * gVal[:,:,i-1]+ SG * Gb+ (RatVal / Vg)   )
            raw_dG = gVal[:,:,i-1] + tau * raw_mix
            
            max_inc = 400.0  # max mg/dL per timestep you consider safe
            mask    = tf.logical_and(
                          tf.math.is_finite(raw_dG),
                          tf.abs(raw_dG - gVal[:,:,i-1]) <= max_inc
                      )

            # 3) select either the new value or carry forward the old
            dummygVal = tf.where(mask,
                          raw_dG,   # if safe
                          gVal[:,:,i-1]    # otherwise no change
                         )
             
                     
            #dummygVal = gVal[:, :, i - 1] + tau * (-tf.math.multiply(SG + tf.math.multiply(risk, xVal[:, :, i - 1]),
            #                                                         gVal[:, :, i - 1]) + tf.math.multiply(SG, Gb) + (
            #                                                   RatVal / Vg))
            #dummygVal = tf.debugging.check_numerics(dummygVal, "dummygVal")
            dummyG1 = GVal[:, :, i - 1] + tau * (-(1 / alpha) * (GVal[:, :, i - 1] - gVal[:, :, i - 1]))
            #dummyG1 = tf.debugging.check_numerics(dummyG1, "dummyG1")
            diffDummy = dummyG1 - GVal[:, :, i - 1]
            #sumDiff = tf.math.reduce_max(tf.math.reduce_max(tf.math.square(diffDummy), axis=0), axis=0)  - 300.0
            #dummyG1 = tf.cond(sumDiff > 0.0, lambda: GVal[:, :, i - 1], lambda: dummyG1)
            
            #old_G  = GVal[:, :, i-1]                    # shape [batch, …]
            #raw_dG = old_G + tau * raw_mix              # same shape

            # 2) build a boolean mask of “safe” updates per example
            #    only those with |ΔG| <= 300 pass through
            mask   = tf.abs(diffDummy) <= 3000.0    # shape [batch, …]

            # 3) pick either the new value or keep the old one, per element
            dummyG1  = tf.where(mask,
                              dummyG1,    # if within ±300
                              GVal[:,:,i-1]      # else stick with previous
                             )



            dummyIsc1 = tf.expand_dims(dummyIsc1, 2)
            Isc1Val = tf.concat([Isc1Val, dummyIsc1], 2)
            dummyIsc2 = tf.expand_dims(dummyIsc2, 2)
            Isc2Val = tf.concat([Isc2Val, dummyIsc2], 2)
            dummyIp = tf.expand_dims(dummyIp, 2)
            IpVal = tf.concat([IpVal, dummyIp], 2)
            dummyQsto1 = tf.expand_dims(dummyQsto1, 2)
            Qsto1Val = tf.concat([Qsto1Val, dummyQsto1], 2)
            dummyQsto2 = tf.expand_dims(dummyQsto2, 2)
            Qsto2Val = tf.concat([Qsto2Val, dummyQsto2], 2)
            dummyQgut = tf.expand_dims(dummyQgut, 2)
            QgutVal = tf.concat([QgutVal, dummyQgut], 2)
            dummygVal = tf.expand_dims(dummygVal, 2)
            gVal = tf.concat([gVal, dummygVal], 2)
            dummyXVal = tf.expand_dims(dummyXVal, 2)
            xVal = tf.concat([xVal, dummyXVal], 2)
            dummyG1 = tf.expand_dims(dummyG1, 2)
            GVal = tf.concat([GVal, dummyG1], 2)
            # print("Isc1Val: {}".format(Isc1Val))
            # print("kd: {}".format(kd))
            # print("self.y_basal: {}".format(self.y_basal))
            # print("self.y_ins: {}".format(self.y_ins))
            # print("Vi: {}".format(Vi))

        err = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None])

        # yDummy = tf.reshape(tf.tile(yDummy,[1, 1, 1, 241]),[tf.shape(yDummy)[0],tf.shape(yDummy)[1],tf.shape(yDummy)[2],241])
        # self.yDummy = tf.expand_dims(self.yDummy)
        # xVal = - tf.math.multiply(self.y_pred2,tf.math.square(1-xVal))*0.1 - tf.math.multiply(self.y_pred2,xVal)*0.1
        # yDummy = tf.reshape(tf.tile(self.y_pred2,241),[tf.shape(self.y_pred2)[0],tf.shape(self.y_pred2)[1],tf.shape(self.y_pred2)[2],241])
        # print("yDummy {}".format(tf.shape(yDummy)))

        # for i in range(1,241,1):
        #    xVal2 = tf.concat([xVal2,xVal2[:,:,i-1]-tf.math.multiply(self.yDummy[:,:,0],tf.math.square(1-xVal2[:,:,i-1]))*0.1-tf.math.multiply(self.yDummy[:,:,1],xVal2[:,:,i-1])*0.1],2)

        # with tf.Session() as sess:
        #    sess.run(yDummy)

        # breakpoint()
        # xValD2 = sess.run(xVal)

        # print(xValD2)
        err = 100*tf.math.reduce_sum(tf.math.divide_no_nan(tf.math.abs(self.y_true2[:, :, 0:limitLoop] - GVal),self.y_true2[:,:,0:limitLoop]) / (limitLoop+1), axis=2)
        # err = tf.math.sqrt(err)

        # print("OverErr: {}".format(err))

        #breakpoint()
        # overErr = tf.convert_to_tensor(overErr, dtype=tf.float32)
        # print("overErr: {}".format(err))

        return err
        
    def lossF(self):        
        ############################# Parameters ##################################
        maxChange = 95;
        kempt = (1 + (0.5 - self.y_pred2[:, :, 0]) * maxChange / 100) * 0.18
        kabs = (1 + (0.5 - self.y_pred2[:, :, 1]) * maxChange / 100) * 0.012
        #kempt = ((self.y_pred2[:, :, 0]) * maxChange / 100) * 0.18
        #kabs = ((self.y_pred2[:, :, 1]) * maxChange / 100) * 0.012
        f = 0.9
        Gb = (1 + (0.5 - self.y_pred2[:, :, 2]) * maxChange / 100) * 119.13
        SG = (1 + (0.5 - self.y_pred2[:, :, 3]) * maxChange / 100) * 0.025
        #Gb = ((self.y_pred2[:, :, 2]) * maxChange / 100) * 119.13
        #SG = ((self.y_pred2[:, :, 3]) * maxChange / 100) * 0.025        
        Vg = 1.45
        p2 = (1 + (0.5 - self.y_pred2[:, :, 4]) * maxChange / 100) * 0.012
        SI = (1 + (0.5 - self.y_pred2[:, :, 5]) * maxChange / 100) * 0.001035 / Vg
        Ipb = (1 + (0.5 - self.y_pred2[:, :, 6]) * maxChange / 100)
        #p2 = ((self.y_pred2[:, :, 4]) * maxChange / 100) * 0.012
        #SI = ((self.y_pred2[:, :, 5]) * maxChange / 100) * 0.001035 / Vg
        #Ipb = ((self.y_pred2[:, :, 6]) * maxChange / 100)
        alpha = 7
        kd = (1 + (0.5 - self.y_pred2[:, :, 7]) * maxChange / 100) * 0.026
        beta = tf.floor((1 + (0.5 - self.y_pred2[:, :, 9]) * maxChange / 100) * 15)
        #kd = ((self.y_pred2[:, :, 7]) * maxChange / 100) * 0.026
        #beta = tf.floor(((self.y_pred2[:, :, 9]) * maxChange / 100) * 15)
        Vi = 0.126
        ka2 = (1 + (0.5 - self.y_pred2[:, :, 8]) * maxChange / 100) * 0.014
        #ka2 = ((self.y_pred2[:, :, 8]) * maxChange / 100) * 0.014
        ke = 0.127
        bolusD = tf.floor((1 + (0.5 - self.y_pred2[:, :, 10]) * maxChange / 100) * 5)
        #bolusD = tf.floor(((self.y_pred2[:, :, 10]) * maxChange / 100) * 5)
        r2 = 0.8124


        print("vals {}".format(kempt))
        print("vals {}".format(kabs))
        print("vals {}".format(Gb))
        print("vals {}".format(SG))
        print("vals {}".format(p2))


        ###########################################################################
        #print("true2 shape: {}".format(self.y_true2))
        GVal = self.y_true2[:,:,0]
        #print("Gval shape is {}".format(GVal))
        pmVal = tf.zeros(tf.shape(GVal),dtype=tf.dtypes.float32)
        Isc1Val = self.y_basal[:,:,0]/(kd*Vi)

        Isc2Val = kd*tf.math.divide_no_nan(Isc1Val,ka2)

        IpVal = tf.math.multiply(ka2, Isc2Val)/ke
        Qsto1Val = tf.zeros(tf.shape(GVal),dtype=tf.dtypes.float32)
        #print("Qsto1Val shape is {}".format(Qsto1Val))
        Ipb = tf.zeros(tf.shape(GVal),dtype=tf.dtypes.float32)
        Qsto2Val = tf.zeros(tf.shape(GVal),dtype=tf.dtypes.float32)
        QgutVal = tf.zeros(tf.shape(GVal),dtype=tf.dtypes.float32)
        RatVal = tf.zeros(tf.shape(GVal),dtype=tf.dtypes.float32)
        insVal = tf.zeros(tf.shape(GVal),dtype=tf.dtypes.float32)
        xVal = tf.zeros(tf.shape(GVal),dtype=tf.dtypes.float32)
        gVal = self.y_true2[:,:,0]
        y_ins = tf.zeros(tf.shape(GVal),dtype=tf.dtypes.float32)
        meal = tf.zeros(tf.shape(GVal),dtype=tf.dtypes.float32)


        GVal = tf.expand_dims(GVal,2)
        Isc1Val = tf.expand_dims(Isc1Val,2)
        Isc2Val = tf.expand_dims(Isc2Val,2)
        IpVal = tf.expand_dims(IpVal,2)
        xVal = tf.expand_dims(xVal,2)
        Qsto1Val = tf.expand_dims(Qsto1Val,2)
        Qsto2Val = tf.expand_dims(Qsto2Val, 2)
        QgutVal = tf.expand_dims(QgutVal, 2)
        RatVal = tf.expand_dims(RatVal,2)
        gVal = tf.expand_dims(gVal,2)
        y_ins = tf.expand_dims(y_ins,2)
        meal = tf.expand_dims(meal, 2)

        
        limitLoop = 144
        tau = 5
        a = np.float32(np.array([100000.0]))
        stableEps = tf.constant(a)
        
        

        #for i in range(2,bolusD):
        #    dummy_y_ins = tf.zeros(tf.shape(pmVal),dtype=tf.dtypes.float32)
        #    dummy_y_ins = tf.expand_dims(dummy_y_ins,2)
        #    y_ins = tf.concat([y_ins,dummy_y_ins],2)
        #    y_ins = self.y_ins

        #for i in range(bolusD,limitLoop):
        #    dummy_y_ins = self.y_ins[:,:,i-bolusD+1]
        #    dummy_y_ins = tf.expand_dims(dummy_y_ins,2)
        #    y_ins = tf.concat([y_ins, dummy_y_ins], 2)

        #for i in range(2, beta):
        #    dummy_meal = tf.zeros(tf.shape(pmVal), dtype=tf.dtypes.float32)
        #    dummy_meal = tf.expand_dims(dummy_meal, 2)
        #    meal = tf.concat([meal, dummy_y_ins], 2)

        #for i in range(beta, limitLoop):
        #    dummy_meal = self.y_meal[:, :, i - bolusD + 1]
        #    dummy_meal = tf.expand_dims(dummy_meal, 2)
        #    y_meal = tf.concat([y_meal, dummy_meal], 2)
        #pdb.set_trace()
        eps = 25
        for i in range(1,limitLoop,1):
            ka1 = 0.0
            #print("Ipb is: {}".format(Ipb))
            #kd_reshaped = tf.broadcast_to(kd, tf.shape(self.u2ss))
            Ipb = IpVal[:,:,0]
            shape1 = tf.shape(GVal[:,:,i-1])

            D = tf.where(tf.logical_and(GVal[:,:,i-1] >= 60.0, GVal[:,:,i-1] < 119.13),tf.fill(shape1, 1.0),  tf.fill(shape1, 0.0))
            E = tf.where(GVal[:,:,i-1] < 60.0,tf.fill(shape1, 1.0),tf.fill(shape1, 0.0))
            safe_GVal = tf.maximum(GVal, eps)
            risk = 1.0+tf.math.multiply(10 * tf.math.square(
                tf.math.pow(tf.math.log(safe_GVal[:, :, i - 1]), r2) - np.power(np.log(119.13), r2)),
                                    D) + 10 * np.power(np.power(np.log(60), r2) - np.power(np.log(119.13), r2), 2) * E

            risk = tf.abs(risk)
            #risk = tf.debugging.check_numerics(risk, "risk")
            
            dummyIsc1 = Isc1Val[:, :, i - 1] + tau * (tf.math.add(-tf.math.multiply(kd, Isc1Val[:, :, i - 1]), (
                tf.math.add(self.y_basal[:, :, i - 1], self.y_ins[:, :, i - 1])) / Vi))
            
            #dummyIsc1 = tf.debugging.check_numerics(dummyIsc1, "dummyIsc1")
            # kd_reshaped = tf.broadcast_to(kd, tf.shape(Isc1Val[:, :, 0]))
            # dummyIsc1 = Isc1Val[:, :, i - 1] + tau * (-tf.math.multiply(kd_reshaped, Isc1Val[:, :, i - 1]))
            dummyIsc2 = Isc2Val[:, :, i - 1] + tau * (
                        tf.math.multiply(kd, Isc1Val[:, :, i - 1]) - tf.math.multiply(ka2, Isc2Val[:, :, i - 1]))
            #dummyIsc2 = tf.debugging.check_numerics(dummyIsc2, "dummyIsc2")            
            dummyIp = IpVal[:, :, i - 1] + tau * (tf.math.multiply(ka2, Isc2Val[:, :, i - 1]) - ke * IpVal[:, :, i - 1])
            #dummyIp = tf.debugging.check_numerics(dummyIp, "dummyIp")
            dummyQsto1 = Qsto1Val[:, :, i - 1] + tau * (
                        -tf.math.multiply(kempt, Qsto1Val[:, :, i - 1]) + self.y_meal[:, :, i - 1])
            #dummyQsto1 = tf.debugging.check_numerics(dummyQsto1, "dummyQsto1")            
            #dummyQsto2 = Qsto2Val[:, :, i - 1] + tau * (
            #            tf.math.multiply(kempt, Qsto1Val[:, :, i - 1]) - tf.math.multiply(kempt, Qsto2Val[:, :, i - 1]))
            #dummyQsto2 = tf.debugging.check_numerics(dummyQsto2, "dummyQsto2")            
            #dummyQgut = QgutVal[:, :, i - 1] + tau * (
            #            tf.math.multiply(kempt, Qsto2Val[:, :, i - 1]) - tf.math.multiply(kabs, QgutVal[:, :, i - 1]))
            #dummyQgut = tf.debugging.check_numerics(dummyQgut, "dummyQgut")            
            #RatVal = f * tf.math.multiply(kabs, QgutVal[:, :, i - 1])




            #dummyQsto2 = Qsto2Val[:, :, i - 1] + tau * (
            #            tf.math.multiply(kempt, Qsto1Val[:, :, i - 1]) - tf.math.multiply(kempt, Qsto2Val[:, :, i - 1]))


            dummyQsto2 = Qsto2Val[:, :, i - 1] + tau * (
                        tf.math.multiply(kempt, dummyQsto1) - tf.math.multiply(kempt, Qsto2Val[:, :, i - 1]))

            #dummyQsto2 = tf.debugging.check_numerics(dummyQsto2, "dummyQsto2")
            #dummyQgut = QgutVal[:, :, i - 1] + tau * (
            #            tf.math.multiply(kempt, Qsto2Val[:, :, i - 1]) - tf.math.multiply(kabs, QgutVal[:, :, i - 1]))


            dummyQgut = QgutVal[:, :, i - 1] + tau * (
                        tf.math.multiply(kempt, dummyQsto2) - tf.math.multiply(kabs, dummyQsto2))

            #dummyQgut = tf.debugging.check_numerics(dummyQgut, "dummyQgut")
            #RatVal = f * tf.math.multiply(kabs, QgutVal[:, :, i - 1])

            RatVal = f * tf.math.multiply(kabs, dummyQgut)

            #print("risk is {}".format(risk))
            dummyXVal = xVal[:, :, i - 1] + tau * (
                        -tf.math.multiply(p2, xVal[:, :, i - 1]) - tf.math.multiply(SI, (IpVal[:, :, i - 1] - Ipb)))
            #dummyXVal = tf.debugging.check_numerics(dummyXVal, "dummyXVal")  
            
            
            raw_mix = (-(SG + risk * xVal[:, :, i-1]) * gVal[:,:,i-1]+ SG * Gb+ (RatVal / Vg)   )
            raw_dG = gVal[:,:,i-1] + tau * raw_mix
            
            max_inc = 400.0  # max mg/dL per timestep you consider safe
            mask    = tf.logical_and(
                          tf.math.is_finite(raw_dG),
                          tf.abs(raw_dG - gVal[:,:,i-1]) <= max_inc
                      )

            # 3) select either the new value or carry forward the old
            dummygVal = tf.where(mask,
                          raw_dG,   # if safe
                          gVal[:,:,i-1]    # otherwise no change
                         )

                      
            #dummygVal = gVal[:, :, i - 1] + tau * (-tf.math.multiply(SG + tf.math.multiply(risk, xVal[:, :, i - 1]),
            #                                                         gVal[:, :, i - 1]) + tf.math.multiply(SG, Gb) + (
            #                                                   RatVal / Vg))
            #dummygVal = tf.debugging.check_numerics(dummygVal, "dummygVal")
            
            
            
            dummyG1 = GVal[:, :, i - 1] + tau * (-(1 / alpha) * (GVal[:, :, i - 1] - gVal[:, :, i - 1]))
            #dummyG1 = tf.debugging.check_numerics(dummyG1, "dummyG1")
            diffDummy = dummyG1 - GVal[:,:,i-1]
            #sumDiff = tf.math.reduce_max(tf.math.reduce_max(tf.math.square(diffDummy),axis=0),axis=0)-300.0
            #dummyG1 = tf.cond(sumDiff > 0.0, lambda: GVal[:,:,i-1], lambda: dummyG1)
            mask   = tf.abs(diffDummy) <= 3000.0    # shape [batch, …]

            # 3) pick either the new value or keep the old one, per element
            dummyG1  = tf.where(mask,
                              dummyG1,    # if within ±300
                              GVal[:,:,i-1]      # else stick with previous
                             )
            dummyIsc1 = tf.expand_dims(dummyIsc1,2)
            Isc1Val = tf.concat([Isc1Val,dummyIsc1],2)
            dummyIsc2 = tf.expand_dims(dummyIsc2, 2)
            Isc2Val = tf.concat([Isc2Val, dummyIsc2], 2)
            dummyIp = tf.expand_dims(dummyIp, 2)
            IpVal = tf.concat([IpVal, dummyIp], 2)
            dummyQsto1 = tf.expand_dims(dummyQsto1, 2)
            Qsto1Val = tf.concat([Qsto1Val, dummyQsto1], 2)
            dummyQsto2 = tf.expand_dims(dummyQsto2, 2)
            Qsto2Val = tf.concat([Qsto2Val, dummyQsto2], 2)
            dummyQgut  = tf.expand_dims(dummyQgut, 2)
            QgutVal = tf.concat([QgutVal, dummyQgut], 2)
            dummygVal = tf.expand_dims(dummygVal,2)
            gVal = tf.concat([gVal, dummygVal],2)
            dummyXVal = tf.expand_dims(dummyXVal, 2)
            xVal = tf.concat([xVal, dummyXVal], 2)
            dummyG1 = tf.expand_dims(dummyG1, 2)
            GVal = tf.concat([GVal, dummyG1], 2)
            #print("Isc1Val: {}".format(Isc1Val))
            #print("kd: {}".format(kd))
            #print("self.y_basal: {}".format(self.y_basal))
            #print("self.y_ins: {}".format(self.y_ins)) 
            #print("Vi: {}".format(Vi)) 


        err = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None])
        
        #yDummy = tf.reshape(tf.tile(yDummy,[1, 1, 1, 241]),[tf.shape(yDummy)[0],tf.shape(yDummy)[1],tf.shape(yDummy)[2],241])
        #self.yDummy = tf.expand_dims(self.yDummy)
        #xVal = - tf.math.multiply(self.y_pred2,tf.math.square(1-xVal))*0.1 - tf.math.multiply(self.y_pred2,xVal)*0.1
        #yDummy = tf.reshape(tf.tile(self.y_pred2,241),[tf.shape(self.y_pred2)[0],tf.shape(self.y_pred2)[1],tf.shape(self.y_pred2)[2],241]) 
        #print("yDummy {}".format(tf.shape(yDummy))) 
        
        #for i in range(1,241,1):
        #    xVal2 = tf.concat([xVal2,xVal2[:,:,i-1]-tf.math.multiply(self.yDummy[:,:,0],tf.math.square(1-xVal2[:,:,i-1]))*0.1-tf.math.multiply(self.yDummy[:,:,1],xVal2[:,:,i-1])*0.1],2)
        
        #with tf.Session() as sess:
        #    sess.run(yDummy)
        
        #breakpoint()
        #xValD2 = sess.run(xVal)

        #print(xValD2)
        err =  tf.math.sqrt(tf.math.reduce_sum(tf.math.square(self.y_true2[:,:,0:limitLoop]-GVal)/limitLoop,axis=2))
        #err = tf.math.sqrt(err)
            
        #print("OverErr: {}".format(err))
        
        #breakpoint()
        #overErr = tf.convert_to_tensor(overErr, dtype=tf.float32)
        #print("overErr: {}".format(err))
        
        return err

    

    

        

class HarData:

    def __init__(self,seq_len=16):
        tf.compat.v1.disable_eager_execution()
        tf.disable_v2_behavior()
        print("Parsing for Patient File {}".format(numID))
        train_x = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/P{numID}_glucose.txt")
        train_y = (np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/P{numID}_glucose.txt")-1)#.astype(np.int32)
        train_ins = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/P{numID}_bolus.txt")/100
        train_meal = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/P{numID}_meal.txt")
        train_basal = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/P{numID}_basal.txt")/100
        train_initIsc1 = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/P{numID}_Isc1.txt")
        train_initIsc2 = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/P{numID}_Isc2.txt")
        train_initIp = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/P{numID}_Ip.txt")
        train_BW = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/P{numID}_BW.txt")
        train_u2ss = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/P{numID}_u2ss.txt")
        print("train_x: {}".format(train_x.shape))
        print("train_y: {}".format(train_y.shape))
        print("train_ins: {}".format(train_ins.shape))
        print("train_meal: {}".format(train_meal.shape))
        print("train_basal: {}".format(train_basal.shape))
        print("train_initIsc1: {}".format(train_initIsc1.shape))
        print("train_initIsc2: {}".format(train_initIsc2.shape))
        print("train_initIp: {}".format(train_initIp.shape))
        print("train_BW: {}".format(train_BW.shape))
        print("train_u2ss: {}".format(train_u2ss.shape))
        

        train_meal2 = train_meal

        test_x = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/P{numID}_glucose.txt")
        test_y = (np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/P{numID}_glucose.txt")-1)#.astype(np.int32)
        test_ins = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/P{numID}_bolus.txt")/100
        test_meal = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/P{numID}_meal.txt")
        test_meal2 = test_meal
        test_basal = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/P{numID}_basal.txt")/100
        test_initIsc1 = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/P{numID}_Isc1.txt")
        test_initIsc2 = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/P{numID}_Isc2.txt")
        test_initIp = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/P{numID}_Ip.txt")
        test_BW = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/P{numID}_BW.txt")
        test_u2ss = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/P{numID}_u2ss.txt")
        print("test_x: {}".format(test_x.shape))
        print("test_y: {}".format(test_y.shape))
        print("test_ins: {}".format(test_ins.shape))
        print("test_meal: {}".format(test_meal.shape))
        print("test_basal: {}".format(test_basal.shape))
        print("test_initIsc1: {}".format(test_initIsc1.shape))
        print("test_initIsc2: {}".format(test_initIsc2.shape))
        print("test_initIp: {}".format(test_initIp.shape))
        print("test_BW: {}".format(test_BW.shape))
        print("test_u2ss: {}".format(test_u2ss.shape))
        shape = tf.shape(test_basal)
        with tf.compat.v1.Session() as sess:
            numpy_number = sess.run(shape[1])
        Nloop = numpy_number
        print("Nloop {}".format(Nloop))
        train_x,train_y = cut_in_sequences(train_x,train_y,seq_len)
        train_ins,train_meal = cut_in_sequences(train_ins,train_meal,seq_len)
        train_basal,train_initIsc1 = cut_in_sequences(train_basal,train_initIsc1,seq_len)
        train_initIsc2, train_initIp = cut_in_sequences(train_initIsc2, train_initIp, seq_len)
        train_BW, train_u2ss = cut_in_sequences(train_BW, train_u2ss, seq_len)
        
        
        ## initiate Isc1 and Isc2
        
        
        ##
        

        test_x,test_y = cut_in_sequences(test_x,test_y,seq_len,inc=8)
        test_ins,test_meal = cut_in_sequences(test_ins,test_meal,seq_len,inc=8)

        test_basal, test_initIsc1 = cut_in_sequences(test_basal, test_initIsc1, seq_len,inc=8)
        test_initIsc2, test_initIp = cut_in_sequences(test_initIsc2, test_initIp, seq_len,inc=8)
        test_BW, test_u2ss = cut_in_sequences(test_BW, test_u2ss, seq_len,inc=8)
        print("Total number of testing sequences: {}".format(test_initIsc1.shape[1]))
        #permutation = np.random.RandomState(893429).permutation(train_x.shape[1])
        valid_size = int(0.1*train_x.shape[1])
        print("Validation split: {}, training split: {}".format(valid_size,train_x.shape[1]-valid_size))

        self.valid_x = train_x[:,:valid_size]
        self.valid_ins = train_ins[:,:valid_size]
        self.valid_meal = train_meal[:,:valid_size]
        self.valid_y = train_y[:,:valid_size]
        self.valid_basal = train_basal[:,:valid_size]
        self.valid_initIsc1 = train_initIsc1[:,:valid_size]
        self.valid_initIsc2 = train_initIsc2[:, :valid_size]
        self.valid_initIp = train_initIp[:, :valid_size]
        self.valid_BW = train_BW[:, :valid_size]
        self.valid_u2ss = train_u2ss[:, :valid_size]



        self.train_x = train_x[:,valid_size:]
        self.train_ins = train_ins[:,valid_size:]
        self.train_meal = train_meal[:,valid_size:]
        self.train_y = train_y[:,valid_size:]
        self.train_basal = train_basal[:,valid_size:]
        self.train_initIsc1 = train_initIsc1[:, valid_size:]
        self.train_initIsc2 = train_initIsc2[:, valid_size:]
        self.train_initIp = train_initIp[:, valid_size:]
        self.train_BW = train_BW[:, valid_size:]
        self.train_u2ss = train_u2ss[:, valid_size:]

        self.test_x = test_x
        self.test_ins = test_ins
        self.test_meal = test_meal
        self.test_y = test_y
        self.test_basal = test_basal
        self.test_initIsc1 = test_initIsc1
        self.test_initIsc2 = test_initIsc2
        self.test_initIp = test_initIp
        self.test_BW = test_BW
        self.test_u2ss = test_u2ss

        #pdb.set_trace()

        print("Total number of test sequences: {}".format(self.test_x.shape[1]))

    def iterate_train(self,batch_size=32):
        total_seqs = self.train_x.shape[1]
        permutation = np.random.permutation(total_seqs)
        total_batches = total_seqs // batch_size

        for i in range(total_batches):
            start = i*batch_size
            end = start + batch_size
            batch_x = self.train_x[:,start:end]
            batch_ins = self.train_ins[:,start:end]
            batch_mL = self.train_meal[:,start:end]
            batch_y = self.train_x[:,start:end]
            batch_basal = self.train_basal[:,start:end]
            batch_initIsc1 = self.train_initIsc1[:, start:end]
            batch_initIsc2 = self.train_initIsc2[:, start:end]
            batch_initIp = self.train_initIp[:, start:end]
            batch_BW = self.train_BW[:, start:end]
            batch_u2ss = self.train_u2ss[:, start:end]




            yield (batch_x,batch_y,batch_ins,batch_mL,batch_basal,batch_initIsc1,batch_initIsc2,batch_initIp,batch_BW,batch_u2ss)

class HarModel:

    def __init__(self,model_type,model_size,learning_rate = 0.001):
        self.model_type = model_type
        self.constrain_op = None

        self.x = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,Nloop])
        self.ins = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,Nloop])
        self.mL = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,Nloop])
        self.target_y = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,Nloop])
        self.basal = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,Nloop])
        self.initIsc1 = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,Nloop])
        self.initIsc2 = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,Nloop])
        self.initIp = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,Nloop])
        self.BW = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,Nloop])
        self.u2ss = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,Nloop])

        self.model_size = model_size
        head = self.x



        print("Beginning ")

        if(model_type == "lstm"):
            self.fused_cell = tf.nn.rnn_cell.LSTMCell(model_size)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type.startswith("ltc")):
            learning_rate = 0.01 # LTC needs a higher learning rate
            self.wm = ltc.LTCCell(model_size)
            if(model_type.endswith("_rk")):
                self.wm._solver = ltc.ODESolver.RungeKutta
            elif(model_type.endswith("_ex")):
                self.wm._solver = ltc.ODESolver.Explicit
            else:
                self.wm._solver = ltc.ODESolver.SemiImplicit

            head,_ = tf.nn.dynamic_rnn(self.wm,head,dtype=tf.float32,time_major=True)
            self.constrain_op = self.wm.get_param_constrain_op()
        elif(model_type == "node"):
            self.fused_cell = NODE(model_size,cell_clip=-1)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "ctgru"):
            self.fused_cell = CTGRU(model_size,cell_clip=-1)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "ctrnn"):
            self.fused_cell = CTRNN(model_size,cell_clip=-1,global_feedback=True)
            head,_ = tf.compat.v1.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)

        else:
            raise ValueError("Unknown model type '{}'".format(model_type))


        self.y = tf.compat.v1.layers.Dense(11,activation='sigmoid')(head) # Dense layer output should be same as the number of model parameter
        print("logit shape: ")
        print(str(self.y.shape))
        print("self.y: ")
        print(self.y)
        #self.loss = tf.reduce_mean(evaluate_loss(self.target_y, self.y))
        # Add model estimation error to the loss

        #self.loss = tf.reduce_mean(tf.compat.v1.losses.sparse_softmax_cross_entropy(
        #    labels = self.target_y,
        #    logits = self.y,
        #))
        #print("target_y: {}".format(self.target_y))
        #print("y: {}".format(self.y#))
        #print("ins: {}".format(self.ins))
        #print("meal: {}".format(self.mL))
        #print("basal: {}".format(self.basal))
        #print("initIsc1: {}".format(self.initIsc1))
        #print("initIsc2: {}".format(self.initIsc2))
        #print("Ip: {}".format(self.initIp))
        #print("BW: {}".format(self.BW))
        #print("u2ss: {}".format(self.u2ss))


        #with tf.Session() as sess:
        #    sess.run(self.initIsc1)
        #breakpoint()
        lossVal = Custom_CE_Loss(labels = self.target_y,logits = self.y,insulin = self.ins,meal=self.mL,basal=self.basal,initIsc1=self.initIsc1,initIsc2=self.initIsc2,initIp=self.initIp,BW=self.BW,u2ss=self.u2ss).lossF()
        print("lossVal {}".format(lossVal))


        self.loss = tf.reduce_mean(Custom_CE_Loss(labels = self.target_y,logits = self.y,insulin=self.ins,meal=self.mL,basal=self.basal,initIsc1=self.initIsc1,initIsc2=self.initIsc2,initIp=self.initIp,BW=self.BW,u2ss=self.u2ss).lossF())
        self.lossV2 = tf.reduce_mean(Custom_CE_Loss(labels=self.target_y, logits=self.y, insulin=self.ins, meal=self.mL, basal=self.basal,initIsc1=self.initIsc1, initIsc2=self.initIsc2, initIp=self.initIp, BW=self.BW,u2ss=self.u2ss).lossFV2())

        #self.loss = Custom_CE_Loss()


        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
        
        
        #gvs = optimizer.compute_gradients(self.loss)
        # clip each gradient by global norm of 5.0
        #clipped_gvs = [(tf.clip_by_norm(g, 100.0), v) for g, v in gvs]
        # apply the clipped gradients
        #self.train_step = optimizer.apply_gradients(clipped_gvs)
        
        self.train_step = optimizer.minimize(self.loss)

        model_prediction = tf.argmax(input=self.y, axis=2)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(1.0,tf.int64), tf.cast(1.0,tf.int64)), tf.float32))

        # configure GPU memory growth and device placement


        config = tf.compat.v1.ConfigProto()


        config.gpu_options.allow_growth = True


        config.allow_soft_placement = True


        config.log_device_placement = False


        self.sess = tf.compat.v1.InteractiveSession(config=config)
        self.sess.run(tf.global_variables_initializer())

        self.result_file = os.path.join("results","har","{}_{}.csv".format(model_type,model_size))
        if(not os.path.exists("results/har")):
            os.makedirs("results/har")
        if(not os.path.isfile(self.result_file)):
            with open(self.result_file,"w") as f:
                f.write("best epoch, train loss, train accuracy, valid loss, valid accuracy, test loss, test accuracy\n")

        self.checkpoint_path = os.path.join("tf_sessions","har","{}".format(model_type))
        if(not os.path.exists("tf_sessions/har")):
            os.makedirs("tf_sessions/har")

        self.saver = tf.train.Saver()


    def save(self):
        self.saver.save(self.sess, self.checkpoint_path)

    def restore(self):
        self.saver.restore(self.sess, self.checkpoint_path)


    def fit(self,gesture_data,epochs,verbose=True,log_period=50):

        best_valid_loss = 10000
        best_loss = 10000
        best_valid_stats = (0,0,0,0,0,0,0,0)
        self.save()
        count = 0
        count2 = 0
        count3 = 0
        for e in range(epochs):
            if(e%log_period == 0):
                print("x data {}".format({self.x: gesture_data.test_x}.keys()))
                print("y data {}".format({self.target_y:gesture_data.valid_y}.keys()))

                test_acc,test_loss,test_lossV2 = self.sess.run([self.accuracy,self.loss,self.lossV2],{self.x:gesture_data.test_x,self.target_y: gesture_data.test_x,self.ins: gesture_data.test_ins,self.mL: gesture_data.test_meal,self.basal: gesture_data.test_basal,self.initIsc1: gesture_data.test_initIsc1,self.initIsc2: gesture_data.test_initIsc2,self.initIp: gesture_data.test_initIp,self.BW: gesture_data.test_BW, self.u2ss: gesture_data.test_u2ss})
                valid_acc,valid_loss = self.sess.run([self.accuracy,self.loss],{self.x:gesture_data.valid_x,self.target_y: gesture_data.valid_x, self.ins: gesture_data.valid_ins, self.mL: gesture_data.valid_meal, self.basal: gesture_data.valid_basal,self.initIsc1: gesture_data.valid_initIsc1,self.initIsc2: gesture_data.valid_initIsc2,self.initIp: gesture_data.valid_initIp,self.BW: gesture_data.valid_BW, self.u2ss: gesture_data.valid_u2ss})
                # Accuracy metric -> higher is better
                if(valid_loss < best_valid_loss and e > 0):
                    best_valid_loss = valid_loss
                    best_valid_stats = (
                            e,
                            np.mean(losses),np.mean(accs)*100,
                            valid_loss,valid_acc*100,
                            test_loss,test_lossV2,test_acc*100
                            )
                    self.save()

            losses = []
            mardLoss = []
            accs = []
            tY = []
            err = self.loss 
            print("err {}".format(err))
            #pdb.set_trace()
            #breakpoint()
            for batch_x,batch_y,batch_ins,batch_mL,batch_basal,batch_initIsc1,batch_initIsc2,batch_initIp,batch_BW,batch_u2ss in gesture_data.iterate_train(batch_size=16):
                acc,loss,lossV2,t_step,t_y = self.sess.run([self.accuracy,self.loss,self.lossV2,self.train_step,self.y],{self.x:batch_x,self.target_y: batch_y, self.ins: batch_ins, self.mL: batch_mL, self.basal: batch_basal,self.initIsc1: batch_initIsc1,self.initIsc2: batch_initIsc2,self.initIp: batch_initIp, self.BW: batch_BW,self.u2ss: batch_u2ss})
                
                print("loss iter: {} ".format(loss))
                #breakpoint()
                if(not self.constrain_op is None):
                    self.sess.run(self.constrain_op)
                tY.append(tf.reduce_mean(tf.reduce_mean(t_y,axis=0),axis=0))
                losses.append(loss)
                mardLoss.append(lossV2)
                accs.append(acc)
            losses2 = losses[:-1]    
            mardLosses2 = mardLoss[:-1]
            print("loss: {}".format(sum(losses2)/len(losses2)))

            newLoss = sum(losses2)/len(losses2)

            newMardLoss = sum(mardLosses2)/len(mardLosses2)


            tyMean = tf.reduce_mean(tf.reduce_mean(t_y,axis=0),axis=0)
            tyMean2 = self.sess.run(tyMean)
            maxChange = 95
            kempt = (1+(0.5-tyMean2[0]) * maxChange / 100) * 0.18
            kabs = (1 + (0.5 - tyMean2[1]) * maxChange / 100) * 0.012

            #kempt = ((tyMean2[0]) * maxChange / 100) * 0.18
            #kabs = ((tyMean2[1]) * maxChange / 100) * 0.012

            f = 0.9
            Gb = (1 + (0.5 - tyMean2[2]) * maxChange / 100) * 119.13
            SG = (1 + (0.5 - tyMean2[3]) * maxChange / 100) * 0.025
            #Gb = ((tyMean2[2]) * maxChange / 100) * 119.13
            #SG = ((tyMean2[3]) * maxChange / 100) * 0.025
            Vg = 1.45
            p2 = (1 + (0.5 - tyMean2[4]) * maxChange / 100) * 0.012
            SI = (1 + (0.5 - tyMean2[ 5]) * maxChange / 100) * 0.001035 / Vg
            Ipb = (1 + (0.5 - tyMean2[ 6]) * maxChange / 100)

            #p2 = (( tyMean2[4]) * maxChange / 100) * 0.012
            #SI = (( tyMean2[ 5]) * maxChange / 100) * 0.001035 / Vg
            #Ipb = ((tyMean2[ 6]) * maxChange / 100)

            alpha = 7
            #kd = ((tyMean2[ 7]) * maxChange / 100) * 0.026
            #beta = (( tyMean2[ 9]) * maxChange / 100) * 15
            kd = (1 + (0.5 - tyMean2[ 7]) * maxChange / 100) * 0.026
            beta = (1 + (0.5 - tyMean2[ 9]) * maxChange / 100) * 15
    
            Vi = 0.126
            ka2 = (1 + (0.5 - tyMean2[ 8]) * maxChange / 100) * 0.014

            #ka2 = ((tyMean2[ 8]) * maxChange / 100) * 0.014
            ke = 0.127
            #bolusD = (( tyMean2[ 10]) * maxChange / 100) * 5

            bolusD = (1 + (0.5 - tyMean2[ 10]) * maxChange / 100) * 5


            r2 = 0.8124
            print("Gb {}".format(Gb))

            #breakpoint()

            #### Early stopping
            earlyStop = False
            if earlyStop :
                if best_loss > newLoss :
                    best_loss = newLoss
                    count = 0
                else :
                    count = count + 1
                    if count > 10:
                        break
                if np.abs(best_loss - newLoss) < 0.001*best_loss and best_loss < 2:
                    count2 = count2 + 1
                elif np.abs(best_loss - newLoss) < 0.0005*best_loss and best_loss >= 2 :
                    count3 = count3 + 1
                else:
                    count2 = 0
                if count2 > 5 :
                    break

                if count3 > 10 :
                    break


            ####

                

            if(verbose and e%log_period == 0):
                print("Epochs {:03d}, train loss: {:0.2f}, train accuracy: {:0.2f}%, valid loss: {:0.2f}, valid accuracy: {:0.2f}%, test loss: {:0.2f}, test accuracy: {:0.2f}%".format(
                    e,
                    np.mean(losses),np.mean(accs)*100,
                    valid_loss,valid_acc*100,
                    test_loss,test_acc*100
                ))
                
               
                
            if(e > 0 and (not np.isfinite(np.mean(losses)))):
                break
        self.restore()
        best_epoch,train_loss,train_acc,valid_loss,valid_acc,test_loss,test_lossV2,test_acc = best_valid_stats
        print("Best epoch {:03d}, train loss: {:0.2f}, train accuracy: {:0.2f}%, valid loss: {:0.2f}, valid accuracy: {:0.2f}%, test loss: {:0.2f}, test MARD: {:0.2f}, test accuracy: {:0.2f}%".format(
            best_epoch,
            train_loss,train_acc,
            valid_loss,valid_acc,
            test_loss,test_lossV2,test_acc
        ))
        maxChange = 95
        kempt = (1+(0.5-tyMean2[0]) * maxChange / 100) * 0.18
        kabs = (1 + (0.5 - tyMean2[1]) * maxChange / 100) * 0.012
        
        #kempt = ((tyMean2[0]) * maxChange / 100) * 0.18
        #kabs = ((tyMean2[1]) * maxChange / 100) * 0.012
        
        f = 0.9
        Gb = (1 + (0.5 - tyMean2[2]) * maxChange / 100) * 119.13
        SG = (1 + (0.5 - tyMean2[3]) * maxChange / 100) * 0.025
        #Gb = ((tyMean2[2]) * maxChange / 100) * 119.13
        #SG = ((tyMean2[3]) * maxChange / 100) * 0.025
        Vg = 1.45
        p2 = (1 + (0.5 - tyMean2[4]) * maxChange / 100) * 0.012
        SI = (1 + (0.5 - tyMean2[ 5]) * maxChange / 100) * 0.001035 / Vg
        Ipb = (1 + (0.5 - tyMean2[ 6]) * maxChange / 100)
        
        #p2 = (( tyMean2[4]) * maxChange / 100) * 0.012
        #SI = (( tyMean2[ 5]) * maxChange / 100) * 0.001035 / Vg
        #Ipb = ((tyMean2[ 6]) * maxChange / 100)
        
        alpha = 7
        #kd = ((tyMean2[ 7]) * maxChange / 100) * 0.026
        #beta = (( tyMean2[ 9]) * maxChange / 100) * 15
        kd = (1 + (0.5 - tyMean2[ 7]) * maxChange / 100) * 0.026
        beta = (1 + (0.5 - tyMean2[ 9]) * maxChange / 100) * 15

        Vi = 0.126
        ka2 = (1 + (0.5 - tyMean2[ 8]) * maxChange / 100) * 0.014

        #ka2 = ((tyMean2[ 8]) * maxChange / 100) * 0.014
        ke = 0.127
        #bolusD = (( tyMean2[ 10]) * maxChange / 100) * 5

        bolusD = (1 + (0.5 - tyMean2[ 10]) * maxChange / 100) * 5


        r2 = 0.8124
        print("kempt {}".format(kempt))

        with open(self.result_file,"a") as f:
            f.write("{:03d}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}\n".format(
                best_epoch,
                train_loss, train_acc,
                valid_loss, valid_acc,
                test_loss, test_acc
        ))

        file_path = 'outputV2GPUV3.csv'

        # Define the column labels and values
        columns = ['numID', 'test_loss', 'test_mard', 'kempt', 'kabs','f', 'Gb', 'SG','Vg', 'p2', 'SI', 'Ipb', 'alpha', 'kd', 'beta', 'Vi', 'ka2', 'ke', 'bolusD', 'r2']
        values = [numID, test_loss, test_lossV2, kempt, kabs, f, Gb, SG, Vg, p2, SI, Ipb, alpha, kd, beta, Vi, ka2, ke, bolusD, r2]  # Replace with your float values
        if test_loss < 40:
            # Check if the file exists
            file_exists = os.path.exists(file_path)

            # Open the file in append mode if it exists, or write mode if it doesn't
            with open(file_path, mode='a' if file_exists else 'w', newline='') as file:
                writer = csv.writer(file)

                # If the file doesn't exist, write the header
                if not file_exists:
                    writer.writerow(columns)

                # Write the new row of values
                writer.writerow(values)

            print(f"Values added to {'existing' if file_exists else 'new'} CSV file.")


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default="ltc")
    parser.add_argument('--log',default=1,type=int)
    parser.add_argument('--size',default=32,type=int)
    parser.add_argument('--epochs',default=80,type=int)
    parser.add_argument('--id',default=1,type=int)
    args = parser.parse_args()

    numID = args.id

    test_x = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/P{numID}_glucose.txt")
    Nloop = test_x.shape[1] # convert to 1 if batchsize > 1
    print("Nloop main {}".format(Nloop))

    har_data = HarData()
    model = HarModel(model_type = args.model,model_size=args.size)

    model.fit(har_data,epochs=args.epochs,log_period=args.log)

    print(model.y)

    #breakpoint()

