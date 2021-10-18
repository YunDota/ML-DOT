# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:23:22 2019

@author: 邹运
"""
import numpy as np
import math
import csv
import matplotlib.pyplot as plt


def Born_weights(ref, depth, radius, mua, musp):
    weights_b = []

    kernel = []
    vz_all = []
    depth_1_all = []
    for n in range(ref.shape[0]):
        ref_ref = ref[n]
        radius_t = radius[n]
        depth_t = depth[n]
        mua0 = mua[n]
        musp0 = musp[n]
        
        if radius_t<=0.51:
            depth_1 = depth_t
            depth_0 = depth_1
            while depth_0 > 0.51:
                depth_0=depth_0-0.5
            if depth_0<=0:
                depth_0 = depth_0+0.5
        elif radius_t<=0.71:
            depth_1 = depth_t - 0.25
            depth_0 = depth_1
            while depth_0 > 0.51:
                depth_0=depth_0-0.5
            if depth_0<=0:
                depth_0 = depth_0+0.5
        elif radius_t<=1.1:
            depth_1 = depth_t - 0.5
            depth_0 = depth_1
            while depth_0 > 0.51:
                depth_0=depth_0-0.5
            if depth_0<=0:
                depth_0 = depth_0+0.5    
        elif radius_t<=1.24:
            depth_1 = depth_t - 0.75
            depth_0 = depth_1
            while depth_0 > 0.51:
                depth_0=depth_0-0.5
            if depth_0<=0:
                depth_0 = depth_0+0.5  
        else :
            depth_1 = depth_t - 1
            depth_0 = depth_1
            while depth_0 > 0.51:
                depth_0=depth_0-0.5
            if depth_0<=0:
                depth_0 = depth_0+0.5                  
        vel_c=3e10
        freq=140
        n_ref=1.33333
        vel = vel_c/n_ref
        omega = 2.0*math.pi*freq*1e6
        
        
        #weights = np.zeros([768,126]).astype('complex')
        sx=np.array([[2.794,1.397,0,-1.397,-2.794,2.096,0,-2.096,0]])
        sy=np.array([[1.408,1.408,1.408,1.408,1.408,2.678,2.678,2.678,4.012]])
        sz=np.zeros([1,9])
        
        dx=np.array([[-1.746,0,1.746,2.667,1.016,0,-0.508,-2.667,0.508,1.524,2.54,-1.016,-1.524,-2.54]])
        dy=np.array([[-3.267,-3.581,-3.267,-2.688,-2.569,-2.569,-1.68,-2.688,-1.68,-1.68,-1.68,-2.569,-1.68,-1.68]])
        dz=np.zeros([1,14])
        
        sd_dist=((sx.T-dx)**2+(sy.T-dy)**2)**0.5
        
        vx=np.array([[-1.875,-1.625,-1.375,-1.125,-0.875,-0.625,-0.375,-0.125,0.125,0.375,0.625,0.875,1.125,1.375,1.625,1.875]*16*7])
        vy=np.array([([-1.875]*16+[-1.625]*16+[-1.375]*16+[-1.125]*16+[-0.875]*16+[-0.625]*16+[-0.375]*16+[-0.125]*16+[0.125]*16
                     +[0.375]*16+[0.625]*16+[0.875]*16+[1.125]*16+[1.375]*16+[1.625]*16+[1.875]*16)*7])
        vz=np.array([[depth_0]*256+[depth_0+0.5]*256+[depth_0+1.0]*256+[depth_0+1.5]*256+[depth_0+2]*256+[depth_0+2.5]*256+[depth_0+3]*256])
        
        ref_r = ref_ref[:126]
        ref_i = ref_ref[126:]
        ref_complex = ref_r+1j*ref_i
        ki = np.polyfit(sd_dist.reshape(14*9),np.log((sd_dist**2*np.abs(ref_complex.reshape(9,14))).reshape(14*9)),1)[0]
        kr = np.polyfit(sd_dist.reshape(14*9),np.unwrap(np.angle(ref_complex.reshape(14*9))),1)[0]
        
        tar_x = np.zeros((1,1792))
        tar_y = np.zeros((1,1792))
        tar_z = np.zeros((1,1792))+depth_t
        
        
        
        finemesh = vz-tar_z
        #finemesh=np.where(finemesh<radius_t,finemesh,finemesh*10)
        
        r_vt = ((vx-tar_x)**2+(vy-tar_y)**2+(finemesh**2))**0.4
        ones = np.ones_like(r_vt)
        r_vt=np.where((r_vt<radius_t*1)*(np.abs(finemesh)<radius_t),r_vt/2,(r_vt)**1)
        r_vt=np.where((vz<depth_1),ones,r_vt)
        r_vt = r_vt.reshape(16,16,7,order='F')
        
        g_kernel = ((math.pi*radius_t*np.exp(r_vt))/1e1)**2
        
        
        #g_kernel=np.where((vz<depth_1),ones,g_kernel)
        #g_kernel[g_kernel>10]=0

        kernel.append(g_kernel)
        
        
        D = omega/2/ki/kr/vel
        #mua0 = (-ki**2+kr**2)*D
        #musp0 = -1/3/D
        
        
        ztr = 1/(musp0+mua0)
        D0=ztr/3
        ikw = 1j*np.sqrt((-mua0+1j*omega/vel)/D0)
        Refl =0
        zb= 2.0*ztr*(1+Refl)/(3.0*(1.0-Refl))
        
        sz_1 = sz+ztr
        sz_imag=-sz_1-2*zb
        vz_imag = -vz-2*zb
        
        r_sd = ((sx.T-dx)**2+(sy.T-dy)**2+(sz_1.T-dz)**2)**0.5
        r_isd=((sx.T-dx)**2+(sy.T-dy)**2+(sz_imag.T-dz)**2)**0.5
        
        r_sv = ((sx.T-vx)**2+(sy.T-vy)**2+(sz_1.T-vz)**2)**0.5
        r_vd = ((vx.T-dx)**2+(vy.T-dy)**2+(vz.T-dz)**2)**0.5
        r_isv = ((sx.T-vx)**2+(sy.T-vy)**2+(sz_imag.T-vz)**2)**0.5
        r_ivd = ((vx.T-dx)**2+(vy.T-dy)**2+(vz_imag.T-dz)**2)**0.5
        
        abs_factr = -1/D0
        
        Uo_d1 = np.exp(ikw*r_sd)/r_sd
        Uo_d1 = Uo_d1-np.exp(ikw*r_isd)/r_isd
        Uo_d1 = Uo_d1/(4*math.pi*D0)
        
        Uo_v = np.exp(ikw*r_sv)/r_sv
        Green = np.exp(ikw*r_vd)/r_vd
        Uoi_v = np.exp(ikw*r_isv)/r_isv
        Green_i =np.exp(ikw*r_ivd)/r_ivd
        Uo_v = Uo_v - Uoi_v
        Green = Green - Green_i#semi_inf Green function
        Green = Green/(4*math.pi)
        Uo_v=Uo_v/(4*math.pi*D0)
        
        #Uo_v = np.repeat(Uo_v[:,np.newaxis,:], 14, axis=1)
        
        #Green = np.repeat(Green.T[np.newaxis,:,:], 9, axis=0)
        weight_a = np.zeros((256*7,9,14)).astype('complex')
        for ss in range(9):
            for dd in range(14):
                for vv in range(256*7):
                    weight_a[vv,ss,dd]=abs_factr*Uo_v[ss,vv]*Green[vv,dd]/Uo_d1[ss,dd]
        weight_reshape = weight_a.reshape(16*16*7,14*9)
        
        weight_reshape1 = np.concatenate((np.real(weight_reshape),np.imag((weight_reshape))),1)
        weights_b.append(weight_reshape1)
        vz[vz<(depth_t-radius_t)]=0
        vz[vz>(depth_t+radius_t)]=0
        vz_all.append(vz)
        depth_1_all.append(depth_1)
        #mua.append(mua0)
        #musp.append(musp0)
    return np.array(weights_b),np.array(kernel),np.array(vz_all).reshape(-1,16,16,7,order='F').reshape(-1,1792),np.array(depth_1_all).reshape(-1,1)#.reshape(-1,252,16,16,5,order='F')
        #weight_a = (abs_factr*Uo_v*Green/Uo_d1[:,:,np.newaxis]).reshape([9*14,768])
        #aaaa =abs(weight_a.reshape(126,768))[11,256:512].reshape((16,16))
'''
for v in range(768):
    for sd in range(126):
        s1=((((v%256%16)*0.25-1.875 - sx[int(sd%9)])**2 + (int(v%256/16)*0.25-1.875 - sy[int(sd%9)])**2 + \
            (int(v/256)*0.5+0.25)**2)**0.5)
        d1=((((v%256%16)*0.25-1.875 - dx[int(sd/9)])**2 + (int(v%256/16)*0.25-1.875 \
             - dy[int(sd/9)])**2 + (int(v/256)*0.5+0.25)**2)**0.5)
        weights[v,sd] = np.exp(-(s1+d1)*1j)/s1/d1
weights1 = np.concatenate((weights.real,weights.imag),axis=1)
weights = np.transpose(weights1).astype('float32')
'''