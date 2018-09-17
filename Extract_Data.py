import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt

from astropy.io import fits

from scipy.optimize import curve_fit

def Gaus_2d((Xa,Ya),x0,y0,fwx,fwy,bgc,amp):
    xterm=((Xa-x0)**2.)/(2.*fwx**2.)
    yterm=((Ya-y0)**2.)/(2.*fwy**2.)
    return (amp*(np.exp(-1.0*(xterm+yterm)))+bgc).ravel()

def Extract_Data(path,savepath,n_obj,fw_g,ap,ver,ex_all):
    locations=np.empty([n_obj,2])*np.nan
    chips=np.empty([n_obj])*np.nan
    for o in range(0,n_obj):
        if o==0:
            l=np.load(savepath+'LOCATION_TARGET.npz')
            locations[0,:]=l['coords']
            chips[0]=l['chip']
        else:
            l=np.load(savepath+'LOCATION_'+str(int(o))+'.npz')
            locations[o,:]=l['coords']
            chips[o]=l['chip']
     
    n_files=0
    for file in os.listdir(path):
        if file.endswith('.fits.gz'):
            n_files+=1
    n_exp=int(n_files/4)
    print 'NUMBER OF EXPOSURES: ', n_exp
    
    #### LOAD DATA ###
    
    filt=np.array([])
    obs_times=np.array([])
    exp_times=np.array([])
    
    data_c1=np.empty([2048,2048,n_exp])*np.nan
    data_c2=np.empty([2048,2048,n_exp])*np.nan
    data_c3=np.empty([2048,2048,n_exp])*np.nan
    data_c4=np.empty([2048,2048,n_exp])*np.nan
    n_exp=0
    for file in os.listdir(path):
        if file.endswith('.fits.gz'):
            split=file.split('c')
            root=file.split('c')[0]
            chip=int(file.split('c')[1].split('.')[0])
            if chip==1:
                n_exp+=0.25
                if ex_all==False:
                    continue
                data_c1[:,:,int(n_exp)]=(fits.open(path+file))[0].data
            if chip==2:
                n_exp+=0.25
                data_c2[:,:,int(n_exp)]=(fits.open(path+file))[0].data
                head=fits.open(path+file)[0].header
                filt=np.append(filt,head['FILTER'])
                obs_times=np.append(obs_times,head['DATE-OBS']+'T'+head['UT-TIME'])
                exp_times=np.append(exp_times,head['EXPTIME'])
                #n_exp+=0.25
                print int(n_exp), head['FILTER'], head['DATE-OBS']+'T'+head['UT-TIME'], head['EXPTIME']
            if chip==3:
                n_exp+=0.25
                if ex_all==False:
                    continue
                data_c3[:,:,int(n_exp)]=(fits.open(path+file))[0].data
                #n_exp+=0.25
            if chip==4:
                n_exp+=0.25
                if ex_all==False:
                    continue
                data_c4[:,:,int(n_exp)]=(fits.open(path+file))[0].data
                #n_exp+=0.25

    #np.savez(savepath+'EXTRACTED_IMAGES.npz',chip1=data_c1,chip2=data_c2, chip3=data_c3,chip4=data_c4)
    Fit_Params=np.empty([n_obj,int(n_exp),6])*np.nan
    counts=np.empty([n_obj,int(n_exp)])*np.nan
    ### FIT GAUSSIANS ###
    fw=fw_g
    for o in range(0,n_obj):
        print '  >>>>>> FITTING FOR OBJECT: ', o
        chip=chips[o]
        for t in range(0,int(n_exp)):
            if t%10==0:
                print '           time ',t
            data_frame=eval('data_c'+str(int(chip)))[:,:,t]
            
            xcent=int(locations[o,1])
            ycent=int(locations[o,0])
            
            A=data_frame[xcent,ycent]
            bg=np.nanmedian(data_frame)
            
            swt=int(fw*ap*5.0)
            
            data_t=data_frame[xcent-swt:xcent+swt,ycent-swt:ycent+swt]
            
            X_arr=np.linspace(xcent-swt,xcent+swt-1,data_t.shape[1])
            Y_arr=np.linspace(ycent-swt,ycent+swt-1,data_t.shape[0])
            X_mat,Y_mat=np.meshgrid(X_arr,Y_arr)
            
            Z_guess=(Gaus_2d((X_mat,Y_mat),xcent,ycent,fw,fw,bg,A)).reshape(data_t.shape[0],data_t.shape[1])
            
            ################### FITTING 2D GAUSSIAN ##############
            p0=np.array([xcent,ycent,fw,fw,bg,A])
            Fit_Params[o,t,:],fit_cov=curve_fit(Gaus_2d,(X_mat,Y_mat),data_t.ravel(),p0=p0)
            x_f=np.int(Fit_Params[o,t,0])
            y_f=np.int(Fit_Params[o,t,1])
            sigx=np.int(Fit_Params[o,t,2])
            sigy=np.int(Fit_Params[o,t,3])
            bg_f=np.int(Fit_Params[o,t,4])
            ap_f=np.int(Fit_Params[o,t,5])
            if ver==True:
                if t%10==0:
                    print '------------------------------------------------------------------------'
                    print '         X    Y    sX    sY    BG       A'
                    print 'GUESS: ', np.int(xcent), ' ',np.int(ycent),' ',np.int(fw),' ',np.int(fw),' ',np.int(bg), ' ',np.int(A)
                    print '  FIT: ', x_f, ' ',y_f,' ', sigx,' ', sigy, ' ',bg_f, ' ',ap_f
                    print '------------------------------------------------------------------------'
            
            Z_fit=(Gaus_2d((X_mat,Y_mat),*Fit_Params[o,t,:])).reshape(data_t.shape[0],data_t.shape[1])
            
            APR=((X_mat-Fit_Params[o,t,0])**2.)/(ap*Fit_Params[o,t,2])**2.+((Y_mat-Fit_Params[o,t,1])**2.)/(ap*Fit_Params[o,t,3])**2.
                 
            APR_MASK=np.empty(data_t.shape)
            for i in range(0,data_t.shape[0]):
                for j in range(0,data_t.shape[1]):
                    if APR[i,j]<=1.0:
                        APR_MASK[i,j]=1.0
                    else:
                        APR_MASK[i,j]=np.nan
            
            residuals_g=data_t-Z_guess
            residuals_f=data_t-Z_fit
            
            counts[o,t]=np.nansum((data_t*APR_MASK)-Fit_Params[o,t,4])
            
                 
            if ver==True:
                if t%10==0:
                    fig,ax=plt.subplots(3,3,figsize=(9,9))#,sharex='col', sharey='row')
                    fig.subplots_adjust(wspace=0, hspace=0)
                    for i in range(0,ax.shape[0]):
                        for j in range(0,ax.shape[1]):
                            if i==0 or i==1:
                                if j!=0:
                                    ax[i,j].set_xticks([])
                                    ax[i,j].set_yticks([])
                            if i==2:
                                if j!=0:
                                    ax[i,j].set_yticks([])

                    l=data_t.shape[0]/2+fw*ap*2.5
                    r=data_t.shape[0]/2-fw*ap*2.5
                    tp=data_t.shape[0]/2-fw*ap*2.5
                    b=data_t.shape[0]/2+fw*ap*2.5
                    for i in range(0,ax.shape[0]):                
                        for j in range(0,ax.shape[1]):
                            ax[i,j].set_ylim(tp,b)
                            ax[i,j].set_xlim(l,r)


                    colors=plt.cm.PuRd_r
                    ax[0,0].imshow(data_t,cmap=colors,vmin=np.nanmedian(data_t)*0.98,vmax=np.nanmedian(data_t)*1.2)
                    ax[0,1].imshow(Z_guess,cmap=colors,vmin=np.nanmedian(data_t)*0.98,vmax=np.nanmedian(data_t)*1.2)
                    ax[0,2].imshow(residuals_g,cmap=colors,vmin=np.nanmin(residuals_g),vmax=np.nanmax(residuals_g))

                    ax[1,0].imshow(data_t-Fit_Params[o,t,4],cmap=colors,vmin=0.0,vmax=np.nanmedian(data_t)*1.2)
                    ax[1,0].contour(APR,levels=[1.0],colors='white',linewidths=2.5)
                    ax[1,1].imshow(Z_fit-Fit_Params[o,t,4],cmap=colors,vmin=0.0,vmax=np.nanmedian(data_t)*1.2)
                    ax[1,1].contour(APR,levels=[1.0],colors='white',linewidths=2.5)
                    ax[1,2].imshow(residuals_f,cmap=colors,vmin=np.nanmin(residuals_f),vmax=np.nanmax(residuals_f))        

                    ax[2,0].imshow(data_t*APR_MASK-Fit_Params[o,t,4],cmap=colors,vmin=0.0,vmax=np.nanmedian(data_t)*1.2)
                    #ax[0].contour(X_matt,Y_matt,APR_t,levels=[1.0],colors='white',linewidths=2.5)
                    ax[2,1].imshow(Z_fit*APR_MASK-Fit_Params[o,t,4],cmap=colors,vmin=0.0,vmax=np.nanmedian(data_t)*1.2)
                    #ax[1].contour(X_matt,Y_matt,APR_t,levels=[1.0],colors='white',linewidths=2.5)
                    ax[2,2].imshow(residuals_f*APR_MASK,cmap=colors,vmin=np.nanmin(residuals_f),vmax=np.nanmax(residuals_f))


                    plt.figtext(0.14,0.14,str(np.round(counts[o,t],2)),fontsize=20,color='darkred')

                    plt.show(block=False)

        np.savez(savepath+'Obj_'+str(int(o))+'_Extracted.npz',
                 params=Fit_Params[o,:,:],counts=counts[o,:], filters=filt, obs_times=obs_times,exp_times=exp_times)
