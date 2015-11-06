"""
Written by Jacob N. McLane
Last Modified Nov. 6th, 2015

Basic MCMC fitter for RV measurements of HD 209458.
"""
from math import *
import numpy as np
import random as rand
import scipy.stats as stat
import time
import sys
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

start_time = time.time() #Timer to time program
#Initial Guesses for Parameters

m=100
p=1
off=2451883.7197
########################

mf=[]
pf=[]
of=[]
af=[]
G=6.674e-11
Ms=1.13*1.989e30

########################
#Sine function, assumed to fit data
def s(x,m,p,off):
    y=m*sin((2*pi)/p*(x-off))
    return y

sine=np.vectorize(s)

########################
#Gibbs Sampler
def step(m,p,off):
    m_new=m
    p_new=p
    off_new=off
    for i in range(100):
        m_new=m_new+rand.uniform(-0.1,0.1)
        if m_new < 0:
            m_new=0
        p_new=p_new+rand.uniform(-1,1)
        if p_new < 0:
            p_new=0.1
        off_new=off_new+rand.uniform(-0.1,0.1)
        if off_new < 0:
            off_new=0
    return m_new, p_new, off_new

########################
#Find acceptance probability

def accept(RV_old, RV_new, RV_obs):
    X=sum((RV_old-RV_obs)**2/abs(RV_obs))
    Xp=sum((RV_new-RV_obs)**2/abs(RV_obs))
    Beta=X-Xp
    if Beta > 100:
        Beta=100
    a=exp(Beta/2)
    if a > 1:
        a=1
    return a

########################
#Perform burn fits

def burn(itt, m, p, off, RV_old, RV_obs):
    for i in range(itt):
        m_new, p_new, off_new = step(m,p,off)
        RV_new=sine(JD,m_new,p,off)
        a=accept(RV_old, RV_new, RV_obs)
        b=rand.uniform(0,1)
        if b <= a:
            m=m_new
            RV_old=RV_new
        RV_new=sine(JD,m,p_new,off)
        a=accept(RV_old, RV_new, RV_obs)
        b=rand.uniform(0,1)
        if b <= a:
            p=p_new
            RV_old=RV_new
        RV_new=sine(JD,m,p,off_new)
        a=accept(RV_old, RV_new, RV_obs)
        b=rand.uniform(0,1)
        if b <= a:
            off=off_new
            RV_old=RV_new
        if (i+1)%1000 == 0:
            print('Finished burn step '+str(int(i+1))+'.')
    return m, p, off, RV_old

########################
#Perform MCMC fit
def mcmc(itt, m, p, off, RV_old, RV_obs, rate, cmin,mmin,pmin,omin):
    for i in range(itt):
        m_new, p_new, off_new = step(m,p,off)
        RV_new=sine(JD,m_new,p,off)
        a=accept(RV_old, RV_new, RV_obs)
        c=Xp=np.std(RV_new-RV_obs)
        if c < cmin:
            cmin=c
            mmin=m_new
            pmin=p
            omin=off
        b=rand.uniform(0,1)
        if b <= a:
            m=m_new
            RV_old=RV_new
            rate += 1
            mf.append(m)
            pf.append(p)
            of.append(off)
            af.append(a)
        RV_new=sine(JD,m,p_new,off)
        a=accept(RV_old, RV_new, RV_obs)
        c=Xp=np.std(RV_new-RV_obs)
        if c < cmin:
            cmin=c
            mmin=m
            pmin=p_new
            omin=off
        b=rand.uniform(0,1)
        if b <= a:
            p=p_new
            RV_old=RV_new
            rate += 1
            mf.append(m)
            pf.append(p)
            of.append(off)
            af.append(a)
        RV_new=sine(JD,m,p,off_new)
        a=accept(RV_old, RV_new, RV_obs)
        c=Xp=np.std(RV_new-RV_obs)
        if c < cmin:
            cmin=c
            mmin=m
            pmin=p
            omin=off_new
        b=rand.uniform(0,1)
        if b <= a:
            off=off_new
            RV_old=RV_new
            rate += 1
            mf.append(m)
            pf.append(p)
            of.append(off)
            af.append(a)
        if (i+1)%10000 == 0:
            print('Finished MCMC step '+str(int(i+1))+'.')
    return m, p, off, RV_old, rate, cmin, mmin, pmin, omin

########################
#Calculate mass from rv

def fm(RVs,period):
    per=period*86400
    r=((G*Ms)/(4*pi**2)*per**2)**(1/3)
    vp=sqrt(G*Ms/r)
    Mp=(Ms*RVs/vp)/1.898e27
    return Mp

find_mass=np.vectorize(fm)

########################
#Plot contours, assume Gaussian

def contour(medx,medy,sigx,sigy, xname, yname, xunits, yunits, plotn):
    ax99, bx99 = stat.norm.interval(0.997, loc=medx, scale=sigx)
    ay99, by99 = stat.norm.interval(0.997, loc=medy, scale=sigy)
    ay95, by95 = stat.norm.interval(0.95, loc=medy, scale=sigy)
    ay68, by68 = stat.norm.interval(0.68, loc=medy, scale=sigy)
    w1x=mlab.normpdf(medx,medx,sigx)
    w1y=mlab.normpdf(ay68,medy,sigy)
    w2y=mlab.normpdf(ay95,medy,sigy)
    w3y=mlab.normpdf(ay99,medy,sigy)
    z1, z2, z3 = w1x*w1y, w1x*w2y, w1x*w3y
    xrang=np.linspace(ax99,bx99,100)
    yrang=np.linspace(ay99,by99,100)
    X=mlab.normpdf(xrang,medx,sigx)
    Y=mlab.normpdf(yrang,medy,sigy)
    a=np.zeros((100,100))
    for i in range(100):
        for j in range(100):
            a[j,i]=Y[j]*X[i]
    cs = plt.contour(xrang,yrang,a,[z1,z2,z3])
    plt.xlabel(str(xname)+' in '+str(xunits))
    plt.ylabel(str(yname)+' in '+str(yunits))
    plt.savefig(str(plotn)+'.png')
    plt.clf()
    
########################

print(' ')
data=np.loadtxt('data.txt') #import data

JD=data[:,0]
RV_obs=data[:,1]
Err=data[:,2]

RV_old=sine(JD,m,p,off)

m, p, off, RV_old=burn(10000, m, p, off, RV_old, RV_obs)

print(' ')

rate=0
cmin=20
mmin=0
pmin=1
omin=0
trial=100000

m, p, off, RV_old, rate, cmin, mmin, pmin, omin = mcmc(trial, m, p, off, RV_old, RV_obs, rate, cmin, mmin, pmin, omin)

if cmin==20:
    sys.exit('MCMC fit poor, run again.')

print(' ')
print('MCMC fitting done.')
print(' ')

acceptance=rate/(3*trial)*100

q=len(JD)-1
mJD=np.linspace(JD[0],JD[q],10000)
mRV=sine(mJD,mmin,pmin,omin)

########################
#Determine Fit############
########################

Mp=find_mass(mmin,pmin)

Mf=find_mass(mf,pmin)

m_med, m_std = np.median(Mf), np.std(Mf, ddof=1)
p_med, p_std = np.median(pf), np.std(pf, ddof=1)
o_med, o_std = np.median(of), np.std(of, ddof=1)

am, bm = stat.norm.interval(0.68, loc=m_med, scale=m_std)
ap, bp = stat.norm.interval(0.68, loc=p_med, scale=p_std)
ao, bo = stat.norm.interval(0.68, loc=o_med, scale=o_std)

print('Writing Output')

log=open('mcmc_fit.txt','w')

print('Best Fit Parameters', file=log)
print('The Msin(i) of the planet is '+str(Mp)+' Jupiter Masses.', file=log)
print('The orbital period is '+str(pmin)+' days.', file=log)
print('The zero piont for the system is '+str(omin)+' JD.', file=log)
print('Acceptance rate was '+str(acceptance)+'%.', file=log)
print(' ', file=log)
print(' ', file=log)
print('Statistical Fits', file=log)
print('Median Msin(i) is '+str(m_med)+' Jupiter masses.', file=log)
print('The 1 sigma confidence interval is '+str(am)+' to '+str(bm)+' J_m.', file=log)
print('Median period is '+str(p_med)+' days.', file=log)
print('The 1 sigma confidence interval is '+str(ap)+' to '+str(bp)+' days', file=log)
print('Median zero point  is '+str(o_med)+' JD', file=log)
print('The 1 sigma confidence interval is '+str(ao)+' to '+str(bo)+' JD', file=log)

log.close

print('Output Written')
print(' ')

plt.show()
########################
#Make Phased data#########
########################

print('Phasing Data')

#Observed
jdphase=((JD-omin)%pmin)/pmin
phase_obs=np.column_stack((jdphase,RV_obs,Err))
Phase_obs=phase_obs[phase_obs[:, 0].argsort()]

log=open('phased_data.txt','w')

for i in range(len(Phase_obs)):
    print(Phase_obs[i,0],Phase_obs[i,1],Phase_obs[i,2], file=log)

log.close()

#Model
jdphase_m=((mJD-omin)%pmin)/pmin
phase_mod=np.column_stack((jdphase_m,mRV))
Phase_mod=phase_mod[phase_mod[:, 0].argsort()]

log=open('phased_model.txt','w')

for i in range(len(Phase_mod)):
    print(Phase_mod[i,0],Phase_mod[i,1], file=log)

log.close()
print('Data Phased')
print(' ')

contour(m_med,p_med,m_std,p_std, 'Msin(i)', 'Period', 'Jupiter Mass', 'Days', 'mass_per')
contour(m_med,o_med-2451800,m_std,o_std, 'Msin(i)', 'Zero Point', 'Jupiter Mass', 'JD+2451800', 'mass_zp')
contour(p_med,o_med-2451800,p_std,o_std, 'Period', 'Zero Point', 'Days', 'JD+2451800', 'per_zp')

########################
print("--- %s seconds ---" % (time.time() - start_time))
