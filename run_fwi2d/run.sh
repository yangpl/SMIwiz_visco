suwaveform type=ricker1 dt=0.002 ns=2500 fpeak=5 |sustrip > fricker.bin

echo "#input parameters
===============================================================================
mode=1 //0, forward modeling; 1, FWI; 2, RTM; 3, LSRTM; 4, FWI gradient; 5, source inversion

acquifile=acqui.txt
vpfile=../Marmousi/vp_init.bin
rhofile=../Marmousi/rho_init.bin
qinvfile=../Marmousi/qinv_init.bin
stffile=fricker.bin

nrelax=0
freesurf=1 //free surface boundary condition
fm=10
nt=2500 //number of time steps
dt=0.002 //temporal sampling
nb=20  //Sponge ABC
n1=151 //size of input FD model
n2=461 //size of input FD model
d1=20 //grid spacing of the input FD model
d2=20 //grid spacing of the input FD model


===============================================================================
opt_method=1 //1=l-BFGS; 2=Newton-CG
ncg=5
niter=50 //number of iterations using l-BFGS
nls=10 //number of line search per iteration
npair=5 //memory length in l-BFGS
preco=0 //implicit reparameterization preconditioner

family=1
npar=2
idxpar=1,2 //only 1 parameter - velocity
bound=1 //bound the inversion parameters
vpmin=1500
vpmax=5500
rhomin=1000
rhomax=3000
qinvmin=0.001
qinvmax=0.1

===============================================================================
dxwdat=100 //dx for data weighting
xwdat=0,0.3,0.7,1,1 //weights for dx increment

===============================================================================
muteopt=0 //0, no mute; 1, front mute; 2, tail mute; 3, front and tail mute
ntaper=20 //number of points for taper
xmute1=50,775.8,2227.4 
tmute1=0.1,0.4,1 
xmute2=50,603.3,1954.3 
tmute2=0.3,0.64,1

preco_r1=2
preco_r2=2
preco_r3=1

" >inputpar.txt

export OMP_NUM_THREADS=1
mpirun -n 24 ../bin/SMIwiz $(cat inputpar.txt)  >&out.log &
