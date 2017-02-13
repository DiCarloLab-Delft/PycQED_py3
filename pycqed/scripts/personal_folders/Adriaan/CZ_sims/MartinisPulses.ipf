#pragma rtGlobals=1		// Use modern global access method.

// FUNCTION: SETUPFLUXPULSING
// --------------------------------------------------
function SetupFluxPulsing([manifold, g1])
variable manifold 		// determines what manifold we work in, 1-exciation or 2-excitation
variable g1
if(paramisdefault(manifold))
	manifold=1;
endif
if(paramisdefault(g1))
	g1=0.025; // in GHz;
endif*

make/o/n=(2) LatestValues

variable/g Fmax=5.94; // in GHz
variable/g Fbus=4.8; 	// in GHz
variable/g Ec=0.3;	// in GHz
variable/g g=g1

if(manifold==2)
	g=g1*sqrt(2);
endif


// Functional form of the bias dependent qubit frequency is

variable/g Vo

if(manifold==1)
	//  F_01(V)=(Fmax+Ec)*sqrt(Cos(V/Vo))-Ec;
	// The scaling factor Vo is chosen so that the 01-10 avoided crossing happens at V=1;
	Vo=1/acos(((Fbus+Ec)/(Fmax+Ec))^2)
else
	//  F_12(V)=(Fmax+Ec)*sqrt(Cos(V/Vo))-2Ec;
	// The scaling factor Vo is chosen so that the 11-02 avoided crossing happens at V=1;
	Vo=1/acos(((Fbus+2*Ec)/(Fmax+Ec))^2)
endif
//print g, Vo

// estimate 2Xi
//variable TwoXi=(sqrt(2)*g)^2*Ec/(Fmax-Fbus)/(Fmax-Fbus-Ec)*1e3	// in MHz
//print TwoXi
end


// FUNCTION: GETTHETAFROMV
// ----------------------------------------------
function getThetaFromV(V, [manifold])
variable V		// scaled so that V=1 lines up with crossing.
variable manifold

if(paramisdefault(manifold))
	manifold=1
endif

NVAR Fmax	// in GHz
NVAR Fbus	// in GHz
NVAR Ec	// in GHz
NVAR g		// in GHz
NVAR Vo

variable thisf
if(manifold==1)
	thisf=(fmax+Ec)*sqrt(cos(V/Vo))-Ec;	// in GHz
else
	thisf=(fmax+Ec)*sqrt(cos(V/Vo))-2*Ec;	// in GHz
endif

variable det=thisf-Fbus;					// in GHz
variable theta=atan2(2*g, det);
//print det, thisf

return theta // in radians

end

// FUNCTION: GETV
// --------------------------
// converts from theta to bias
function getVFromTheta(theta, [manifold])
variable theta 	// in radians.
variable manifold

if(paramisdefault(manifold))
	manifold=1
endif

NVAR Fmax	// in GHz
NVAR Fbus	// in GHz
NVAR Ec	// in GHz
NVAR g		// in GHz
NVAR Vo

variable det=2*g/tan(theta); 	// in GHz

variable V
if(manifold==1)
	V=Vo*Acos(((Fbus+det+Ec)/(Fmax+Ec))^2)
else
	V=Vo*Acos(((Fbus+det+2*Ec)/(Fmax+Ec))^2)
endif

return V;
end


// FUNCTION: GETDETFROMV
// ------------------------------------------
function getDetfromV(V, [manifold])
variable V
variable manifold

if(paramisdefault(manifold))
	manifold=1
endif


NVAR Fmax		// in GHz
NVAR Fbus		// in GHz
NVAR Ec		// in GHz
NVAR Vo		// lever arm for flux coupling

variable det

if(manifold==1)
	det=((Fmax+Ec)*sqrt(cos(V/Vo))-Ec)-Fbus;
else
	det=((Fmax+Ec)*sqrt(cos(V/Vo))-2*Ec)-Fbus;
endif

return det		// in GHz
end

// FUNCTION: GETFLUXPULSE
// -----------------------------------
function getFluxPulse(tp, thetamax, frac,[manifold, lambda1, lambda2, lambda3])
variable tp			// in ns
variable thetamax		// in radians;
variable frac			// numerical overhead.
variable manifold
variable lambda1
variable lambda2
variable lambda3

if(paramisdefault(manifold))
	manifold=2
endif
if (paramisdefault(lambda1))
	lambda1=1;
endif
if (paramisdefault(lambda2))
	lambda2=0;
endif
if(paramisdefault(lambda3))
	lambda3=0;
endif

setupfluxpulsing(manifold=manifold);


variable numt=tp*frac+1;
make/o/n=(numT) ThetaVec;
setscale/i x, 0, tp, ThetaVec;
duplicate/o ThetaVec dThetaVec

variable lambda0=1-lambda1;
variable Thetamin=getThetaFromV(0, manifold=manifold);

// Martinis version
dThetavec=lambda0+lambda1*(1-cos(1*2*pi*x/tp))/2+lambda2*(1-cos(2*2*pi*x/tp))/2 //+ lambda2*(1-cos(2*2*pi*x/tp))// + lambda3*(1-cos(3*2*pi*x/tp))+ lambda4*(1-cos(4*2*pi*x/tp))
// My version
//dThetavec=lambda0+lambda1*sin(2*pi*x/tp/2)//+lambda2*sin(2*pi*x/tp*3/2)+lambda3*sin(2*pi*x/tp*5/2)

// Scaling the dTheta vector
Wavestats/q dThetavec
Thetavec=ThetaMin+dThetaVec*(ThetaMax-ThetaMin)/V_max

// For debugging only: square pulse
//ThetaVec=ThetaMax;

duplicate/o ThetaVec Vvec;
duplicate/o ThetaVec DetVec;
Vvec=Nan;
Detvec=Nan

Vvec[]=getVfromTheta(ThetaVec[p], manifold=manifold);

//// KLOOGE
//Vvec[]=(lambda0+lambda1*(1-cos(1*2*pi*x/tp))/2)
//Wavestats/q Vvec;
////print thetaMax
//Vvec*=ThetaMax/V_max

variable i=0
Do
	if(numtype(Vvec[i])==2)
		Vvec[i]=0;
	endif
	i+=1
While(i<numT)

// Quantize the Vvec to mimic 1ns quantization.
//duplicate/o Vvec VvecQ
//VvecQ=Vvec(floor(x));
//Detvec[]=getDetfromV(Vvec[p])

Detvec[]=getDetfromV(Vvec[p], manifold=manifold);

end


// FUNCTION: EVOLVEH
// ---------------------------------
// Evolves the time-dependent hamiltonian as defined by the detuning vector.
function EvolveH(tp, ThetaMax,frac, [manifold, lambda1, lambda2])
variable tp			// in ns
variable ThetaMax	// in radians
variable frac			// numerical overhead
variable manifold		// specify manifold
variable lambda1		// softening parameter
variable lambda2		// softening parameter

if(paramisdefault(manifold))
	manifold=2
endif
if(paramisdefault(lambda1))
	lambda1=1
endif
if(paramisdefault(lambda2))
	lambda2=0
endif

//Setup pulsing basics, and import global variables
SetupFluxPulsing(manifold=manifold);
NVAR Fmax	// in GHz
NVAR Fbus	// in GHz
NVAR Ec	// in GHz
NVAR g		// in GHz

//Get the detuning vector
getFluxPulse(tp,ThetaMax,frac, manifold=manifold, lambda1=lambda1, lambda2=lambda2);
WAVE DetVec;

variable numT=dimsize(DetVec,0);
variable dt=dimdelta(DetVec,0);
//print numT, dT

make/o/n=(2,2)/C thisH
make/o/n=(2,2)/C thisHm
make/o/n=(2,2)/C thisHp

make/o/n=(2,2)/C Eye=0
Eye[0][0]=1;
Eye[1][1]=1;

make/o/n=(numT+1)  Pop11Vec=NaN;
setscale/P x, 0, dt, Pop11Vec
make/o/n=(numT+1)  PhaseVec=Nan;
setscale/P x, 0, dt, PhaseVec
make/o/n=(numT+1)  ReA11Vec=Nan;
setscale/P x, 0, dt, ReA11Vec


make/o/n=(2)/C PsiT=0
variable Delta=getDetFromV(0, manifold=manifold );
variable Em=Delta/2-sqrt(g^2+(Delta/2)^2);
variable Pnorm=sqrt(1+(Em/g)^2);

// get eigenbasis in OFF state
PsiT[0]=Em/g;
PsiT[1]=1;
PsiT/=Pnorm;
duplicate/o PsiT PsiStart;

matrixop/o/c  overlap=PsiStart^h x PsiT
Pop11vec[0]=cabs(overlap[0])^2;
PhaseVec[0]=atan2(imag(PsiT[1]),real(PsiT[1]))/pi;
ReA11Vec[0]=real(PsiT[1]);

variable i=1
Do
	// construct the hamiltonian
	thisH[0][0]=DetVec[i-1];
	thisH[0][1]=g;
	thisH[1][0]=g;
	thisH[1][1]=0;

	// Jos Thijssen's trick: see his book
	thisHm=(Eye+cmplx(0,-1)*thisH/2*dt*2*pi);
	thisHp=(Eye+cmplx(0,1)* thisH/2*dt*2*pi);
	matrixop/o/c PsiT=inv(thisHp) x thisHm x PsiT

	// normalize
	matrixop/o  foo=PsiT^h x PsiT
	PsiT/=sqrt(foo[0]);

	matrixop/o/c  overlap=PsiStart^h x PsiT
	Pop11vec[i]=cabs(overlap[0])^2;
	PhaseVec[i]=atan2(imag(PsiT[1]),real(PsiT[1]));
	ReA11Vec[i]=real(PsiT[1]);
	i+=1
While(i<=numT);

phaseunwrap(PhaseVec);
PhaseVec/=pi;

make/o/n=(2) latestValues;
latestValues[0]=1-Pop11vec[numT];
latestValues[1]=PhaseVec[numT];

return  latestValues[1]
end



