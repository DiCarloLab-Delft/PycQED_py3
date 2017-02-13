#pragma rtGlobals=3		// Use modern global access method and strict wave access.

// Function: getLandscape
// ----------------------------------
function getLandscape(tg, Vstart, Vend, numV, startL1, endL1, numL1)
variable tg 		// in ns
variable Vstart
variable Vend
variable numV
variable startL1
variable endL1
variable numL1


wave latestvalues

make/o/n=(numV, numL1) Phase2Q=NaN
setscale/I x, Vstart, Vend, Phase2Q
setscale/I y, startL1, endL1, Phase2Q
duplicate/o Phase2Q Leakage

setupfluxpulsing(manifold=2);

variable i=0
variable j=0
Do
	i=0
	Do
		Phase2Q[i][j]=evolveH(tg, getthetafromV(x, manifold=2),10, manifold=2, lambda1=y);
		//Phase2Q[i][j]=evolveH(tg, x,10, manifold=2, lambda1=y)
		Leakage[i][j]=latestvalues[0];
		i+=1
		doupdate
	While(i<numV)
	//doupdate;
	j+=1
While(j<numL1);
end



// Function: LeakageatTg
// ---------------------------------
function LeakageatTg(tg, frac, [manifold, lambda1, lambda2])
variable tg
variable frac			// numerical overhead
variable manifold
variable lambda1
variable lambda2

if(paramisdefault(manifold))
	manifold=2
endif
if(paramisdefault(lambda1))
	lambda1=1
endif
if(paramisdefault(lambda1))
	lambda2=0
endif

variable num=101
variable Vstart=0.9
variable Vend=1.1

make/o/n=(num)/D LeakageVec=NaN;
setscale/I x, Vstart, Vend, LeakageVec
duplicate/o LeakageVec Phase2QVec

WAVE LatestValues

SetupFluxPulsing(manifold=manifold);

variable ThetaMax
variable i=0
Do
	ThetaMax=getThetafromV(pnt2x(LeakageVec,i),manifold=manifold)
	Phase2Qvec[i]=EvolveH(tg,ThetaMax,frac, manifold=manifold, lambda1=lambda1, lambda2=lambda2);
	Leakagevec[i]=LatestValues[0];
	i+=1
While(i<num)

findlevel/Q Phase2Qvec, 1;
variable thisleakage=LeakageVec(V_levelX);
//print V_levelX, thisleakage
return thisleakage
end


// Function: LeakagevsTg
// ---------------------------------
function LeakagevsTg(tgstart,tgend, numtg,  frac, [manifold, lambda1, lambda2])
variable tgstart
variable tgend
variable numtg
variable frac
variable manifold
variable lambda1
variable lambda2

make/o/n=(numtg)  LeakageAtPi=NaN
setscale/I x, tgstart, tgend, LeakageAtPi

if(paramisdefault(manifold))
	manifold=2;
endif
if(paramisdefault(lambda1))
	lambda1=1;
endif
if(paramisdefault(lambda2))
	lambda2=0;
endif

Variable i=0
variable thistg
Do
	thistg=pnt2x(LeakageAtPi,i);
	LeakageAtPi[i]=LeakageatTg(thistg, frac, manifold=manifold, lambda1=lambda1,lambda2=lambda2)
	doupdate;
	wait(1);

	i+=1
While(i<numtg)
end


macro ManyRuns()
saveexperiment;


//leakagevsTg(20,60,41,30, manifold=2, lambda1=1,lambda2=0.00);
//duplicate/o LeakageAtPi LeakageAtPi_1_pt00;
////appendtograph LeakageAtPi_1_pt00;
//auto(); dotize();
//saveexperiment;
//
//leakagevsTg(20,60,41,30, manifold=2, lambda1=1,lambda2=0.02);
//duplicate/o LeakageAtPi LeakageAtPi_1_pt02
////appendtograph LeakageAtPi_1_pt02;
//auto(); dotize();
//saveexperiment;
//
//leakagevsTg(20,60,41,30, manifold=2, lambda1=1,lambda2=0.04);
//duplicate/o LeakageAtPi LeakageAtPi_1_pt04
////appendtograph LeakageAtPi_1_pt04;
//auto(); dotize();
//saveexperiment;
//
//leakagevsTg(20,60,41,30, manifold=2, lambda1=1,lambda2=0.06);
//duplicate/o LeakageAtPi LeakageAtPi_1_pt06
////appendtograph LeakageAtPi_1_pt06;
//auto(); dotize();
//saveexperiment;
//
//leakagevsTg(20,60,41,30, manifold=2, lambda1=1,lambda2=0.08);
//duplicate/o LeakageAtPi LeakageAtPi_1_pt08
////appendtograph LeakageAtPi_1_pt08;
//auto(); dotize();
//saveexperiment;
//
//leakagevsTg(20,60,41,30, manifold=2, lambda1=1,lambda2=0.10);
//duplicate/o LeakageAtPi LeakageAtPi_1_pt10
////appendtograph LeakageAtPi_1_pt10;
//auto(); dotize();
//saveexperiment;
//
//leakagevsTg(20,60,41,30, manifold=2, lambda1=1,lambda2=0.12);
//duplicate/o LeakageAtPi LeakageAtPi_1_pt12
////appendtograph LeakageAtPi_1_pt12;
//auto(); dotize();
//saveexperiment;
//
//leakagevsTg(20,60,41,30, manifold=2, lambda1=1,lambda2=0.14);
//duplicate/o LeakageAtPi LeakageAtPi_1_pt14
//appendtograph LeakageAtPi_1_pt14;
//auto(); dotize();
//saveexperiment;
//
//
//leakagevsTg(20,60,41,30, manifold=2, lambda1=1,lambda2=0.15);
//duplicate/o LeakageAtPi LeakageAtPi_1_pt15
//appendtograph LeakageAtPi_1_pt15;
//auto(); dotize();
//saveexperiment;


leakagevsTg(20,60,41,30, manifold=2, lambda1=1,lambda2=0.16);
duplicate/o LeakageAtPi LeakageAtPi_1_pt16
appendtograph LeakageAtPi_1_pt16;
auto(); dotize();
saveexperiment;

leakagevsTg(20,60,41,30, manifold=2, lambda1=1,lambda2=0.17);
duplicate/o LeakageAtPi LeakageAtPi_1_pt17
appendtograph LeakageAtPi_1_pt17;
auto(); dotize();
saveexperiment;

leakagevsTg(20,60,41,30, manifold=2, lambda1=1,lambda2=0.18);
duplicate/o LeakageAtPi LeakageAtPi_1_pt18
appendtograph LeakageAtPi_1_pt18;
auto(); dotize();
saveexperiment;

leakagevsTg(20,60,41,30, manifold=2, lambda1=1,lambda2=0.19);
duplicate/o LeakageAtPi LeakageAtPi_1_pt19
appendtograph LeakageAtPi_1_pt19;
auto(); dotize();
saveexperiment;

leakagevsTg(20,60,41,30, manifold=2, lambda1=1,lambda2=0.20);
duplicate/o LeakageAtPi LeakageAtPi_1_pt20
appendtograph LeakageAtPi_1_pt20;
auto(); dotize();
saveexperiment;

end
