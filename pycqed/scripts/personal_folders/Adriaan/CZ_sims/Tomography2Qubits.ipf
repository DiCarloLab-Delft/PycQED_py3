#pragma rtGlobals=1     // Use modern global access method.

// Function: setupST2Q
// -------------------------------
function setupST2Q()

make/o/n=12 BetaVec
variable/g doffsetP
variable/g doffsetM

end


// Function: Minimizeoffset2Q
// ------------------------------------
function minimizeoffset2Q(start, [stop, phase, base, directory,datestr, dotwo])
variable start
variable stop
string base
string phase
string directory
string datestr
variable dotwo


if(paramisdefault(phase))
    phase="Q"
endif

if(paramisdefault(stop))
    stop=start;
endif

SVAR datecode
if(paramisdefault(datestr))
    datestr=datecode
endif
if(paramisdefault(directory))
    directory="";
endif
if(paramisdefault(base))
    base="ST2Q";
endif
if(paramisdefault(dotwo))
    dotwo=1;
endif

variable numruns=stop-start+1;

NVAR doffset
NVAR scalefac

variable numoffsets=21;
variable startoffset=-50;
variable deltaoffset=5;

make/o/n=(numoffsets) VsqvecP=Nan
setscale/P x, startoffset, deltaoffset, VsqvecP

make/o/n=(numoffsets) VsqvecM=Nan
setscale/P x, startoffset, deltaoffset, VsqvecM

NVAR fitwait

variable i=0;
variable j=0;

make/o/n=(numruns) minoffsetsP=Nan;
setscale/P x, start,1, minoffsetsP
make/o/n=(numruns) minoffsetsM=Nan;
setscale/P x, start,1, minoffsetsM

//cs(start,stop,phase,base=base, directory=directory,datestring=datestr);

WAVE Pvec2QP
WAVE Pvec2QM

NVAR doffsetP
NVAR doffsetM

make/o/n=(16) thisPvec2Q

Do
    VsqvecP=NaN;
    VsqvecM=NaN;
    i=0
    Do
            doffsetP=startoffset+i*deltaoffset;
            doffsetM=startoffset+i*deltaoffset;
            scalefac=1.0;
            quick2Q(start+j,phase=phase, datestr=datestr,opcode=1,savenum=-1);

        thisPvec2Q=Pvec2QP;
            thisPvec2Q[5]=Nan;
            thisPvec2Q[10]=NaN;
            thisPvec2Q[15]=NaN;
        Wavestats/q thisPvec2Q
        VsqVecP[i]=(V_rms^2*16)/4

        thisPvec2Q=Pvec2QM;
            thisPvec2Q[5]=Nan;
            thisPvec2Q[10]=NaN;
            thisPvec2Q[15]=NaN;
        Wavestats/q thisPvec2Q
        VsqVecM[i]=(V_rms^2*16)/4

        if(fitwait>0)
            doupdate
            wait(fitwait);
        endif

        i+=1;
    While(i<numoffsets)

    Wavestats/q VsqVecP;
    MinoffsetsP[j]=V_minloc;
    Wavestats/q VsqVecM;
    MinoffsetsM[j]=V_minloc;


    doffsetP=MinoffsetsP[j];
    doffsetM=MinoffsetsM[j];
    scalefac=1.0;
    quick2Q(start+j,phase=phase, datestr=datestr,opcode=1,savenum=1);

    doupdate;
    j+=1
While(j<numruns)

print MinoffsetsP[0], MinoffsetsM[0];

return 1
end

// Function: EofF
// ---------------------
function  EofF(C)
variable C
variable x=(1+sqrt(1-c^2))/2
variable E=-x*log(x)/log(2)-(1-x)*log(1-x)/log(2);
return E

end


// Function: quick2Q
// ---------------------------
function  quick2Q(start, [opcode, phase, savenum, stop, datestr,base])
variable opcode
variable start
string phase
variable savenum            // 0 : do not save to disk with the num as a suffix
                        // 1:  save to disk with num as a suffix
variable stop
string datestr
string base

SVAR datecode

if(ParamisDefault(opcode))
    opcode=1
endif

if(ParamIsDefault(phase))
    phase="Q"
endif

if(ParamisDefault(savenum))
    savenum=1
endif
if(ParamisDefault(datestr))
    datestr=datecode
endif
if(ParamisDefault(base))
    base="ST2Q"
endif
if(Paramisdefault(stop))
    stop=start
endif

variable numruns=abs(stop-start)+1;
variable i=0;

NVAR doplots;

    Do
        if(opcode==0)
            //doplots=0;  CSrunsv2(datestr,base,"A",start+i,start+i,1,0,4,5,0,0,1);
        endif
        if(opcode==0 || opcode==1)
            doplots=0;  dualST2Q(start+i,base,phase,datestr,1,2,0,2,savenum)
        endif
        i+=1
        //doupdate
While(i<numruns)
end


// Function: DualST2Q
// ------------------------------
function DualST2Q(filenum, basename, phase, datestr, dotwo, opcode, donorm, whichplot,savenum)
    variable filenum
    string basename
    string phase
    string datestr
    variable dotwo      // dotwo=0: one set of ST2Q vectors
                        // dotwo=1: two sets of ST2Q vectors back to back.
    variable opcode     // opcode==0: asume 1 point for each type of measurements,
                        // opcode==1: assume 2 points for each type of measurements
                        // opcode==2: assume 8 points for each of the 4 pi/2,pi/2 measurements.
    variable donorm     // donorm==1: normalize Betas to Beta_1; donorm==0 : do not normalize
    variable whichplot      // whichplot==0, display normwave(s), whichplot==1, display ST2Qvec(s). whichplot==2, display ST2QPaulivec(s)
    variable savenum


    NVAR doffsetP
    NVAR doffsetM

    variable calSegments=20;

    variable numExpsTotal=calSegments;
    if(opcode==0)
            numExpsTotal+=15;
    elseif(opcode==1)
            numExpstotal+=30;
    elseif(opcode==2)
            numExpsTotal+=54;
    endif

    //import the already loaded experiment vector
    string inputwavename=fancynum2str(filenum)+"_"+phase+"2DPlot"+basename+"_"+datestr+"_ex1_v";
    //print inputwavename
    wave inputwave=$inputwavename


    variable Beta1, Beta2, Beta12
    variable dBeta1, dBeta2, dBeta12
    variable M1, M2, M3, M4
    variable dM1, dM2, dM3, dM4
    variable offset

    duplicate/o inputwave normwave
    WAVE normwave

    wavestats/Q/R=[0,calSegments-1] inputwave
    offset=V_avg
    normwave-=offset

    // define some strings
    string ST2Qvecfilename1
    string ST2Qvecfilename2
    string ST2QPaulivecfilename1
    string ST2QPaulivecfilename2
    string ST2QvecPname
    string ST2QPaulivecPname
    string ST2QvecMname
    string ST2QPaulivecMname
    string ST2QPaulivecHname


    NVAR doplots
    NVAR bequiet

    WAVE Betavec

    WAVE ST2QPauliVec

// Analyze the first or only half
    wavestats/Q/R=[0,calSegments/4-1] normwave
    M1=V_avg
    dM1=V_sdev/sqrt(calSegments/4);
    wavestats/Q/R=[calSegments/4,2*calSegments/4-1] normwave
    M2=V_avg
    dM2=V_sdev/sqrt(calSegments/4);
    wavestats/Q/R=[2*calSegments/4,3*calSegments/4-1] normwave
    M3=V_avg
    dM3=V_sdev/sqrt(calSegments/4);
    wavestats/Q/R=[3*calSegments/4,4*calSegments/4-1] normwave
    M4=V_avg
    dM4=V_sdev/sqrt(calSegments/4);

    Beta1=(M1+M3)/2;
    Beta2=((M1+M2)/2);
    Beta12=((M1+M4)/2);

    dBeta1=sqrt(dM1^2+dM3^2)/2;
    dBeta2=sqrt(dM1^2+dM2^2)/2;
    dBeta12=sqrt(dM1^2+dM4^2)/2;

    // Copy the Beta and dBeta values into the BetaVec wave used by other functions, such as analyzeRFsweep
    BetaVec[0]=Beta1;
    BetaVec[1]=Beta2;
    BetaVec[2]=Beta12;
    BetaVec[3]=dBeta1;
    BetaVec[4]=dBeta2;
    BetaVec[5]=dBeta12;

    normwave+=doffsetP;

    if(bequiet==0)
        printf  " from Run P: Beta1=%.4g +/- %.2g,  Beta2=%.4g +/- %.2g,    Beta12=%.4g +/- %.2g    (%.2f, %.2f, %.2f)   offset=%.4g\r", Beta1, dBeta1,  Beta2, dBeta2,   Beta12, dBeta12,   dBeta1/Beta1, dBeta2/Beta2, dBeta12/Beta12, offset
    endif


    make/o/n=(19) ST2Qvec=NaN
    setscale/P x, -2, 1, ST2Qvec;

    ST2Qvec[0]=Beta1;
    ST2Qvec[1]=Beta2;
    ST2Qvec[2]=Beta12;

    if(opcode==0)
        ST2Qvec[3,17]=  normwave[calSegments+(p-3)];
    elseif(opcode==1)
        ST2Qvec[3,17]=  (normwave[calSegments+2*(p-3)]+normwave[calSegments+2*(p-3)+1])/2;
    else
        ST2Qvec[3]=     (normwave[calSegments+0]+normwave[calSegments+1])/2;
        ST2Qvec[4]=     (normwave[calSegments+2]+normwave[calSegments+3])/2;
        ST2Qvec[5]=     (normwave[calSegments+4]+normwave[calSegments+5])/2;
        ST2Qvec[6]=     (normwave[calSegments+6]+normwave[calSegments+7])/2;

        ST2Qvec[7]=     (normwave[calSegments+  8]+normwave[calSegments+  9]+normwave[calSegments+10]+normwave[calSegments+11]+normwave[calSegments+12]+normwave[calSegments+13]+normwave[calSegments+14]+normwave[calSegments+15])/8;
        ST2Qvec[8]=     (normwave[calSegments+16]+normwave[calSegments+17]+normwave[calSegments+18]+normwave[calSegments+19]+normwave[calSegments+20]+normwave[calSegments+21]+normwave[calSegments+22]+normwave[calSegments+23])/8;

        ST2Qvec[9]=     (normwave[calSegments+24]+normwave[calSegments+25])/2;
        ST2Qvec[10]=    (normwave[calSegments+26]+normwave[calSegments+27])/2;

        ST2Qvec[11]=    (normwave[calSegments+28]+normwave[calSegments+29]+normwave[calSegments+30]+normwave[calSegments+31]+normwave[calSegments+32]+normwave[calSegments+33]+normwave[calSegments+34]+normwave[calSegments+35])/8;
        ST2Qvec[12]=    (normwave[calSegments+36]+normwave[calSegments+37]+normwave[calSegments+38]+normwave[calSegments+39]+normwave[calSegments+40]+normwave[calSegments+41]+normwave[calSegments+42]+normwave[calSegments+43])/8;

        ST2Qvec[13]=    (normwave[calSegments+44]+normwave[calSegments+45])/2;
        ST2Qvec[14]=    (normwave[calSegments+46]+normwave[calSegments+47])/2;
        ST2Qvec[15]=    (normwave[calSegments+48]+normwave[calSegments+49])/2;
        ST2Qvec[16]=    (normwave[calSegments+50]+normwave[calSegments+51])/2;
        ST2Qvec[17]=    (normwave[calSegments+52]+normwave[calSegments+53])/2;
    endif
    ST2Qvec[18]=1

    make/o/n=(16) tempvec
    tempvec[]=ST2Qvec[3+p];

    // Finally convert to measurements into the Pauli basis
    make/o/n=(16,16) ObsFromPauli=0;

    ObsFromPauli[15][0]=1;

    ObsFromPauli[0][3]=Beta1;
    ObsFromPauli[1][3]=-Beta1;
    ObsFromPauli[2][3]=Beta1;
    ObsFromPauli[3][2]=Beta1;
    ObsFromPauli[4][2]=Beta1;
    ObsFromPauli[5][2]=Beta1;
    ObsFromPauli[6][2]=Beta1;
    ObsFromPauli[7][1]=-Beta1;
    ObsFromPauli[8][1]=-Beta1;
    ObsFromPauli[9][1]=-Beta1;
    ObsFromPauli[10][1]=    -Beta1;
    ObsFromPauli[11][3]=    Beta1;
    ObsFromPauli[12][3]=    -Beta1;
    ObsFromPauli[13][3]=Beta1;
    ObsFromPauli[14][3]=-Beta1;

    ObsFromPauli[0][12]=Beta2
    ObsFromPauli[1][12]=Beta2
    ObsFromPauli[2][12]=-Beta2
    ObsFromPauli[3][12]=Beta2
    ObsFromPauli[4][8]=Beta2
    ObsFromPauli[5][4]=-Beta2
    ObsFromPauli[6][12]=-Beta2
    ObsFromPauli[7][12]=Beta2
    ObsFromPauli[8][8]=Beta2
    ObsFromPauli[9][4]=-Beta2
    ObsFromPauli[10][12]=-Beta2
    ObsFromPauli[11][8]=Beta2
    ObsFromPauli[12][8]=Beta2
    ObsFromPauli[13][4]=-Beta2
    ObsFromPauli[14][4]=-Beta2

    ObsFromPauli[0][15]=Beta12
    ObsFromPauli[1][15]=-Beta12
    ObsFromPauli[2][15]=-Beta12
    ObsFromPauli[3][14]=Beta12
    ObsFromPauli[4][10]=Beta12
    ObsFromPauli[5][6]=-Beta12
    ObsFromPauli[6][14]=-Beta12
    ObsFromPauli[7][13]=-Beta12
    ObsFromPauli[8][9]=-Beta12
    ObsFromPauli[9][5]=Beta12
    ObsFromPauli[10][13]=Beta12
    ObsFromPauli[11][11]=Beta12
    ObsFromPauli[12][11]=-Beta12
    ObsFromPauli[13][7]=-Beta12
    ObsFromPauli[14][7]=Beta12

    duplicate/o ObsFromPauli ObsFromPauliP
    WAVE ObsFromPauliP

    MatrixOP/O PauliFromObs=Inv(ObsFromPauli)
    MatrixOP/O ST2QPauliVec= PauliFromObs x tempvec
    setscale/P x, 1, 1, ST2QPauliVec


    // Save vectors  to the tomopath
    duplicate/o ST2QPaulivec Pvec2Q
    duplicate/o ST2QPaulivec Pvec2QP

if(savenum>-1)
    ST2Qvecfilename1="ST2QvecP"+phase+".txt"
    ST2Qvecfilename2="ST2QvecP"+".txt"
    Save/O/G/M="\r\n"/P=tomopath ST2Qvec as ST2Qvecfilename1
    Save/O/G/M="\r\n"/P=tomopath ST2Qvec as ST2Qvecfilename2
    ST2Qvecfilename1="ST2Qvec"+phase+".txt"
    ST2Qvecfilename2="ST2Qvec"+".txt"
    Save/O/G/M="\r\n"/P=tomopath ST2Qvec as ST2Qvecfilename1
    Save/O/G/M="\r\n"/P=tomopath ST2Qvec as ST2Qvecfilename2

    if(savenum==1)
        ST2Qvecfilename1="ST2QvecP"+phase+"_"+num2str(filenum)+".txt"
        ST2Qvecfilename2="ST2QvecP"+"_"+num2str(filenum)+".txt"
        Save/O/G/M="\r\n"/P=tomopath ST2Qvec as ST2Qvecfilename1
        Save/O/G/M="\r\n"/P=tomopath ST2Qvec as ST2Qvecfilename2
        ST2Qvecfilename1="ST2Qvec"+phase+"_"+num2str(filenum)+".txt"
        ST2Qvecfilename2="ST2Qvec"+"_"+num2str(filenum)+".txt"
        Save/O/G/M="\r\n"/P=tomopath ST2Qvec as ST2Qvecfilename1
        Save/O/G/M="\r\n"/P=tomopath ST2Qvec as ST2Qvecfilename2
    endif

    ST2QPaulivecfilename1="ST2QPaulivecP"+phase+".txt"
    ST2QPaulivecfilename2="ST2QPaulivecP"+".txt"
    Save/O/G/M="\r\n"/P=tomopath ST2QPaulivec as ST2QPaulivecfilename1
    Save/O/G/M="\r\n"/P=tomopath ST2QPaulivec as ST2QPaulivecfilename2
    ST2QPaulivecfilename1="ST2QPaulivec"+phase+".txt"
    ST2QPaulivecfilename2="ST2QPaulivec"+".txt"
    Save/O/G/M="\r\n"/P=tomopath ST2QPaulivec as ST2QPaulivecfilename1
    Save/O/G/M="\r\n"/P=tomopath ST2QPaulivec as ST2QPaulivecfilename2

    if(savenum==1)
        ST2QPaulivecfilename1="ST2QPaulivecP"+phase+"_"+num2str(filenum)+".txt"
        ST2QPaulivecfilename2="ST2QPaulivecP"+"_"+num2str(filenum)+".txt"
        Save/O/G/M="\r\n"/P=tomopath ST2QPaulivec as ST2QPaulivecfilename1
        Save/O/G/M="\r\n"/P=tomopath ST2QPaulivec as ST2QPaulivecfilename2
        ST2QPaulivecfilename1="ST2QPaulivec"+phase+"_"+num2str(filenum)+".txt"
        ST2QPaulivecfilename2="ST2QPaulivec"+"_"+num2str(filenum)+".txt"
        Save/O/G/M="\r\n"/P=tomopath ST2QPaulivec as ST2QPaulivecfilename1
        Save/O/G/M="\r\n"/P=tomopath ST2QPaulivec as ST2QPaulivecfilename2
    endif
endif


    // Make copies of vectors with run-specific names
    ST2QvecPname="ST2QvecP_"+fancynum2str(filenum)+"_"+phase+"_"+basename+"_"+datestr;
    duplicate/O ST2Qvec $ST2QvecPname
    WAVE thisST2QvecP=$ST2QvecPname

    ST2QPaulivecPname="ST2QPaulivecP_"+fancynum2str(filenum)+"_"+phase+"_"+basename+"_"+datestr;
    duplicate/O ST2QPaulivec $ST2QPaulivecPname
    WAVE thisST2QPaulivecP=$ST2QPaulivecPname

    make/o/n=(30) PQ1vecD=NaN
    make/o/n=(30) PQ2vecD=Nan
    make/o/n=(30) PQ12vecD=Nan

    setscale/P x, 3,1, PQ1vecD
    setscale/P x, 3,1, PQ2vecD
    setscale/P x, 3,1, PQ12vecD

    PQ1vecD[0]=ST2QPaulivec[1];
    PQ1vecD[2]=ST2QPaulivec[2];
    PQ1vecD[4]=ST2QPaulivec[3];

    PQ2vecD[6]=ST2QPaulivec[4]
    PQ2vecD[8]=ST2QPaulivec[8]
    PQ2vecD[10]=ST2QPaulivec[12]

    PQ12vecD[12]=ST2QPaulivec[9]
    PQ12vecD[14]=ST2QPaulivec[13]
    PQ12vecD[16]=ST2QPaulivec[6]
    PQ12vecD[18]=ST2QPaulivec[14]
    PQ12vecD[20]=ST2QPaulivec[7]
    PQ12vecD[22]=ST2QPaulivec[11]
    PQ12vecD[24]=ST2QPaulivec[5]
    PQ12vecD[26]=ST2QPaulivec[10]
    PQ12vecD[28]=ST2QPaulivec[15]

    duplicate/o PQ1vecD PQ1vecDP
    duplicate/o PQ2vecD PQ2vecDP
    duplicate/o PQ12vecD PQ12vecDP

    PQ1vecD=NaN;
    PQ2vecD=NaN;
    PQ12vecD=NaN;
    duplicate/o PQ1vecD PQ1vecDM
    duplicate/o PQ2vecD PQ2vecDM
    duplicate/o PQ12vecD PQ12vecDM

if(dotwo==1)

    if(dimsize(normwave,0)==2*numExpsTotal)

        duplicate/o inputwave normwaveM
        WAVE normwaveM
        normwaveM[0,numExpsTotal-1]=inputwave[numExpsTotal+p];
        wavestats/Q/R=[0,calSegments-1] normwaveM
        offset=V_avg
        normwaveM-=offset


        wavestats/Q/R=[0,calSegments/4-1] normwaveM
        M1=V_avg
        dM1=V_sdev/sqrt(calSegments/4);
        wavestats/Q/R=[calSegments/4,2*calSegments/4-1] normwaveM
        M2=V_avg
        dM2=V_sdev/sqrt(calSegments/4);
        wavestats/Q/R=[2*calSegments/4,3*calSegments/4-1] normwaveM
        M3=V_avg
        dM3=V_sdev/sqrt(calSegments/4);
        wavestats/Q/R=[3*calSegments/4,4*calSegments/4-1] normwaveM
        M4=V_avg
        dM4=V_sdev/sqrt(calSegments/4);

        Beta1=(M1+M3)/2;
        Beta2=((M1+M2)/2);
        Beta12=((M1+M4)/2);

        dBeta1=sqrt(dM1^2+dM3^2)/2;
        dBeta2=sqrt(dM1^2+dM2^2)/2;
        dBeta12=sqrt(dM1^2+dM4^2)/2;

        // Copy the Beta and dBeta values into the BetaVec wave used by other functions, such as analyzeRFsweep
        BetaVec[6]=Beta1;
        BetaVec[7]=Beta2;
        BetaVec[8]=Beta12;
        BetaVec[9]=dBeta1;
        BetaVec[10]=dBeta2;
        BetaVec[11]=dBeta12;

        normwaveM+=doffsetM;

        if(bequiet==0)
            printf  " from Run M: Beta1=%.4g +/- %.2g,  Beta2=%.4g +/- %.2g,    Beta12=%.4g +/- %.2g    (%.2f, %.2f, %.2f)   offset=%.4g\r", Beta1, dBeta1,  Beta2, dBeta2,   Beta12, dBeta12,   dBeta1/Beta1, dBeta2/Beta2, dBeta12/Beta12, offset
        endif

        make/o/n=(19) ST2Qvec=NaN
        setscale/P x, -2, 1, ST2Qvec;

        ST2Qvec[0]=Beta1;
        ST2Qvec[1]=Beta2;
        ST2Qvec[2]=Beta12;

        if(opcode==0)
            ST2Qvec[3,17]=normwaveM[calSegments+(p-3)]
        elseif(opcode==1)
            ST2Qvec[3,17]=(normwaveM[calSegments+2*(p-3)]+normwaveM[calSegments+2*(p-3)+1])/2
        else
            ST2Qvec[3]=     (normwaveM[calSegments+0]+normwaveM[calSegments+1])/2;
            ST2Qvec[4]=     (normwaveM[calSegments+2]+normwaveM[calSegments+3])/2;
            ST2Qvec[5]=     (normwaveM[calSegments+4]+normwaveM[calSegments+5])/2;
            ST2Qvec[6]=     (normwaveM[calSegments+6]+normwaveM[calSegments+7])/2;

            ST2Qvec[7]=     (normwaveM[calSegments+  8]+normwaveM[calSegments+  9]+normwaveM[calSegments+10]+normwaveM[calSegments+11]+normwaveM[calSegments+12]+normwaveM[calSegments+13]+normwaveM[calSegments+14]+normwaveM[calSegments+15])/8;
            ST2Qvec[8]=     (normwaveM[calSegments+16]+normwaveM[calSegments+17]+normwaveM[calSegments+18]+normwaveM[calSegments+19]+normwaveM[calSegments+20]+normwaveM[calSegments+21]+normwaveM[calSegments+22]+normwaveM[calSegments+23])/8;

            ST2Qvec[9]=     (normwaveM[calSegments+24]+normwaveM[calSegments+25])/2;
            ST2Qvec[10]=    (normwaveM[calSegments+26]+normwaveM[calSegments+27])/2;

            ST2Qvec[11]=    (normwaveM[calSegments+28]+normwaveM[calSegments+29]+normwaveM[calSegments+30]+normwaveM[calSegments+31]+normwaveM[calSegments+32]+normwaveM[calSegments+33]+normwaveM[calSegments+34]+normwaveM[calSegments+35])/8;
            ST2Qvec[12]=    (normwaveM[calSegments+36]+normwaveM[calSegments+37]+normwaveM[calSegments+38]+normwaveM[calSegments+39]+normwaveM[calSegments+40]+normwaveM[calSegments+41]+normwaveM[calSegments+42]+normwaveM[calSegments+43])/8;

            ST2Qvec[13]=    (normwaveM[calSegments+44]+normwaveM[calSegments+45])/2;
            ST2Qvec[14]=    (normwaveM[calSegments+46]+normwaveM[calSegments+47])/2;
            ST2Qvec[15]=    (normwaveM[calSegments+48]+normwaveM[calSegments+49])/2;
            ST2Qvec[16]=    (normwaveM[calSegments+50]+normwaveM[calSegments+51])/2;
            ST2Qvec[17]=    (normwaveM[calSegments+52]+normwaveM[calSegments+53])/2;
        endif
        ST2Qvec[18]=1

        make/o/n=(16) tempvec
        tempvec[]=ST2Qvec[3+p];

        // Finally convert to measurements into the Pauli basis
        make/o/n=(16,16) ObsFromPauli=0;

        ObsFromPauli[15][0]=1;

        ObsFromPauli[0][3]=Beta1;
        ObsFromPauli[1][3]=-Beta1;
        ObsFromPauli[2][3]=Beta1;
        ObsFromPauli[3][2]=-Beta1;
        ObsFromPauli[4][2]=-Beta1;
        ObsFromPauli[5][2]=-Beta1;
        ObsFromPauli[6][2]=-Beta1;
        ObsFromPauli[7][1]=Beta1;
        ObsFromPauli[8][1]=Beta1;
        ObsFromPauli[9][1]=Beta1;
        ObsFromPauli[10][1]=Beta1;
        ObsFromPauli[11][3]=Beta1;
        ObsFromPauli[12][3]=    -Beta1;
        ObsFromPauli[13][3]=Beta1;
        ObsFromPauli[14][3]=-Beta1;

        ObsFromPauli[0][12]=Beta2
        ObsFromPauli[1][12]=Beta2
        ObsFromPauli[2][12]=-Beta2
        ObsFromPauli[3][12]=Beta2
        ObsFromPauli[4][8]=-Beta2
        ObsFromPauli[5][4]=Beta2
        ObsFromPauli[6][12]=-Beta2
        ObsFromPauli[7][12]=Beta2
        ObsFromPauli[8][8]=-Beta2
        ObsFromPauli[9][4]=Beta2
        ObsFromPauli[10][12]=-Beta2
        ObsFromPauli[11][8]=-Beta2
        ObsFromPauli[12][8]=-Beta2
        ObsFromPauli[13][4]=Beta2
        ObsFromPauli[14][4]=Beta2


        ObsFromPauli[0][15]=Beta12
        ObsFromPauli[1][15]=-Beta12
        ObsFromPauli[2][15]=-Beta12
        ObsFromPauli[3][14]=-Beta12
        ObsFromPauli[4][10]=Beta12
        ObsFromPauli[5][6]=-Beta12
        ObsFromPauli[6][14]=Beta12
        ObsFromPauli[7][13]=Beta12
        ObsFromPauli[8][9]=-Beta12
        ObsFromPauli[9][5]=Beta12
        ObsFromPauli[10][13]=-Beta12
        ObsFromPauli[11][11]=-Beta12
        ObsFromPauli[12][11]=Beta12
        ObsFromPauli[13][7]=Beta12
        ObsFromPauli[14][7]=-Beta12

        duplicate/o ObsFromPauli ObsFromPauliM
        WAVE ObsFromPauliM

        MatrixOP/O PauliFromObs=Inv(ObsFromPauli)
        MatrixOP/O ST2QPauliVec= PauliFromObs x tempvec
        setscale/P x, 1, 1, ST2QPauliVec

        // Make generic copies
        duplicate/o ST2QPaulivec Pvec2Q
        duplicate/o ST2QPaulivec Pvec2QM

if(savenum>-1)
        // Save vectors to the tomopath
        ST2Qvecfilename1="ST2QvecM"+phase+".txt"
        ST2Qvecfilename2="ST2QvecM"+".txt"
        Save/O/G/M="\r\n"/P=tomopath ST2Qvec as ST2Qvecfilename1
        Save/O/G/M="\r\n"/P=tomopath ST2Qvec as ST2Qvecfilename2

        if(savenum==1)
            ST2Qvecfilename1="ST2QvecM"+phase+"_"+num2str(filenum)+".txt"
            ST2Qvecfilename2="ST2QvecM"+"_"+num2str(filenum)+".txt"
            Save/O/G/M="\r\n"/P=tomopath ST2Qvec as ST2Qvecfilename1
            Save/O/G/M="\r\n"/P=tomopath ST2Qvec as ST2Qvecfilename2
        endif

        ST2QPaulivecfilename1="ST2QPaulivecM"+phase+".txt"
        ST2QPaulivecfilename2="ST2QPaulivecM"+".txt"
        Save/O/G/M="\r\n"/P=tomopath ST2QPaulivec as ST2QPaulivecfilename1
        Save/O/G/M="\r\n"/P=tomopath ST2QPaulivec as ST2QPaulivecfilename2

        if(savenum==1)
            ST2QPaulivecfilename1="ST2QPaulivecM"+phase+"_"+num2str(filenum)+".txt"
            ST2QPaulivecfilename2="ST2QPaulivecM"+"_"+num2str(filenum)+".txt"
            Save/O/G/M="\r\n"/P=tomopath ST2QPaulivec as ST2QPaulivecfilename1
            Save/O/G/M="\r\n"/P=tomopath ST2QPaulivec as ST2QPaulivecfilename2
        endif
endif

        // Make copies of vectors with run specific names
        ST2QvecMname="ST2QvecM_"+fancynum2str(filenum)+"_"+phase+"_"+basename+"_"+datestr;
        duplicate/O ST2Qvec $ST2QvecMname
        WAVE thisST2QvecM=$ST2QvecMname

        ST2QPaulivecMname="ST2QPaulivecM_"+fancynum2str(filenum)+"_"+phase+"_"+basename+"_"+datestr;
        duplicate/O ST2QPaulivec $ST2QPaulivecMname
        WAVE thisST2QPaulivecM=$ST2QPaulivecMname

        PQ1vecD[1]=ST2QPaulivec[1];
        PQ1vecD[3]=ST2QPaulivec[2];
        PQ1vecD[5]=ST2QPaulivec[3];

        PQ2vecD[7]=ST2QPaulivec[4]
        PQ2vecD[9]=ST2QPaulivec[8]
        PQ2vecD[11]=ST2QPaulivec[12]

        PQ12vecD[13]=ST2QPaulivec[9]
        PQ12vecD[15]=ST2QPaulivec[13]
        PQ12vecD[17]=ST2QPaulivec[6]
        PQ12vecD[19]=ST2QPaulivec[14]
        PQ12vecD[21]=ST2QPaulivec[7]
        PQ12vecD[23]=ST2QPaulivec[11]
        PQ12vecD[25]=ST2QPaulivec[5]
        PQ12vecD[27]=ST2QPaulivec[10]
        PQ12vecD[29]=ST2QPaulivec[15]

    duplicate/o PQ1vecD PQ1vecDM
    duplicate/o PQ2vecD PQ2vecDM
    duplicate/o PQ12vecD PQ12vecDM

    Beta1=(BetaVec[0]+BetaVec[6])/2;
    Beta2=(BetaVec[1]+BetaVec[7])/2;
    Beta12=(BetaVec[2]+BetaVec[8])/2;

// DOING INVERSION USING HOME-GROWN METHOD
ST2QPauliVec=NaN
ST2QPauliVec[0]=1;                                                                                                                  //ii
ST2QPauliVec[1]=1/2*(thisST2QvecP(8)+thisST2QvecP(11))/(-2*BetaVec[0])+1/2*(thisST2QvecM(8)+thisST2QvecM(11))/(2*BetaVec[6]);           //xi
ST2QPauliVec[2]=1/2*(thisST2QvecP(4)+thisST2QvecP(7))/(2*BetaVec[0])+1/2*(thisST2QvecM(4)+thisST2QvecM(7))/(-2*BetaVec[6]);             //yi
ST2QPauliVec[3]=1/2*(thisST2QvecP(1)+thisST2QvecP(3))/(2*BetaVec[0])+1/2*(thisST2QvecM(1)+thisST2QvecM(3))/(2*BetaVec[6]);              //zi
ST2QPauliVec[4]=1/2*(thisST2QvecP(14)+thisST2QvecP(15))/(-2*BetaVec[1])+1/2*(thisST2QvecM(14)+thisST2QvecM(15))/(2*BetaVec[7]);         //ix
ST2QPauliVec[5]=(thisST2QvecP(10)+thisST2QvecM(10))/(2*Beta12);                                                                     //xx
ST2QPauliVec[6]=(thisST2QvecP(6)+thisST2QvecM(6))/(-2*Beta12);                                                                      //yx
ST2QPauliVec[7]=1/2*(-thisST2QvecP(14)+thisST2QvecP(15))/(2*BetaVec[2])+1/2*(thisST2QvecM(14)-thisST2QvecM(15))/(2*BetaVec[8]);         //zx
ST2QPauliVec[8]=1/2*(thisST2QvecP(12)+thisST2QvecP(13))/(2*BetaVec[1])+1/2*(thisST2QvecM(12)+thisST2QvecM(13))/(-2*BetaVec[7])          //iy
ST2QPauliVec[9]=(thisST2QvecP(9)+thisST2QvecM(9))/(-2*Beta12);                                                                      //xy
ST2QPauliVec[10]=(thisST2QvecP(5)+thisST2QvecM(5))/(2*Beta12);                                                                      //yy
ST2QPauliVec[11]=1/2*(thisST2QvecP(12)-thisST2QvecP(13))/(2*BetaVec[2])+1/2*(-thisST2QvecM(12)+thisST2QvecM(13))/(2*BetaVec[8]);            //zy
ST2QPauliVec[12]=1/2*(thisST2QvecP(1)+thisST2QvecP(2))/(2*BetaVec[1])+1/2*(thisST2QvecM(1)+thisST2QvecM(2))/(2*BetaVec[7])              //iz
ST2QPauliVec[13]=1/2*(-thisST2QvecP(8)+thisST2QvecP(11))/(2*BetaVec[2])+1/2*(thisST2QvecM(8)-thisST2QvecM(11))/(2*BetaVec[8]);          //xz
ST2QPauliVec[14]=1/2*(thisST2QvecP(4)-thisST2QvecP(7))/(2*BetaVec[2])+1/2*(-thisST2QvecM(4)+thisST2QvecM(7))/(2*BetaVec[8]);                //yz
ST2QPauliVec[15]=1/2*(thisST2QvecP(2)+thisST2QvecP(3))/(-2*BetaVec[2])+1/2*(thisST2QvecM(2)+thisST2QvecM(3))/(-2*BetaVec[8]);           //zz


// DOING INVERSION USING LEAST-SQUARES PSEUDO INVERSE

NVAR doleastsq

if(doleastsq==1)
    make/o/n=(32) tempvecALL
    tempvecALL[0,15]=thisST2QvecP[p+3];
    tempvecALL[16,31]=thisST2QvecM[p-16+3];

    make/o/n=(32,16) ObsFromPauliALL
    ObsFromPauliALL[0,15][]=ObsFromPauliP[p][q]
    ObsFromPauliALL[16,31][]=ObsFromPauliM[p-16][q]

    MatrixOP/O PauliFromObs=Inv(ObsFromPauliALL^t x ObsFromPauliALL) x obsFromPauliALL^t
    MatrixOP/O ST2QPauliVec= PauliFromObs x tempvecALL
    setscale/P x, 1, 1, ST2QPauliVec
endif


    // Make generic copies
    duplicate/o ST2QPaulivec Pvec2Q
    duplicate/o ST2QPaulivec Pvec2QH


    make/o/n=(16) PQ1vecH=NaN
    make/o/n=(16) PQ2vecH=Nan
    make/o/n=(16) PQ12vecH=Nan

    PQ1vecH[1]=ST2QPaulivec[1];
    PQ1vecH[2]=ST2QPaulivec[2];
    PQ1vecH[3]=ST2QPaulivec[3];

    PQ2vecH[4]=ST2QPaulivec[4]
    PQ2vecH[5]=ST2QPaulivec[8]
    PQ2vecH[6]=ST2QPaulivec[12]

    PQ12vecH[7]=ST2QPaulivec[9]
    PQ12vecH[8]=ST2QPaulivec[13]
    PQ12vecH[9]=ST2QPaulivec[6]
    PQ12vecH[10]=ST2QPaulivec[14]
    PQ12vecH[11]=ST2QPaulivec[7]
    PQ12vecH[12]=ST2QPaulivec[11]
    PQ12vecH[13]=ST2QPaulivec[5]
    PQ12vecH[14]=ST2QPaulivec[10]
    PQ12vecH[15]=ST2QPaulivec[15]

    setscale /p x, 1, 1, PQ1vecH
    setscale /p x, 1, 1, PQ2vecH
    setscale /p x, 1, 1, PQ12vecH

if(savenum>-1)
        ST2QPaulivecfilename1="ST2QPaulivecH"+phase+".txt"
        ST2QPaulivecfilename2="ST2QPaulivecH"+".txt"
        Save/O/G/M="\r\n"/P=tomopath ST2QPaulivec as ST2QPaulivecfilename1
        Save/O/G/M="\r\n"/P=tomopath ST2QPaulivec as ST2QPaulivecfilename2

        if(savenum==1)
            ST2QPaulivecfilename1="ST2QPaulivecH"+phase+"_"+num2str(filenum)+".txt"
            ST2QPaulivecfilename2="ST2QPaulivecH"+"_"+num2str(filenum)+".txt"
            Save/O/G/M="\r\n"/P=tomopath ST2QPaulivec as ST2QPaulivecfilename1
            Save/O/G/M="\r\n"/P=tomopath ST2QPaulivec as ST2QPaulivecfilename2
        endif
endif
        // Make copies of vectors with run specific name
        ST2QPaulivecHname="ST2QPaulivecH_"+fancynum2str(filenum)+"_"+phase+"_"+basename+"_"+datestr;
        duplicate/O ST2QPaulivec $ST2QPaulivecHname
        WAVE thisST2QPaulivecH=$ST2QPaulivecHname

    else
        if(bequiet==0)
            printf  "You'd like to analyze dual ST2Qruns, but it doesn't look like you have two of them back to back on this file!\r"
        endif
    endif

endif

    if(doplots==1)
        display;
    endif


    if(doplots>0)
        if(whichplot==0)
            appendtograph  normwave
            if(dotwo==1 && dimsize(normwave,0)==2*numExpsTotal)
                appendtograph normwavem
            endif
            label bottom "Experiment number"
        elseif(whichplot==1)
            appendtograph thisST2QvecP
            if(dotwo==1 && dimsize(normwave,0)==2*numExpsTotal)
                appendtograph thisST2QvecM
            endif
            label bottom "Raw measurement number"
            ModifyGraph zero(bottom)=1
            WAVE TicksST2Q
            WAVE TickNamesST2Q
            ModifyGraph userticks(bottom)={TicksST2Q,TickNamesST2Q}
        elseif(whichplot==2)
            appendtograph thisST2QPaulivecP
            if(dotwo==1 && dimsize(normwave,0)==2*numExpsTotal)
                appendtograph thisST2QPaulivecM
                appendtograph thisST2QPaulivecH
            endif
            label bottom "Pauli measurement"
            SetAxis left -1.3,1.3
            WAVE TicksST2QP
            WAVE TickNamesST2QP
            ModifyGraph userticks(bottom)={TicksST2QP,TickNamesST2QP}
        endif

        fixl(); autocolor(); gridon(2); dotize();
        fixg();
        ModifyGraph nticks(left)=5
        label left "Mean Value"
        if(donorm==1)
            SetAxis left -3,3
        endif
    endif



end



// Function: getBetas
// ------------------------------------
function getBetas(filenum, basename, phase, datestr, expStart, expStop)
variable filenum
string basename
string phase
string datestr
variable expStart
variable expStop

string expnamebase=fancynum2str(filenum)+"_"+phase+"2DPlot"+basename+"_"+datestr;

variable calSegments=20;

string thisexpname;
string normwavename;
variable offset
variable Beta1, Beta2, Beta12
variable dBeta1, dBeta2,dBeta12
variable M1, M2, M3, M4
variable dM1, dM2, dM3, dM4

NVAR bequiet

variable i=expStart;


Do
    thisexpname=expnamebase+"_ex"+num2str(i)+"_v";
    WAVE thisexp=$thisexpname
    //prepare the normalized vector
    normwavename=thisexpname+"n"
    duplicate/o thisexp $normwavename
    wave normwave=$normwavename

    wavestats/Q/R=[0,calSegments-1] thisexp
    offset=V_avg
    normwave-=offset

    wavestats/Q/R=[0,calSegments/4-1] normwave
    M1=V_avg
    dM1=V_sdev/sqrt(calSegments/4);
    wavestats/Q/R=[calSegments/4,2*calSegments/4-1] normwave
    M2=V_avg
    dM2=V_sdev/sqrt(calSegments/4);
    wavestats/Q/R=[2*calSegments/4,3*calSegments/4-1] normwave
    M3=V_avg
    dM3=V_sdev/sqrt(calSegments/4);
    wavestats/Q/R=[3*calSegments/4,4*calSegments/4-1] normwave
    M4=V_avg
    dM4=V_sdev/sqrt(calSegments/4);

    Beta1=(M1+M3)/2;
    Beta2=((M1+M2)/2);
    Beta12=((M1+M4)/2);

    dBeta1=sqrt(dM1^2+dM3^2)/2;
    dBeta2=sqrt(dM1^2+dM2^2)/2;
    dBeta12=sqrt(dM1^2+dM4^2)/2;

    if(bequiet==0)
        printf  "Beta1=%.4g +/- %.2g,   Beta2=%.4g +/- %.2g,    Beta12=%.4g +/- %.2g        (%.2f, %.2f, %.2f)   offset=%.4g\r", Beta1, dBeta1,  Beta2, dBeta2,   Beta12, dBeta12,   dBeta1/Beta1, dBeta2/Beta2, dBeta12/Beta12, offset
    endif

i+=1
while(i<=expStop)

    // Copy the Beta and dBeta values into the BetaVec wave used by other functions, such as analyzeRFsweep
    WAVE BetaVec
    BetaVec[0]=Beta1;
    BetaVec[1]=Beta2;
    BetaVec[2]=Beta12;
    BetaVec[3]=dBeta1;
    BetaVec[4]=dBeta2;
    BetaVec[5]=dBeta12;

end




// Function: Graph_DualFancyPauli
// ------------------------------------------------
Window Graph_DualFancyPauli() : Graph
    PauseUpdate; Silent 1       // building window...
    Display /W=(1605,37.25,2049,272) PQ1vecDP,PQ2vecDP,PQ12vecDP,PQ1vecDM,PQ2vecDM,PQ12vecDM
    ModifyGraph userticks(bottom)={TicksST2QP2dual,TickNamesST2QP2dual}
    ModifyGraph margin(top)=30,margin(right)=10
    ModifyGraph mode=5
    ModifyGraph rgb(PQ2vecDP)=(0,43520,65280),rgb(PQ12vecDP)=(36864,14592,58880),rgb(PQ1vecDM)=(39168,0,0)
    ModifyGraph rgb(PQ2vecDM)=(0,26112,39168),rgb(PQ12vecDM)=(19712,0,39168)
    ModifyGraph hbFill(PQ1vecDP)=6,hbFill(PQ2vecDP)=6,hbFill(PQ12vecDP)=6,hbFill(PQ1vecDM)=7
    ModifyGraph hbFill(PQ2vecDM)=7,hbFill(PQ12vecDM)=7
    ModifyGraph hBarNegFill(PQ1vecDM)=7,hBarNegFill(PQ2vecDM)=7,hBarNegFill(PQ12vecDM)=7
    ModifyGraph offset(PQ1vecDP)={-1,0},offset(PQ2vecDP)={-1,0},offset(PQ12vecDP)={-1,0}
    ModifyGraph offset(PQ1vecDM)={-1,0},offset(PQ2vecDM)={-1,0},offset(PQ12vecDM)={-1,0}
    ModifyGraph grid(left)=1
    ModifyGraph tick=2
    ModifyGraph zero(left)=2
    ModifyGraph mirror=1
    ModifyGraph fSize=12
    ModifyGraph lblMargin(left)=4
    ModifyGraph btLen=2
    Label left "Mean value"
    Label bottom "Pauli Measurement"
    SetAxis left -1.3,1.3
    Legend/C/N=text0/J/F=0/B=1/A=MC/X=4.39/Y=43.77/E=2 "\\s(PQ1vecDP)PQ1vecDP\\s(PQ2vecDP)PQ2vecDP\\s(PQ12vecDP)PQ12vecDP"
    AppendText "\\s(PQ1vecDM)PQ1vecDM\\s(PQ2vecDM)PQ2vecDM\\s(PQ12vecDM)PQ12vecDM"
EndMacro



// Function: Graph_FancyPauli
// ----------------------------------------
Window Graph_fancypauli() : Graph
    PauseUpdate; Silent 1       // building window...
    Display /W=(10.5,46.25,426,255.5) PQ1vec,PQ2vec,PQ12vec
    ModifyGraph userticks(bottom)={TicksST2QP2,TickNamesST2QP2}
    ModifyGraph margin(top)=20,margin(right)=10
    ModifyGraph mode=5
    ModifyGraph rgb(PQ1vec)=(65280,0,0),rgb(PQ2vec)=(0,43520,65280),rgb(PQ12vec)=(29440,0,58880)
    ModifyGraph hbFill=6
    ModifyGraph offset(PQ1vec)={-0.5,0},offset(PQ2vec)={-0.5,0},offset(PQ12vec)={-0.5,0}
    ModifyGraph grid(left)=1
    ModifyGraph tick=2
    ModifyGraph zero(left)=2
    ModifyGraph mirror=1
    ModifyGraph fSize=12
    ModifyGraph btLen=2
    Label left "Ensemble average"
    Label bottom "Pauli operator"
    SetAxis left -1.05,1.05
    SetAxis bottom 1.3,16.7
    Legend/C/N=text0/J/F=0/B=1/A=MC/X=5.05/Y=45.52/E=2 "\\Z12\\s(PQ1vec) PQ1vec \\Z12\\s(PQ2vec) PQ2vec \\Z12\\s(PQ12vec) PQ12vec"
EndMacro


// Function: PrepareST2QTicks()
// --------------------------------------------
function prepareST2QTicks()
//User defined ticks
make/o/n=19 TicksST2Q
 TicksST2Q[0]=-2;
 TicksST2Q[1]=-1;
 TicksST2Q[2]=0;
 TicksST2Q[3]=1;
 TicksST2Q[4]=2;
 TicksST2Q[5]=3;
 TicksST2Q[6]=4;
 TicksST2Q[7]=5;
 TicksST2Q[8]=6;
 TicksST2Q[9]=7;
 TicksST2Q[10]=8;
 TicksST2Q[11]=9;
 TicksST2Q[12]=10;
 TicksST2Q[13]=11;
 TicksST2Q[14]=12;
 TicksST2Q[15]=13;
 TicksST2Q[16]=14;
 TicksST2Q[17]=15;
TicksST2Q[18]=16;


make/o/n=(19,2)/T TickNamesST2Q
setDimLabel 1,1,'Tick Type',TickNamesST2Q
TickNamesST2Q[0][1]="Major"
TickNamesST2Q[1][1]="Major"
TickNamesST2Q[2][1]="Major"
TickNamesST2Q[3][1]="Major"
TickNamesST2Q[4][1]="Major"
TickNamesST2Q[5][1]="Major"
TickNamesST2Q[6][1]="Major"
TickNamesST2Q[7][1]="Major"
TickNamesST2Q[8][1]="Major"
TickNamesST2Q[9][1]="Major"
TickNamesST2Q[10][1]="Major"
TickNamesST2Q[11][1]="Major"
TickNamesST2Q[12][1]="Major"
TickNamesST2Q[13][1]="Major"
TickNamesST2Q[14][1]="Major"
TickNamesST2Q[15][1]="Major"
TickNamesST2Q[16][1]="Major"
TickNamesST2Q[17][1]="Major"
TickNamesST2Q[18][1]="Major"


TickNamesST2Q[0][0]="B1"
TickNamesST2Q[1][0]="B2"
TickNamesST2Q[2][0]="B12"
TickNamesST2Q[3][0]="1"
TickNamesST2Q[4][0]="2"
TickNamesST2Q[5][0]="3"
TickNamesST2Q[6][0]="4"
TickNamesST2Q[7][0]="5"
TickNamesST2Q[8][0]="6"
TickNamesST2Q[9][0]="7"
TickNamesST2Q[10][0]="8"
TickNamesST2Q[11][0]="9"
TickNamesST2Q[12][0]="10"
TickNamesST2Q[13][0]="11"
TickNamesST2Q[14][0]="12"
TickNamesST2Q[15][0]="13"
TickNamesST2Q[16][0]="14"
TickNamesST2Q[17][0]="15"
TickNamesST2Q[18][0]="16"


make/o/n=19 TicksST2Q2
 TicksST2Q2[0]=-2;
 TicksST2Q2[1]=-1;
 TicksST2Q2[2]=0;
 TicksST2Q2[3]=1;
 TicksST2Q2[4]=2;
 TicksST2Q2[5]=3;
 TicksST2Q2[6]=4;
 TicksST2Q2[7]=5;
 TicksST2Q2[8]=6;
 TicksST2Q2[9]=7;
 TicksST2Q2[10]=8;
 TicksST2Q2[11]=9;
 TicksST2Q2[12]=10;
 TicksST2Q2[13]=11;
 TicksST2Q2[14]=12;
 TicksST2Q2[15]=13;
 TicksST2Q2[16]=14;
 TicksST2Q2[17]=15;
TicksST2Q2[18]=16;


make/o/n=(19,2)/T TickNamesST2Q2
setDimLabel 1,1,'Tick Type',TickNamesST2Q2
TickNamesST2Q2[0][1]="Major"
TickNamesST2Q2[1][1]="Major"
TickNamesST2Q2[2][1]="Major"
TickNamesST2Q2[3][1]="Major"
TickNamesST2Q2[4][1]="Major"
TickNamesST2Q2[5][1]="Major"
TickNamesST2Q2[6][1]="Major"
TickNamesST2Q2[7][1]="Major"
TickNamesST2Q2[8][1]="Major"
TickNamesST2Q2[9][1]="Major"
TickNamesST2Q2[10][1]="Major"
TickNamesST2Q2[11][1]="Major"
TickNamesST2Q2[12][1]="Major"
TickNamesST2Q2[13][1]="Major"
TickNamesST2Q2[14][1]="Major"
TickNamesST2Q2[15][1]="Major"
TickNamesST2Q2[16][1]="Major"
TickNamesST2Q2[17][1]="Major"
TickNamesST2Q2[18][1]="Major"

TickNamesST2Q2[0][0]="B1"
TickNamesST2Q2[1][0]="B2"
TickNamesST2Q2[2][0]="B12"
TickNamesST2Q2[3][0]="Id"
TickNamesST2Q2[4][0]="LXp"
TickNamesST2Q2[5][0]="RXp"
TickNamesST2Q2[6][0]="LX90p"
TickNamesST2Q2[7][0]="LX90pRX90p"
TickNamesST2Q2[8][0]="LX90pRY90p"
TickNamesST2Q2[9][0]="LX90pRXp"
TickNamesST2Q2[10][0]="LY90p"
TickNamesST2Q2[11][0]="LY90pRX90p"
TickNamesST2Q2[12][0]="LY90pRY90p"
TickNamesST2Q2[13][0]="LY90pRXp"
TickNamesST2Q2[14][0]="RX90p"
TickNamesST2Q2[15][0]="LXpRX90p"
TickNamesST2Q2[16][0]="RY90p"
TickNamesST2Q2[17][0]="LXpRY90P"
TickNamesST2Q2[18][0]="II"

make/o/n=16 TicksST2QP
 TicksST2QP[0]=1;
 TicksST2QP[1]=2;
 TicksST2QP[2]=3;
 TicksST2QP[3]=4;
 TicksST2QP[4]=5;
 TicksST2QP[5]=6;
 TicksST2QP[6]=7;
 TicksST2QP[7]=8;
 TicksST2QP[8]=9;
 TicksST2QP[9]=10;
 TicksST2QP[10]=11;
 TicksST2QP[11]=12;
 TicksST2QP[12]=13;
 TicksST2QP[13]=14;
 TicksST2QP[14]=15;
 TicksST2QP[15]=16;

make/o/n=(16,2)/T TickNamesST2QP
setDimLabel 1,1,'Tick Type',TickNamesST2QP
TickNamesST2QP[0][1]="Major"
TickNamesST2QP[1][1]="Major"
TickNamesST2QP[2][1]="Major"
TickNamesST2QP[3][1]="Major"
TickNamesST2QP[4][1]="Major"
TickNamesST2QP[5][1]="Major"
TickNamesST2QP[6][1]="Major"
TickNamesST2QP[7][1]="Major"
TickNamesST2QP[8][1]="Major"
TickNamesST2QP[9][1]="Major"
TickNamesST2QP[10][1]="Major"
TickNamesST2QP[11][1]="Major"
TickNamesST2QP[12][1]="Major"
TickNamesST2QP[13][1]="Major"
TickNamesST2QP[14][1]="Major"
TickNamesST2QP[15][1]="Major"

TickNamesST2QP[0][0]="II"
TickNamesST2QP[1][0]="XI"
TickNamesST2QP[2][0]="YI"
TickNamesST2QP[3][0]="ZI"
TickNamesST2QP[4][0]="IX"
TickNamesST2QP[5][0]="XX"
TickNamesST2QP[6][0]="YX"
TickNamesST2QP[7][0]="ZX"
TickNamesST2QP[8][0]="IY"
TickNamesST2QP[9][0]="XY"
TickNamesST2QP[10][0]="YY"
TickNamesST2QP[11][0]="ZY"
TickNamesST2QP[12][0]="IZ"
TickNamesST2QP[13][0]="XZ"
TickNamesST2QP[14][0]="YZ"
TickNamesST2QP[15][0]="ZZ"


make/o/n=16 TicksST2QP2
 TicksST2QP2[0]=1;
 TicksST2QP2[1]=2;
 TicksST2QP2[2]=3;
 TicksST2QP2[3]=4;
 TicksST2QP2[4]=5;
 TicksST2QP2[5]=6;
 TicksST2QP2[6]=7;
 TicksST2QP2[7]=8;
 TicksST2QP2[8]=9;
 TicksST2QP2[9]=10;
 TicksST2QP2[10]=11;
 TicksST2QP2[11]=12;
 TicksST2QP2[12]=13;
 TicksST2QP2[13]=14;
 TicksST2QP2[14]=15;
 TicksST2QP2[15]=16;


make/o/n=(16,2)/T TickNamesST2QP2
setDimLabel 1,1,'Tick Type',TickNamesST2QP2
TickNamesST2QP2[0][1]="Major"
TickNamesST2QP2[1][1]="Major"
TickNamesST2QP2[2][1]="Major"
TickNamesST2QP2[3][1]="Major"
TickNamesST2QP2[4][1]="Major"
TickNamesST2QP2[5][1]="Major"
TickNamesST2QP2[6][1]="Major"
TickNamesST2QP2[7][1]="Major"
TickNamesST2QP2[8][1]="Major"
TickNamesST2QP2[9][1]="Major"
TickNamesST2QP2[10][1]="Major"
TickNamesST2QP2[11][1]="Major"
TickNamesST2QP2[12][1]="Major"
TickNamesST2QP2[13][1]="Major"
TickNamesST2QP2[14][1]="Major"
TickNamesST2QP2[15][1]="Major"

TickNamesST2QP2[0][0]="ii"
TickNamesST2QP2[1][0]="xi"
TickNamesST2QP2[2][0]="yi"
TickNamesST2QP2[3][0]="zi"
TickNamesST2QP2[4][0]="ix"
TickNamesST2QP2[5][0]="iy"
TickNamesST2QP2[6][0]="iz"
TickNamesST2QP2[7][0]="xy"
TickNamesST2QP2[8][0]="xz"
TickNamesST2QP2[9][0]="yx"
TickNamesST2QP2[10][0]="yz"
TickNamesST2QP2[11][0]="zx"
TickNamesST2QP2[12][0]="zy"
TickNamesST2QP2[13][0]="xx"
TickNamesST2QP2[14][0]="yy"
TickNamesST2QP2[15][0]="zz"


make/o/n=(16,2)/T TickNamesST2QP2dual
setDimLabel 1,1,'Tick Type',TickNamesST2QP2dual
TickNamesST2QP2dual[0][1]="Minor"
TickNamesST2QP2dual[1][1]="Major"
TickNamesST2QP2dual[2][1]="Major"
TickNamesST2QP2dual[3][1]="Major"
TickNamesST2QP2dual[4][1]="Major"
TickNamesST2QP2dual[5][1]="Major"
TickNamesST2QP2dual[6][1]="Major"
TickNamesST2QP2dual[7][1]="Major"
TickNamesST2QP2dual[8][1]="Major"
TickNamesST2QP2dual[9][1]="Major"
TickNamesST2QP2dual[10][1]="Major"
TickNamesST2QP2dual[11][1]="Major"
TickNamesST2QP2dual[12][1]="Major"
TickNamesST2QP2dual[13][1]="Major"
TickNamesST2QP2dual[14][1]="Major"
TickNamesST2QP2dual[15][1]="Major"

TickNamesST2QP2dual[0][0]="ii"
TickNamesST2QP2dual[1][0]="xi"
TickNamesST2QP2dual[2][0]="yi"
TickNamesST2QP2dual[3][0]="zi"
TickNamesST2QP2dual[4][0]="ix"
TickNamesST2QP2dual[5][0]="iy"
TickNamesST2QP2dual[6][0]="iz"
TickNamesST2QP2dual[7][0]="xy"
TickNamesST2QP2dual[8][0]="xz"
TickNamesST2QP2dual[9][0]="yx"
TickNamesST2QP2dual[10][0]="yz"
TickNamesST2QP2dual[11][0]="zx"
TickNamesST2QP2dual[12][0]="zy"
TickNamesST2QP2dual[13][0]="xx"
TickNamesST2QP2dual[14][0]="yy"
TickNamesST2QP2dual[15][0]="zz"

make/o/n=16 TicksST2QP2dual
TicksST2QP2dual[]=1+2*p;

make/o/n=16 TicksST2QP3
 TicksST2QP3=x;

make/o/n=(16,2)/T TickNamesST2QP3
setDimLabel 1,1,'Tick Type',TickNamesST2QP3
TickNamesST2QP3[0][1]="Major"
TickNamesST2QP3[1][1]="Major"
TickNamesST2QP3[2][1]="Major"
TickNamesST2QP3[3][1]="Major"
TickNamesST2QP3[4][1]="Major"
TickNamesST2QP3[5][1]="Major"
TickNamesST2QP3[6][1]="Major"
TickNamesST2QP3[7][1]="Major"
TickNamesST2QP3[8][1]="Major"
TickNamesST2QP3[9][1]="Major"
TickNamesST2QP3[10][1]="Major"
TickNamesST2QP3[11][1]="Major"
TickNamesST2QP3[12][1]="Major"
TickNamesST2QP3[13][1]="Major"
TickNamesST2QP3[14][1]="Major"
TickNamesST2QP3[15][1]="Major"


TickNamesST2QP3[0][0]="II"
TickNamesST2QP3[1][0]="XI"
TickNamesST2QP3[2][0]="YI"
TickNamesST2QP3[3][0]="ZI"
TickNamesST2QP3[4][0]="IX"
TickNamesST2QP3[5][0]="IY"
TickNamesST2QP3[6][0]="IZ"
TickNamesST2QP3[7][0]="XX"
TickNamesST2QP3[8][0]="XY"
TickNamesST2QP3[9][0]="XZ"
TickNamesST2QP3[10][0]="YX"
TickNamesST2QP3[11][0]="YY"
TickNamesST2QP3[12][0]="YZ"
TickNamesST2QP3[13][0]="ZX"
TickNamesST2QP3[14][0]="ZY"
TickNamesST2QP3[15][0]="ZZ"

end


// function: getCHSHraw
// --------------------------------
function getCHSHraw(datestr, base, phase,num, type,signloc,rotsign)
string datestr
string base
string phase
variable num
string type
variable signloc
string rotsign

variable CHSHval

make/o/n=(16)/D WeightVec=0;


string thisST2QPaulivecname

if(stringmatch("xyxy",type)==1)
    WeightVec[5]=(-1)^(signloc==0)
    WeightVec[9]=(-1)^(signloc==1)
    WeightVec[6]=(-1)^(signloc==2)
    WeightVec[10]=(-1)^(signloc==3)
elseif(stringmatch("xzxz",type)==1)
    WeightVec[5]=(-1)^(signloc==0)
    WeightVec[13]=(-1)^(signloc==1)
    WeightVec[7]=(-1)^(signloc==2)
    WeightVec[15]=(-1)^(signloc==3)
 elseif(stringmatch("yzyz",type)==1)
    WeightVec[10]=(-1)^(signloc==0)
    WeightVec[14]=(-1)^(signloc==1)
    WeightVec[11]=(-1)^(signloc==2)
    WeightVec[15]=(-1)^(signloc==3)
endif

    thisST2QPaulivecname="ST2QPaulivec"+rotsign+"_"+fancynum2str(num)+"_"+Phase+"_"+base+"_"+datestr
    WAVE thisST2QPaulivec=$thisST2QPaulivecname
    if(stringmatch("xyxy",type)==1)
        CHSHval=(-1)^(signloc==0)*thisST2QPaulivec[5]+(-1)^(signloc==1)*thisST2QPaulivec[9]+(-1)^(signloc==2)*thisST2QPaulivec[6]+(-1)^(signloc==3)*thisST2QPaulivec[10]
    elseif(stringmatch("xzxz",type)==1)
        CHSHval=(-1)^(signloc==0)*thisST2QPaulivec[5]+(-1)^(signloc==1)*thisST2QPaulivec[13]+(-1)^(signloc==2)*thisST2QPaulivec[7]+(-1)^(signloc==3)*thisST2QPaulivec[15]
        //CHSHval=(-1)^(signloc==1)*thisST2QPaulivec[13]+(-1)^(signloc==3)*thisST2QPaulivec[15]
    elseif(stringmatch("yzyz",type)==1)
        CHSHval=(-1)^(signloc==0)*thisST2QPaulivec[10]+(-1)^(signloc==1)*thisST2QPaulivec[14]+(-1)^(signloc==2)*thisST2QPaulivec[11]+(-1)^(signloc==3)*thisST2QPaulivec[15]
    elseif(stringmatch("conc",type)==1)
        CHSHval=getRawConcurrence(datestr, base, phase, num, rotsign)
    elseif(stringmatch("witn",type)==1)
        CHSHval=getRawWitness(datestr, base, phase, num, rotsign,signloc)
    endif

return CHSHval

end


// Function: getRawWitness
// -------------------------------------------
function getRawwitness(datestr, base, phase, num, rotsign,whichone)
string datestr
string base
string phase
variable num
string rotsign
variable whichone


string thisST2QPaulivecname="ST2QPaulivec"+rotsign+"_"+fancynum2str(num)+"_"+Phase+"_"+base+"_"+datestr
WAVE Pvec=$thisST2QPaulivecname

variable Witness

if(whichone==0)
Witness=-2*1/4*(PVec[0]-Pvec[5]+Pvec[10]-Pvec[15])
elseif(whichone==1)
Witness=-2*1/4*(Pvec[0]+Pvec[5]-Pvec[10]-Pvec[15])
elseif(whichone==2)
Witness=-2*1/4*(Pvec[0]-Pvec[5]-Pvec[10]+Pvec[15])
elseif(whichone==3)
Witness=-2*1/4*(Pvec[0]+Pvec[5]+Pvec[10]+Pvec[15])
endif

return Witness
end



// Function: getRawConcurrence
// -------------------------------------------
// This formula of concurrences is only valid for pure states.
function getRawConcurrence(datestr, base, phase, num, rotsign)
string datestr
string base
string phase
variable num
string rotsign


string thisST2QPaulivecname="ST2QPaulivec"+rotsign+"_"+fancynum2str(num)+"_"+Phase+"_"+base+"_"+datestr
WAVE thisST2QPaulivec=$thisST2QPaulivecname

variable Q=thisST2QPaulivec[5]^2+thisST2QPaulivec[6]^2+thisST2QPaulivec[7]^2+thisST2QPaulivec[9]^2+thisST2QPaulivec[10]^2+thisST2QPaulivec[11]^2+thisST2QPaulivec[13]^2+thisST2QPaulivec[14]^2+thisST2QPaulivec[15]^2;
variable concurrence=sqrt((Q-1)/2)
return concurrence

end


//function: getallCHSHrawvecs
// -----------------------------------------
function getallCHSHrawvecs(datestr, base, phase, startrun,endrun, startval, endval,[rotsign,dnum])
string datestr
string base
string phase
variable startrun
variable endrun
variable startval
variable endval
string rotsign
variable dnum;

if(ParamisDefault(rotsign))
    rotsign=""
endif

if(ParamisDefault(dnum))
    dnum=1;
endif


NVAR doplots
variable olddoplots=doplots

doplots=0;
getCHSHrawvec(datestr,base,phase,startrun,endrun,startval,endval,"xyxy",0,rotsign,dnum)
getCHSHrawvec(datestr,base,phase,startrun,endrun,startval,endval,"xyxy",1,rotsign,dnum)
getCHSHrawvec(datestr,base,phase,startrun,endrun,startval,endval,"xyxy",2,rotsign,dnum)
getCHSHrawvec(datestr,base,phase,startrun,endrun,startval,endval,"xyxy",3,rotsign,dnum)

getCHSHrawvec(datestr,base,phase,startrun,endrun,startval,endval,"xzxz",0,rotsign,dnum)
getCHSHrawvec(datestr,base,phase,startrun,endrun,startval,endval,"xzxz",1,rotsign,dnum)
getCHSHrawvec(datestr,base,phase,startrun,endrun,startval,endval,"xzxz",2,rotsign,dnum)
getCHSHrawvec(datestr,base,phase,startrun,endrun,startval,endval,"xzxz",3,rotsign,dnum)

getCHSHrawvec(datestr,base,phase,startrun,endrun,startval,endval,"yzyz",0,rotsign,dnum)
getCHSHrawvec(datestr,base,phase,startrun,endrun,startval,endval,"yzyz",1,rotsign,dnum)
getCHSHrawvec(datestr,base,phase,startrun,endrun,startval,endval,"yzyz",2,rotsign,dnum)
getCHSHrawvec(datestr,base,phase,startrun,endrun,startval,endval,"yzyz",3,rotsign,dnum)

getCHSHrawvec(datestr,base,phase,startrun,endrun,startval,endval,"conc",3,rotsign,dnum)

getCHSHrawvec(datestr,base,phase,startrun,endrun,startval,endval,"witn",0,rotsign,dnum)
getCHSHrawvec(datestr,base,phase,startrun,endrun,startval,endval,"witn",1,rotsign,dnum)
getCHSHrawvec(datestr,base,phase,startrun,endrun,startval,endval,"witn",2,rotsign,dnum)
getCHSHrawvec(datestr,base,phase,startrun,endrun,startval,endval,"witn",3,rotsign,dnum)


doplots=olddoplots

end

// function: getCHSHrawvec
// -------------------------------------
function getCHSHrawvec(datestr, base, phase,startrun, endrun,startval, endval, type,signloc,rotsign,dnum)
string datestr
string base
string phase
variable startrun
variable endrun
variable startval
variable endval
string type
variable signloc
string rotsign
variable dnum


string Xlabel="Rotation angle (deg)"
if (startval==0 && endval==0)
    startval=startrun;
    endval=endrun;
    Xlabel="File number"
endif

variable startnum=min(startrun,endrun);
variable endnum=max(startrun,endrun);
variable numruns=floor((endnum-startnum)/dnum)+1;


make/o/n=(16)/D WeightVec=0;
string CHSHstr;

string thisST2QPaulivecname

if(stringmatch("xyxy",type)==1)
    WeightVec[5]=(-1)^(signloc==0)
    WeightVec[9]=(-1)^(signloc==1)
    WeightVec[6]=(-1)^(signloc==2)
    WeightVec[10]=(-1)^(signloc==3)
    CHSHstr="CHSHxyxy"+num2str(signloc);
elseif(stringmatch("xzxz",type)==1)
    WeightVec[5]=(-1)^(signloc==0)
    WeightVec[13]=(-1)^(signloc==1)
    WeightVec[7]=(-1)^(signloc==2)
    WeightVec[15]=(-1)^(signloc==3)
    CHSHstr="CHSHxzxz"+num2str(signloc);
 elseif(stringmatch("yzyz",type)==1)
    WeightVec[10]=(-1)^(signloc==0)
    WeightVec[14]=(-1)^(signloc==1)
    WeightVec[11]=(-1)^(signloc==2)
    WeightVec[15]=(-1)^(signloc==3)
    CHSHstr="CHSHyzyz"+num2str(signloc);
 elseif(stringmatch("conc",type)==1)
    CHSHstr="Conc";
  elseif(stringmatch("witn",type)==1)
    CHSHstr="Witn"+num2str(signloc);
endif

string CHSHvecname2="raw"+CHSHstr+"vec"
make/o/n=(1) $CHSHvecname2=NaN;

string CHSHvecname="raw"+CHSHstr+"_"+datestr+"_"+phase+"_"+fancynum2str(startnum)+"_"+fancynum2str(endnum)
make/o/n=(numruns)  $CHSHvecname=NaN
wave CHSHvec=$CHSHvecname
setscale/I x, startval, endval, CHSHvec

variable i=0;
variable currnum;

Do
    currnum=startnum+i*dnum;
    thisST2QPaulivecname="ST2QPaulivec"+rotsign+"_"+fancynum2str(currnum)+"_"+Phase+"_"+base+"_"+datestr
    WAVE thisST2QPaulivec=$thisST2QPaulivecname
    CHSHvec[i]= getCHSHraw(datestr, base, phase,currnum, type,signloc,rotsign)

    //if(stringmatch("xyxy",type)==1)
    //  CHSHvec[i]=(-1)^(signloc==0)*thisST2QPaulivec[5]+(-1)^(signloc==1)*thisST2QPaulivec[9]+(-1)^(signloc==2)*thisST2QPaulivec[6]+(-1)^(signloc==3)*thisST2QPaulivec[10]
    //elseif(stringmatch("xzxz",type)==1)
    //  CHSHvec[i]=(-1)^(signloc==0)*thisST2QPaulivec[5]+(-1)^(signloc==1)*thisST2QPaulivec[13]+(-1)^(signloc==2)*thisST2QPaulivec[7]+(-1)^(signloc==3)*thisST2QPaulivec[15]
    //elseif(stringmatch("yzyz",type)==1)
    //  CHSHvec[i]=(-1)^(signloc==0)*thisST2QPaulivec[10]+(-1)^(signloc==1)*thisST2QPaulivec[14]+(-1)^(signloc==2)*thisST2QPaulivec[11]+(-1)^(signloc==3)*thisST2QPaulivec[15]
    //  //CHSHvec[i]=MatrixDot(WeightVec,thisST2QPaulivec);
    //elseif(stringmatch("conc",type)==1)
    //  CHSHvec[i]=getRawConcurrence(datestr, base, phase, currnum, rotsign)
    //endif
    i+=1
While(i<numRuns);

// Make generic wave
duplicate/o CHSHvec $CHSHvecname2
wave genericCHSHvec=$CHSHvecname2

// plot according to doplots
NVAR doplots
if(doplots==1)
    display
endif
if(doplots>=1)
    appendtograph genericCHSHvec
    fixg();
    label bottom Xlabel;
    //lbllchsh();
    gridoff();
    SetAxis left -3,3
    SetAxis bottom startval, endval
    ModifyGraph userticks(left)={ticksCHSH,ticknamesCHSH}
    gridon(1)
    ModifyGraph gridRGB(left)=(0,0,0)
    ModifyGraph margin(right)=10
endif

end


Window Graph_rawandtheoxyxy() : Graph
    PauseUpdate; Silent 1       // building window...
    Display /W=(5.25,39.5,225.75,239.75) rawCHSHxyxy0vec,rawCHSHxyxy1vec,rawCHSHxyxy2vec
    AppendToGraph rawCHSHxyxy3vec,theoCHSHxyxy[0][*],theoCHSHxyxy[1][*],theoCHSHxyxy[2][*]
    AppendToGraph theoCHSHxyxy[3][*]
    ModifyGraph userticks(left)={TicksCHSH,TickNamesCHSH}
    ModifyGraph margin(left)=37,margin(top)=35,margin(right)=10
    ModifyGraph mode(rawCHSHxyxy0vec)=4,mode(rawCHSHxyxy1vec)=4,mode(rawCHSHxyxy2vec)=4
    ModifyGraph mode(rawCHSHxyxy3vec)=4
    ModifyGraph marker(rawCHSHxyxy0vec)=19,marker(rawCHSHxyxy1vec)=19,marker(rawCHSHxyxy2vec)=19
    ModifyGraph marker(rawCHSHxyxy3vec)=19
    ModifyGraph rgb(rawCHSHxyxy0vec)=(65280,0,0),rgb(rawCHSHxyxy1vec)=(65280,32512,16384)
    ModifyGraph rgb(rawCHSHxyxy2vec)=(0,52224,26368),rgb(rawCHSHxyxy3vec)=(0,34816,52224)
    ModifyGraph rgb(theoCHSHxyxy)=(65280,0,0),rgb(theoCHSHxyxy#1)=(65280,32512,16384)
    ModifyGraph rgb(theoCHSHxyxy#2)=(0,52224,26368),rgb(theoCHSHxyxy#3)=(0,34816,52224)
    ModifyGraph msize(rawCHSHxyxy0vec)=1.5,msize(rawCHSHxyxy1vec)=1.5,msize(rawCHSHxyxy2vec)=1.5
    ModifyGraph msize(rawCHSHxyxy3vec)=1.5
    ModifyGraph grid(left)=1
    ModifyGraph tick=2
    ModifyGraph zero(left)=2
    ModifyGraph mirror=1
    ModifyGraph fSize=10
    ModifyGraph gridRGB(left)=(0,0,0)
    ModifyGraph btLen=2
    Label left "CHSH mean value "
    Label bottom "File number"
    SetAxis left -3,3
    Legend/C/N=text0/J/F=0/B=1/A=MC/X=3.40/Y=43.07/E=2
    AppendText "\\Z08\\s(rawCHSHxyxy0vec) \\Z08\\s(rawCHSHxyxy1vec) \\Z08\\s(rawCHSHxyxy2vec) \\Z08\\s(rawCHSHxyxy3vec) rawCHSHxyxy"
    AppendText "\\s(theoCHSHxyxy) \\s(theoCHSHxyxy#1) \\s(theoCHSHxyxy#2) \\s(theoCHSHxyxy#3) theoCHSHxyxy"
EndMacro

Window Graph_rawandtheoxzxz() : Graph
    PauseUpdate; Silent 1       // building window...
    Display /W=(306,41,526.5,241.25) rawCHSHxzxz0vec,rawCHSHxzxz1vec,rawCHSHxzxz2vec
    AppendToGraph rawCHSHxzxz3vec,theoCHSHxzxz[0][*],theoCHSHxzxz[1][*],theoCHSHxzxz[2][*]
    AppendToGraph theoCHSHxzxz[3][*]
    ModifyGraph userticks(left)={TicksCHSH,TickNamesCHSH}
    ModifyGraph userticks(bottom)={TicksCHSHAngles,TickNamesCHSHAngles}
    ModifyGraph margin(left)=37,margin(top)=35,margin(right)=10
    ModifyGraph mode(rawCHSHxzxz0vec)=4,mode(rawCHSHxzxz1vec)=4,mode(rawCHSHxzxz2vec)=4
    ModifyGraph mode(rawCHSHxzxz3vec)=4
    ModifyGraph marker(rawCHSHxzxz0vec)=19,marker(rawCHSHxzxz1vec)=19,marker(rawCHSHxzxz2vec)=19
    ModifyGraph marker(rawCHSHxzxz3vec)=19
    ModifyGraph rgb(rawCHSHxzxz0vec)=(65280,0,0),rgb(rawCHSHxzxz1vec)=(65280,32512,16384)
    ModifyGraph rgb(rawCHSHxzxz2vec)=(0,52224,0),rgb(rawCHSHxzxz3vec)=(0,34816,52224)
    ModifyGraph rgb(theoCHSHxzxz)=(65280,0,0),rgb(theoCHSHxzxz#1)=(65280,21760,0),rgb(theoCHSHxzxz#2)=(0,52224,0)
    ModifyGraph rgb(theoCHSHxzxz#3)=(0,34816,52224)
    ModifyGraph msize(rawCHSHxzxz0vec)=1.5,msize(rawCHSHxzxz1vec)=1.5,msize(rawCHSHxzxz2vec)=1.5
    ModifyGraph msize(rawCHSHxzxz3vec)=1.5
    ModifyGraph grid(left)=1
    ModifyGraph tick=2
    ModifyGraph zero(left)=2
    ModifyGraph mirror=1
    ModifyGraph noLabel(left)=2
    ModifyGraph fSize=10
    ModifyGraph gridRGB(left)=(0,0,0)
    ModifyGraph btLen=2
    Label left "CHSH mean value "
    Label bottom "Rotation angle (deg)"
    SetAxis left -3,3
    SetAxis bottom -210,210
    ShowInfo
    Legend/C/N=text0/J/F=0/B=1/A=MC/X=5.44/Y=42.32/E=2
    AppendText "\\Z08\\s(rawCHSHxzxz0vec) \\Z08\\s(rawCHSHxzxz1vec) \\Z08\\s(rawCHSHxzxz2vec) \\Z08\\s(rawCHSHxzxz3vec) rawCHSHxzxz"
    AppendText "\\s(theoCHSHxzxz) \\s(theoCHSHxzxz#1) \\s(theoCHSHxzxz#2) \\s(theoCHSHxzxz#3) theoCHSHxzxz"
EndMacro

Window Graph_rawandtheoyzyz() : Graph
    PauseUpdate; Silent 1       // building window...
    Display /W=(458.25,38,678.75,238.25) rawCHSHyzyz0vec,rawCHSHyzyz1vec,rawCHSHyzyz2vec
    AppendToGraph rawCHSHyzyz3vec,theoCHSHyzyz[0][*],theoCHSHyzyz[1][*],theoCHSHyzyz[2][*]
    AppendToGraph theoCHSHyzyz[3][*]
    ModifyGraph userticks(left)={TicksCHSH,TickNamesCHSH}
    ModifyGraph margin(left)=37,margin(top)=35,margin(right)=10
    ModifyGraph mode(rawCHSHyzyz0vec)=4,mode(rawCHSHyzyz1vec)=4,mode(rawCHSHyzyz2vec)=4
    ModifyGraph mode(rawCHSHyzyz3vec)=4
    ModifyGraph marker(rawCHSHyzyz0vec)=19,marker(rawCHSHyzyz1vec)=19,marker(rawCHSHyzyz2vec)=19
    ModifyGraph marker(rawCHSHyzyz3vec)=19
    ModifyGraph rgb(rawCHSHyzyz0vec)=(65280,0,0),rgb(rawCHSHyzyz1vec)=(65280,21760,0)
    ModifyGraph rgb(rawCHSHyzyz2vec)=(0,52224,0),rgb(rawCHSHyzyz3vec)=(0,34816,52224)
    ModifyGraph rgb(theoCHSHyzyz)=(65280,0,0),rgb(theoCHSHyzyz#1)=(65280,21760,0),rgb(theoCHSHyzyz#2)=(0,52224,0)
    ModifyGraph rgb(theoCHSHyzyz#3)=(0,34816,52224)
    ModifyGraph msize(rawCHSHyzyz0vec)=1.5,msize(rawCHSHyzyz1vec)=1.5,msize(rawCHSHyzyz2vec)=1.5
    ModifyGraph msize(rawCHSHyzyz3vec)=1.5
    ModifyGraph grid(left)=1
    ModifyGraph tick=2
    ModifyGraph zero(left)=2
    ModifyGraph mirror=1
    ModifyGraph noLabel(left)=2
    ModifyGraph fSize=10
    ModifyGraph gridRGB(left)=(0,0,0)
    ModifyGraph btLen=2
    Label left "CHSH mean value "
    Label bottom "File number"
    SetAxis left -3,3
    Legend/C/N=text0/J/F=0/B=1/A=MC/X=4.76/Y=42.70/E=2
    AppendText "\\Z08\\s(rawCHSHyzyz0vec) \\Z08\\s(rawCHSHyzyz1vec) \\Z08\\s(rawCHSHyzyz2vec) \\Z08\\s(rawCHSHyzyz3vec) rawCHSHyzyz"
    AppendText "\\s(theoCHSHyzyz) \\s(theoCHSHyzyz#1) \\s(theoCHSHyzyz#2) \\s(theoCHSHyzyz#3) theoCHSHyzyz"
EndMacro

Window Graph_RawandTheoWitness() : Graph
    PauseUpdate; Silent 1       // building window...
    Display /W=(232.5,260,453.75,457.25) rawWitn0vec,rawWitn1vec,rawWitn2vec,rawWitn3vec
    AppendToGraph theoWitness[0][*],theoWitness[1][*],theoWitness[2][*],theoWitness[3][*]
    AppendToGraph rawFidvec
    ModifyGraph userticks(bottom)={TicksCHSHAngles,TickNamesCHSHAngles}
    ModifyGraph margin(top)=35,margin(right)=10
    ModifyGraph mode(rawWitn0vec)=4,mode(rawWitn1vec)=4,mode(rawWitn2vec)=4,mode(rawWitn3vec)=4
    ModifyGraph mode(rawFidvec)=4
    ModifyGraph marker(rawWitn0vec)=19,marker(rawWitn1vec)=19,marker(rawWitn2vec)=19
    ModifyGraph marker(rawWitn3vec)=19,marker(rawFidvec)=8
    ModifyGraph rgb(rawWitn0vec)=(65280,0,0),rgb(rawWitn1vec)=(65280,43520,0),rgb(rawWitn2vec)=(0,52224,0)
    ModifyGraph rgb(rawWitn3vec)=(0,16320,65280),rgb(theoWitness)=(65280,0,0),rgb(theoWitness#1)=(65280,43520,0)
    ModifyGraph rgb(theoWitness#2)=(0,52224,0),rgb(theoWitness#3)=(0,16320,65280),rgb(rawFidvec)=(0,0,0)
    ModifyGraph msize(rawWitn0vec)=1.5,msize(rawWitn1vec)=1.5,msize(rawWitn2vec)=1.5
    ModifyGraph msize(rawWitn3vec)=1.5,msize(rawFidvec)=1.5
    ModifyGraph grid(left)=1
    ModifyGraph tick=2
    ModifyGraph zero(left)=2
    ModifyGraph mirror=1
    ModifyGraph fSize=10
    ModifyGraph btLen=2
    Label left "Witness"
    Label bottom "Rotation angle [deg]"
    SetAxis left -1.2,1.2
    Legend/C/N=text0/J/F=0/B=1/A=MC/X=7.12/Y=42.21/E=2 "\\Z08\\s(rawWitn0vec) \\Z08\\s(rawWitn1vec) \\Z08\\s(rawWitn2vec) \\Z08\\s(rawWitn3vec) rawWitn#vec"
    AppendText "\\s(theoWitness) \\s(theoWitness#1) \\s(theoWitness#2) \\s(theoWitness#3) theoWitness\r\\s(rawFidvec) rawFidvec"
EndMacro

Window Graph_CHSHhistogram() : Graph
    PauseUpdate; Silent 1       // building window...
    Display /W=(154.5,299.75,870,476) HistCHSHxzxz0,HistCHSHxzxz2,HistCHSHxzxz3,HistCHSHxzxz1
    ModifyGraph userticks(bottom)={CHSHyticks,CHSHtickText}
    ModifyGraph margin(right)=10
    ModifyGraph mode=5
    ModifyGraph rgb(HistCHSHxzxz2)=(0,52224,0),rgb(HistCHSHxzxz3)=(0,15872,65280),rgb(HistCHSHxzxz1)=(65280,32512,16384)
    ModifyGraph hbFill=4
    ModifyGraph grid=1
    ModifyGraph tick=2
    ModifyGraph zero(left)=2
    ModifyGraph mirror=1
    ModifyGraph fSize=10
    ModifyGraph lblMargin(left)=2
    ModifyGraph axOffset(left)=1.85714
    ModifyGraph btLen=2
    Label left "Counts"
    Label bottom "CHSH value"
    SetAxis bottom -3,3
    Legend/C/N=text0/J/F=0/B=1/A=MC/X=13.31/Y=37.87/E=2
    AppendText "\\s(HistCHSHxzxz0) HistCHSHxzxz0 \\s(HistCHSHxzxz1) HistCHSHxzxz1 \\s(HistCHSHxzxz3) HistCHSHxzxz3 \\s(HistCHSHxzxz2) HistCHSHxzxz2"
EndMacro


// Function: GetRawFidelity
// -------------------------------------
function GetRawFidelity(datestr, base, rotsign, phase,startnum, endnum, [waittime])
string datestr
string base
string rotsign
string phase
variable startnum
variable endnum
variable waittime

string genericVecname="rawFidvec"
make/o/n=1 $genericVecname=Nan;


if(ParamisDefault(waittime))
    waittime=.1;
endif

string allPauliRodname="PauliRod"+rotsign+"_"+datestr+"_"+base+"_"+phase+"_"+num2str(startnum)+"_"+num2str(endnum);
string FidVecname="FidVec"+"_"+datestr+"_"+base+"_"+phase+rotsign+"_"+num2str(startnum)+"_"+num2str(endnum);

WAVE theoPauliRod
WAVE allPaulirod=$allPaulirodname;

variable numPtsExp=dimsize(allPaulirod,0);
variable numPtsTheo=dimsize(theoPaulirod,1);
variable xoffset=dimoffset(allPaulirod,0);
variable xdelta=dimdelta(allPaulirod,0);

make/o/n=(numPtsExp) $FidVecName=NaN
WAVE FidVec=$FidVecName;
setscale/P x, xoffset, xdelta, FidVec


//print numptsexp, numptstheo

if(numPtsExp!=numPtsTheo)
    Abort "Theory and experiment Pauli rods not of equal length. Aborting now!"
endif

variable i=0;

make/o/n=(16) currPauliExp=NaN;
setscale/P x, 1, 1, currPauliExp
make/o/n=(16) currPauliTheo=NaN;
setscale/P x, 1, 1, currPauliTheo
make/o/n=(16) tempVec=NaN;
setscale/P x, 1, 1, currPauliTheo



Do
    currPauliExp[]=allPaulirod[i][p];
    currPauliTheo[]=theoPaulirod[p][i];
    tempVec[]=currPauliExp[p]*currPauliTheo[p];
    Fidvec[i]=sum(tempvec)/4;
    wait(waittime);
    doupdate;
    i+=1
While(i<numPtsExp);

// Make generic wave
duplicate/o Fidvec $genericVecname


NVAR doplots
if(doplots==1)
    display;
endif
if(doplots>0)
    appendtograph fidvec;
    fixg(); lblbra();
    label left "Raw Fidelity"

endif

end




// function: switchCHSHmleMovie
// ---------------------------------------------
function switchCHSHmleMovie(basename)
string basename

string prefix
string filename

prefix="mleCHSHxyxy0Vec"
filename=prefix+"_"+basename
WAVE thisone=$filename;
duplicate/o thisone $prefix

prefix="mleCHSHxyxy1Vec"
filename=prefix+"_"+basename
WAVE thisone=$filename;
duplicate/o thisone $prefix

prefix="mleCHSHxyxy2Vec"
filename=prefix+"_"+basename
WAVE thisone=$filename;
duplicate/o thisone $prefix

prefix="mleCHSHxyxy3Vec"
filename=prefix+"_"+basename
WAVE thisone=$filename;
duplicate/o thisone $prefix


prefix="mleCHSHxzxz0Vec"
filename=prefix+"_"+basename
WAVE thisone=$filename;
duplicate/o thisone $prefix

prefix="mleCHSHxzxz1Vec"
filename=prefix+"_"+basename
WAVE thisone=$filename;
duplicate/o thisone $prefix

prefix="mleCHSHxzxz2Vec"
filename=prefix+"_"+basename
WAVE thisone=$filename;
duplicate/o thisone $prefix

prefix="mleCHSHxzxz3Vec"
filename=prefix+"_"+basename
WAVE thisone=$filename;
duplicate/o thisone $prefix

prefix="mleCHSHyzyz0Vec"
filename=prefix+"_"+basename
WAVE thisone=$filename;
duplicate/o thisone $prefix

prefix="mleCHSHyzyz1Vec"
filename=prefix+"_"+basename
WAVE thisone=$filename;
duplicate/o thisone $prefix

prefix="mleCHSHyzyz2Vec"
filename=prefix+"_"+basename
WAVE thisone=$filename;
duplicate/o thisone $prefix

prefix="mleCHSHyzyz3Vec"
filename=prefix+"_"+basename
WAVE thisone=$filename;
duplicate/o thisone $prefix


prefix="mleTrRhoVec";
filename=prefix+"_"+basename
WAVE thisone=$filename;
duplicate/o thisone $prefix

prefix="mlePurityVec";
filename=prefix+"_"+basename
WAVE thisone=$filename;
duplicate/o thisone $prefix

prefix="mleConcurrenceVec";
filename=prefix+"_"+basename
WAVE thisone=$filename;
duplicate/o thisone $prefix

prefix="mleEntanglementVec";
filename=prefix+"_"+basename
WAVE thisone=$filename;
duplicate/o thisone $prefix

prefix="mleCHSHmaxVec";
filename=prefix+"_"+basename
WAVE thisone=$filename;
duplicate/o thisone $prefix

prefix="mleFid00vec";
filename=prefix+"_"+basename
WAVE thisone=$filename;
duplicate/o thisone $prefix

prefix="mleFid01vec";
filename=prefix+"_"+basename
WAVE thisone=$filename;
duplicate/o thisone $prefix

prefix="mleFid10vec";
filename=prefix+"_"+basename
WAVE thisone=$filename;
duplicate/o thisone $prefix

prefix="mleFid11vec";
filename=prefix+"_"+basename
WAVE thisone=$filename;
duplicate/o thisone $prefix

prefix="mleFidvec";
filename=prefix+"_"+basename
WAVE thisone=$filename;
duplicate/o thisone $prefix

prefix="mleQmlVec";
filename=prefix+"_"+basename
WAVE thisone=$filename;
duplicate/o thisone $prefix


prefix="mleQiVec";
filename=prefix+"_"+basename
WAVE thisone=$filename;
duplicate/o thisone $prefix

prefix="mleQqVec";
filename=prefix+"_"+basename
WAVE thisone=$filename;
duplicate/o thisone $prefix

end

// function getST2QstatsMLE
//----------------------------------------
function getST2QstatsMLE(basename,startval,endval)
string basename
variable startval
variable endval


string filename="mleCHSHxzVec";
string fullfilename=filename+".txt"
string IgorVecName=filename+"_"+basename
LoadWave/Q/O/P=tomopath/G/N=ST2Qwave fullfilename
WAVE ST2Qwave0
if(startval==0 && endval==0)
    startval=0;
    endval=dimsize(ST2Qwave0,0)-1;
endif
setscale/I x startval, endval, ST2Qwave0
duplicate/o ST2Qwave0 $filename
duplicate/o ST2Qwave0 $IgorVecName

filename="mleCHSHxyxy0Vec";
fullfilename=filename+".txt"
IgorVecName=filename+"_"+basename
LoadWave/Q/O/P=tomopath/G/N=ST2Qwave fullfilename
WAVE ST2Qwave0
if(startval==0 && endval==0)
    startval=0;
    endval=dimsize(ST2Qwave0,0)-1;
endif
setscale/I x startval, endval, ST2Qwave0
duplicate/o ST2Qwave0 $filename
duplicate/o ST2Qwave0 $IgorVecName


filename="mleCHSHxyxy1Vec";
fullfilename=filename+".txt";
IgorVecName=filename+"_"+basename
LoadWave/Q/O/P=tomopath/G/N=ST2Qwave fullfilename
WAVE ST2Qwave0
setscale/I x startval, endval, ST2Qwave0
duplicate/o ST2Qwave0 $filename
duplicate/o ST2Qwave0 $IgorVecName


filename="mleCHSHxyxy2Vec";
fullfilename=filename+".txt"
IgorVecName=filename+"_"+basename
LoadWave/Q/O/P=tomopath/G/N=ST2Qwave fullfilename
WAVE ST2Qwave0
setscale/I x startval, endval, ST2Qwave0
duplicate/o ST2Qwave0 $filename
duplicate/o ST2Qwave0 $IgorVecName


filename="mleCHSHxyxy3Vec";
fullfilename=filename+".txt"
IgorVecName=filename+"_"+basename
LoadWave/Q/O/P=tomopath/G/N=ST2Qwave fullfilename
WAVE ST2Qwave0
setscale/I x startval, endval, ST2Qwave0
duplicate/o ST2Qwave0 $filename
duplicate/o ST2Qwave0 $IgorVecName

filename="mleCHSHxzxz0Vec";
fullfilename=filename+".txt";
IgorVecName=filename+"_"+basename
LoadWave/Q/O/P=tomopath/G/N=ST2Qwave fullfilename
WAVE ST2Qwave0
setscale/I x startval, endval, ST2Qwave0
duplicate/o ST2Qwave0 $filename
duplicate/o ST2Qwave0 $IgorVecName

filename="mleCHSHxzxz1Vec";
fullfilename=filename+".txt";
IgorVecName=filename+"_"+basename;
LoadWave/Q/O/P=tomopath/G/N=ST2Qwave fullfilename
WAVE ST2Qwave0
setscale/I x startval, endval, ST2Qwave0
duplicate/o ST2Qwave0 $filename
duplicate/o ST2Qwave0 $IgorVecName


filename="mleCHSHxzxz2Vec";
fullfilename=filename+".txt";
IgorVecName=filename+"_"+basename;
LoadWave/Q/O/P=tomopath/G/N=ST2Qwave fullfilename
WAVE ST2Qwave0
setscale/I x startval, endval, ST2Qwave0
duplicate/o ST2Qwave0 $filename
duplicate/o ST2Qwave0 $IgorVecName



filename="mleCHSHxzxz3Vec";
fullfilename=filename+".txt";
IgorVecName=filename+"_"+basename;
LoadWave/Q/O/P=tomopath/G/N=ST2Qwave fullfilename
WAVE ST2Qwave0
setscale/I x startval, endval, ST2Qwave0
duplicate/o ST2Qwave0 $filename
duplicate/o ST2Qwave0 $IgorVecName


filename="mleCHSHyzyz0Vec";
fullfilename=filename+".txt";
IgorVecName=filename+"_"+basename;
LoadWave/Q/O/P=tomopath/G/N=ST2Qwave fullfilename
WAVE ST2Qwave0
setscale/I x startval, endval, ST2Qwave0
duplicate/o ST2Qwave0 $filename
duplicate/o ST2Qwave0 $IgorVecName


filename="mleCHSHyzyz1Vec";
fullfilename=filename+".txt";
IgorVecName=filename+"_"+basename;
LoadWave/Q/O/P=tomopath/G/N=ST2Qwave fullfilename
WAVE ST2Qwave0
setscale/I x startval, endval, ST2Qwave0
duplicate/o ST2Qwave0 $filename
duplicate/o ST2Qwave0 $IgorVecName


filename="mleCHSHyzyz2Vec";
fullfilename=filename+".txt";
IgorVecName=filename+"_"+basename;
LoadWave/Q/O/P=tomopath/G/N=ST2Qwave fullfilename
WAVE ST2Qwave0
setscale/I x startval, endval, ST2Qwave0
duplicate/o ST2Qwave0 $filename
duplicate/o ST2Qwave0 $IgorVecName


filename="mleCHSHyzyz3Vec";
fullfilename=filename+".txt";
IgorVecName=filename+"_"+basename;
LoadWave/Q/O/P=tomopath/G/N=ST2Qwave fullfilename
WAVE ST2Qwave0
setscale/I x startval, endval, ST2Qwave0
duplicate/o ST2Qwave0 $filename
duplicate/o ST2Qwave0 $IgorVecName

filename="mleTrRhoVec";
fullfilename=filename+".txt";
IgorVecName=filename+"_"+basename
LoadWave/Q/O/P=tomopath/G/N=ST2Qwave fullfilename
WAVE ST2Qwave0
setscale/I x startval, endval, ST2Qwave0
duplicate/o ST2Qwave0 $filename
duplicate/o ST2Qwave0 $IgorVecName

filename="mlePurityVec";
fullfilename=filename+".txt";
IgorVecName=filename+"_"+basename
LoadWave/Q/O/P=tomopath/G/N=ST2Qwave fullfilename
WAVE ST2Qwave0
setscale/I x startval, endval, ST2Qwave0
duplicate/o ST2Qwave0 $filename
duplicate/o ST2Qwave0 $IgorVecName



filename="mleConcurrenceVec";
fullfilename=filename+".txt";
IgorVecName=filename+"_"+basename;
LoadWave/Q/O/P=tomopath/G/N=ST2Qwave fullfilename
WAVE ST2Qwave0
setscale/I x startval, endval, ST2Qwave0
duplicate/o ST2Qwave0 $filename
duplicate/o ST2Qwave0 $IgorVecName

filename="mleEntanglementVec";
fullfilename=filename+".txt";
IgorVecName=filename+"_"+basename;
LoadWave/Q/O/P=tomopath/G/N=ST2Qwave fullfilename
WAVE ST2Qwave0
setscale/I x startval, endval, ST2Qwave0
duplicate/o ST2Qwave0 $filename
duplicate/o ST2Qwave0 $IgorVecName

filename="mleCHSHmaxVec";
fullfilename=filename+".txt";
IgorVecName=filename+"_"+basename;
LoadWave/Q/O/P=tomopath/G/N=ST2Qwave fullfilename
WAVE ST2Qwave0
setscale/I x startval, endval, ST2Qwave0
duplicate/o ST2Qwave0 $filename
duplicate/o ST2Qwave0 $IgorVecName

filename="mleFid00vec";
fullfilename=filename+".txt";
IgorVecName=filename+"_"+basename;
LoadWave/Q/O/P=tomopath/G/N=ST2Qwave fullfilename
WAVE ST2Qwave0
setscale/I x startval, endval, ST2Qwave0
duplicate/o ST2Qwave0 $filename
duplicate/o ST2Qwave0 $IgorVecName


filename="mleFid01vec";
fullfilename=filename+".txt";
IgorVecName=filename+"_"+basename;
LoadWave/Q/O/P=tomopath/G/N=ST2Qwave fullfilename
WAVE ST2Qwave0
setscale/I x startval, endval, ST2Qwave0
duplicate/o ST2Qwave0 $filename
duplicate/o ST2Qwave0 $IgorVecName


filename="mleFid10vec";
fullfilename=filename+".txt";
IgorVecName=filename+"_"+basename;
LoadWave/Q/O/P=tomopath/G/N=ST2Qwave fullfilename
WAVE ST2Qwave0
setscale/I x startval, endval, ST2Qwave0
duplicate/o ST2Qwave0 $filename
duplicate/o ST2Qwave0 $IgorVecName

filename="mleFid11vec";
fullfilename=filename+".txt";
IgorVecName=filename+"_"+basename;
LoadWave/Q/O/P=tomopath/G/N=ST2Qwave fullfilename
WAVE ST2Qwave0
setscale/I x startval, endval, ST2Qwave0
duplicate/o ST2Qwave0 $filename
duplicate/o ST2Qwave0 $IgorVecName

filename="mleFidvec";
fullfilename=filename+".txt";
IgorVecName=filename+"_"+basename;
LoadWave/Q/O/P=tomopath/G/N=ST2Qwave fullfilename
WAVE ST2Qwave0
setscale/I x startval, endval, ST2Qwave0
duplicate/o ST2Qwave0 $filename
duplicate/o ST2Qwave0 $IgorVecName

filename="mleQmlVec";
fullfilename=filename+".txt";
IgorVecName=filename+"_"+basename;
LoadWave/Q/O/P=tomopath/G/N=ST2Qwave fullfilename
WAVE ST2Qwave0
setscale/I x startval, endval, ST2Qwave0
duplicate/o ST2Qwave0 $filename
duplicate/o ST2Qwave0 $IgorVecName


filename="mleQiVec";
fullfilename=filename+".txt";
IgorVecName=filename+"_"+basename;
LoadWave/Q/O/P=tomopath/G/N=ST2Qwave fullfilename
WAVE ST2Qwave0
setscale/I x startval, endval, ST2Qwave0
duplicate/o ST2Qwave0 $filename
duplicate/o ST2Qwave0 $IgorVecName

filename="mleQqVec";
fullfilename=filename+".txt";
IgorVecName=filename+"_"+basename;
LoadWave/Q/O/P=tomopath/G/N=ST2Qwave fullfilename
WAVE ST2Qwave0
setscale/I x startval, endval, ST2Qwave0
duplicate/o ST2Qwave0 $filename
duplicate/o ST2Qwave0 $IgorVecName


NVAR bequiet
if(bequiet==0)

    WAVE mleFid00Vec;
    WAVE mleFid01Vec;
    WAVE mleFid10Vec;
    WAVE mleFid11Vec;

    WAVE mleFidVec;
    WAVE mlePurityVec;
    WAVE mleConcurrenceVec;
    WAVE mleEntanglementVec;
    WAVE mleQmlVec;
    WAVE mleQiVec;
    WAVE mleQqVec;
    WAVE mleCHSHmaxvec;


    wavestats/q mleFid00Vec; printf "Fid00: %g, %g\r", V_avg, V_sdev
    wavestats/q mleFid01Vec; printf "Fid01: %g, %g\r", V_avg, V_sdev
    wavestats/q mleFid10Vec; printf "Fid10: %g, %g\r", V_avg, V_sdev
    wavestats/q mleFid11Vec; printf "Fid11: %g, %g\r", V_avg, V_sdev

    wavestats/q mleFidVec;              printf "Fid: %g, %g\r", V_avg, V_sdev
    wavestats/q mlePurityVec;           printf "Purity: %g, %g\r", V_avg, V_sdev
    wavestats/q mleConcurrenceVec;      printf "Conc: %g, %g\r", V_avg, V_sdev
    wavestats/q mleEntanglementVec;     printf "Ent't: %g, %g\r", V_avg, V_sdev
    wavestats/q mleQmlVec;              printf "Qml: %g, %g\r", V_avg, V_sdev
    wavestats/q mleQiVec;               printf "Qi: %g, %g\r", V_avg, V_sdev
    wavestats/q mleQqVec;               printf "Qq: %g, %g\r", V_avg, V_sdev
    wavestats/q mleCHSHmaxvec;          printf "CHSH: %g, %g\r", V_avg, V_sdev
endif

end


Window Graph_mleandtheoyzyz() : Graph
    PauseUpdate; Silent 1       // building window...
    Display /W=(531.75,260,751.5,460.25) mleCHSHyzyz0Vec,mleCHSHyzyz1Vec,mleCHSHyzyz2Vec
    AppendToGraph mleCHSHyzyz3Vec,mleCHSHmaxVec,theoCHSHyzyz[0][*],theoCHSHyzyz[1][*]
    AppendToGraph theoCHSHyzyz[2][*],theoCHSHyzyz[3][*],theoCHSHmax
    ModifyGraph userticks(left)={TicksCHSH,TickNamesCHSH}
    ModifyGraph userticks(bottom)={TicksCHSHAngles,TickNamesCHSHAngles}
    ModifyGraph margin(left)=37,margin(top)=35,margin(right)=10
    ModifyGraph mode(mleCHSHyzyz0Vec)=4,mode(mleCHSHyzyz1Vec)=4,mode(mleCHSHyzyz2Vec)=4
    ModifyGraph mode(mleCHSHyzyz3Vec)=4,mode(mleCHSHmaxVec)=4
    ModifyGraph marker(mleCHSHyzyz0Vec)=19,marker(mleCHSHyzyz1Vec)=19,marker(mleCHSHyzyz2Vec)=19
    ModifyGraph marker(mleCHSHyzyz3Vec)=19,marker(mleCHSHmaxVec)=19
    ModifyGraph rgb(mleCHSHyzyz0Vec)=(65280,0,0),rgb(mleCHSHyzyz1Vec)=(65280,21760,0)
    ModifyGraph rgb(mleCHSHyzyz2Vec)=(0,39168,0),rgb(mleCHSHyzyz3Vec)=(0,34816,52224)
    ModifyGraph rgb(mleCHSHmaxVec)=(0,0,0),rgb(theoCHSHyzyz)=(65280,0,0),rgb(theoCHSHyzyz#1)=(65280,21760,0)
    ModifyGraph rgb(theoCHSHyzyz#2)=(0,39168,0),rgb(theoCHSHyzyz#3)=(0,34816,52224)
    ModifyGraph rgb(theoCHSHmax)=(0,0,0)
    ModifyGraph msize(mleCHSHyzyz0Vec)=1.5,msize(mleCHSHyzyz1Vec)=1.5,msize(mleCHSHyzyz2Vec)=1.5
    ModifyGraph msize(mleCHSHyzyz3Vec)=1.5,msize(mleCHSHmaxVec)=1.5
    ModifyGraph grid(left)=1
    ModifyGraph tick=2
    ModifyGraph zero(left)=2
    ModifyGraph mirror=1
    ModifyGraph noLabel(left)=2
    ModifyGraph fSize=10
    ModifyGraph lblMargin(bottom)=4
    ModifyGraph axOffset(left)=-0.857143
    ModifyGraph gridRGB(left)=(4352,4352,4352)
    ModifyGraph btLen=2
    Label left "CHSH mean value"
    Label bottom "Rotation angle (deg) "
    SetAxis left -3,3
    Cursor/P A theoCHSHyzyz 15
    ShowInfo
    Legend/C/N=text0/J/F=0/B=1/A=MC/X=6.48/Y=42.32/E=2
    AppendText "\\Z08\\s(mleCHSHyzyz0vec) \\Z08\\s(mleCHSHyzyz1vec) \\Z08\\s(mleCHSHyzyz2vec) \\Z08\\s(mleCHSHyzyz3vec) \\s(mleCHSHmaxVec) mleCHSHyzyz"
    AppendText "\\s(theoCHSHyzyz) \\s(theoCHSHyzyz#1) \\s(theoCHSHyzyz#2) \\s(theoCHSHyzyz#3) \\s(theoCHSHmax) theoCHSHyzyz"
EndMacro

Window Graph_mleandtheoxzxz() : Graph
    PauseUpdate; Silent 1       // building window...
    Display /W=(306,260.75,526.5,461.75) mleCHSHxzxz0Vec,mleCHSHxzxz1Vec,mleCHSHxzxz2Vec
    AppendToGraph mleCHSHxzxz3Vec,mleCHSHmaxVec,theoCHSHxzxz[0][*],theoCHSHxzxz[1][*]
    AppendToGraph theoCHSHxzxz[2][*],theoCHSHxzxz[3][*],theoCHSHmax
    ModifyGraph userticks(left)={TicksCHSH,TickNamesCHSH}
    ModifyGraph userticks(bottom)={TicksCHSHAngles,TickNamesCHSHAngles}
    ModifyGraph margin(left)=37,margin(top)=35,margin(right)=10
    ModifyGraph mode(mleCHSHxzxz0Vec)=4,mode(mleCHSHxzxz1Vec)=4,mode(mleCHSHxzxz2Vec)=4
    ModifyGraph mode(mleCHSHxzxz3Vec)=4,mode(mleCHSHmaxVec)=4
    ModifyGraph marker(mleCHSHxzxz0Vec)=19,marker(mleCHSHxzxz1Vec)=19,marker(mleCHSHxzxz2Vec)=19
    ModifyGraph marker(mleCHSHxzxz3Vec)=19,marker(mleCHSHmaxVec)=19
    ModifyGraph rgb(mleCHSHxzxz0Vec)=(65280,16384,16384),rgb(mleCHSHxzxz1Vec)=(65280,32512,16384)
    ModifyGraph rgb(mleCHSHxzxz2Vec)=(0,52224,0),rgb(mleCHSHxzxz3Vec)=(0,34816,52224)
    ModifyGraph rgb(mleCHSHmaxVec)=(0,0,0),rgb(theoCHSHxzxz)=(65280,16384,16384),rgb(theoCHSHxzxz#1)=(65280,32512,16384)
    ModifyGraph rgb(theoCHSHxzxz#2)=(0,65280,0),rgb(theoCHSHxzxz#3)=(0,34816,52224)
    ModifyGraph rgb(theoCHSHmax)=(0,0,0)
    ModifyGraph msize(mleCHSHxzxz0Vec)=1.5,msize(mleCHSHxzxz1Vec)=1.5,msize(mleCHSHxzxz2Vec)=1.5
    ModifyGraph msize(mleCHSHxzxz3Vec)=1.5,msize(mleCHSHmaxVec)=1.5
    ModifyGraph grid(left)=1
    ModifyGraph tick=2
    ModifyGraph zero(left)=2
    ModifyGraph mirror=1
    ModifyGraph noLabel(left)=2
    ModifyGraph fSize=10
    ModifyGraph lblMargin(bottom)=4
    ModifyGraph axOffset(left)=-0.571429
    ModifyGraph gridRGB(left)=(4352,4352,4352)
    ModifyGraph btLen=2
    Label left "CHSH mean value"
    Label bottom "Rotation angle (deg) "
    SetAxis left -3,3
    Cursor/P A mleCHSHxzxz2Vec 12
    ShowInfo
    Legend/C/N=text0/J/F=0/B=1/A=MC/X=5.78/Y=42.70/E=2
    AppendText "\\Z08\\s(mleCHSHxzxz0vec) \\Z08\\s(mleCHSHxzxz1vec) \\Z08\\s(mleCHSHxzxz2vec) \\Z08\\s(mleCHSHxzxz3vec) \\s(mleCHSHmaxVec) mleCHSHxzxz"
    AppendText "\\s(theoCHSHxzxz) \\s(theoCHSHxzxz#1) \\s(theoCHSHxzxz#2) \\s(theoCHSHxzxz#3) \\s(theoCHSHmax) theoCHSHxzxz"
EndMacro

Window Graph_mleandtheoxyxy() : Graph
    PauseUpdate; Silent 1       // building window...
    Display /W=(79.5,261.5,299.25,459.5) mleCHSHxyxy0Vec,mleCHSHxyxy1Vec,mleCHSHxyxy2Vec
    AppendToGraph mleCHSHxyxy3Vec,mleCHSHmaxVec,theoCHSHxyxy[0][*],theoCHSHxyxy[1][*]
    AppendToGraph theoCHSHxyxy[2][*],theoCHSHxyxy[3][*],theoCHSHmax
    ModifyGraph userticks(left)={TicksCHSH,TickNamesCHSH}
    ModifyGraph userticks(bottom)={TicksCHSHAngles,TickNamesCHSHAngles}
    ModifyGraph margin(left)=37,margin(top)=35,margin(right)=10
    ModifyGraph mode(mleCHSHxyxy0Vec)=4,mode(mleCHSHxyxy1Vec)=4,mode(mleCHSHxyxy2Vec)=4
    ModifyGraph mode(mleCHSHxyxy3Vec)=4,mode(mleCHSHmaxVec)=4
    ModifyGraph marker(mleCHSHxyxy0Vec)=19,marker(mleCHSHxyxy1Vec)=19,marker(mleCHSHxyxy2Vec)=19
    ModifyGraph marker(mleCHSHxyxy3Vec)=19,marker(mleCHSHmaxVec)=19
    ModifyGraph lSize(theoCHSHxyxy)=2
    ModifyGraph rgb(mleCHSHxyxy0Vec)=(65280,16384,16384),rgb(mleCHSHxyxy1Vec)=(65280,32512,16384)
    ModifyGraph rgb(mleCHSHxyxy2Vec)=(0,39168,0),rgb(mleCHSHxyxy3Vec)=(0,34816,52224)
    ModifyGraph rgb(mleCHSHmaxVec)=(0,0,0),rgb(theoCHSHxyxy)=(65280,16384,16384),rgb(theoCHSHxyxy#1)=(65280,32512,16384)
    ModifyGraph rgb(theoCHSHxyxy#2)=(0,39168,0),rgb(theoCHSHxyxy#3)=(0,34816,52224)
    ModifyGraph rgb(theoCHSHmax)=(0,0,0)
    ModifyGraph msize(mleCHSHxyxy0Vec)=1.5,msize(mleCHSHxyxy1Vec)=1.5,msize(mleCHSHxyxy2Vec)=1.5
    ModifyGraph msize(mleCHSHxyxy3Vec)=1.5,msize(mleCHSHmaxVec)=1.5
    ModifyGraph offset(theoCHSHxyxy#1)={0,0.01}
    ModifyGraph grid(left)=1
    ModifyGraph tick=2
    ModifyGraph zero(left)=2
    ModifyGraph mirror=1
    ModifyGraph fSize=10
    ModifyGraph lblMargin(bottom)=4
    ModifyGraph gridRGB(left)=(4352,4352,4352)
    ModifyGraph btLen=2
    Label left "CHSH mean value "
    Label bottom "Rotation angle (deg)"
    SetAxis left -3,3
    Cursor/P A mleCHSHxyxy2Vec 14
    ShowInfo
    Legend/C/N=text0/J/F=0/B=1/A=MC/X=5.46/Y=41.95/E=2
    AppendText "\\Z08\\s(mleCHSHxyxy0vec) \\Z08\\s(mleCHSHxyxy1vec) \\Z08\\s(mleCHSHxyxy2vec) \\Z08\\s(mleCHSHxyxy3vec) \\s(mleCHSHmaxvec) mleCHSHxyxy"
    AppendText "\\s(theoCHSHxyxy) \\s(theoCHSHxyxy#1) \\s(theoCHSHxyxy#2) \\s(theoCHSHxyxy#3) \\s(theoCHSHmax) theoCHSHxyxy"
EndMacro




Window Graph_mleandtheoSimpleCorr() : Graph
    PauseUpdate; Silent 1       // building window...
    Display /W=(66,275,285.75,475.25) mleCHSHxzVec,mleCHSHmaxVec,theoCHSHxz,theoCHSHmax
    ModifyGraph userticks(left)={TicksCHSH,TickNamesCHSH}
    ModifyGraph userticks(bottom)={TicksCHSHAngles,TickNamesCHSHAngles}
    ModifyGraph margin(left)=37,margin(top)=30,margin(right)=10
    ModifyGraph mode(mleCHSHxzVec)=4,mode(mleCHSHmaxVec)=4
    ModifyGraph marker(mleCHSHxzVec)=19,marker(mleCHSHmaxVec)=19,marker(theoCHSHxz)=19
    ModifyGraph rgb(mleCHSHmaxVec)=(0,0,0),rgb(theoCHSHxz)=(65280,0,0),rgb(theoCHSHmax)=(4352,4352,4352)
    ModifyGraph msize(mleCHSHxzVec)=1.5,msize(mleCHSHmaxVec)=1.5,msize(theoCHSHxz)=1.5
    ModifyGraph grid(left)=1
    ModifyGraph tick=2
    ModifyGraph zero(left)=2
    ModifyGraph mirror=1
    ModifyGraph fSize=10
    ModifyGraph lblMargin(bottom)=4
    ModifyGraph gridRGB(left)=(4352,4352,4352)
    ModifyGraph btLen=2
    Label left "CHSH mean value "
    Label bottom "Rotation angle (deg) "
    SetAxis left -3,3
    Legend/C/N=text0/J/F=0/B=1/A=MC/X=2.39/Y=42.70/E=2 "\\Z08\\s(mleCHSHxzVec) mleCHSHxz   \\s(mleCHSHmaxVec) mleCHSHmax"
    AppendText "\\Z08\\s(theoCHSHxz) theoCHSHxz \\s(theoCHSHmax) theoCHSHmax"
EndMacro


Window Graph_mleandtheoMetrics() : Graph
    PauseUpdate; Silent 1       // building window...
    Display /W=(297.75,275,518.25,475.25) theoConcurrence,mleConcurrenceVec,mlePurityVec
    ModifyGraph userticks(bottom)={TicksCHSHAngles,TickNamesCHSHAngles}
    ModifyGraph margin(left)=37,margin(top)=30,margin(right)=10
    ModifyGraph mode(mleConcurrenceVec)=4,mode(mlePurityVec)=4
    ModifyGraph marker=19
    ModifyGraph rgb(theoConcurrence)=(0,65280,0),rgb(mleConcurrenceVec)=(0,65280,0)
    ModifyGraph msize=1.5
    ModifyGraph grid(left)=1
    ModifyGraph tick=2
    ModifyGraph zero(left)=2
    ModifyGraph mirror=1
    ModifyGraph fSize=10
    ModifyGraph lblMargin(bottom)=4
    ModifyGraph gridRGB(left)=(4352,4352,4352)
    ModifyGraph btLen=2
    Label left "Percentage"
    Label bottom "Rotation angle (deg) "
    SetAxis left -0.05,1.05
    Legend/C/N=text0/J/F=0/B=1/A=MC/X=5.10/Y=43.45/E=2 "\\Z08\\s(mlePurityVec) mlePurityVec \t\\s(mleConcurrenceVec) mleConcurrence"
    AppendText "\t\t\\Z08\\s(theoConcurrence) theoConcurrence "
EndMacro



Window Layout_mleandtheo() : Layout
    PauseUpdate; Silent 1       // building window...
    Layout/C=1/W=(56.25,74,932.25,671) Graph_mleandtheoyzyz(432,46.5,651.75,246.75)/O=1/F=0/T
    Append Graph_mleandtheoxzxz(246.75,46.5,467.25,246.75)/O=1/F=0/T,Graph_mleandtheoxyxy(62.25,46.5,282,246.75)/O=1/F=0/T
    Append Graph_mleandtheoSimpleCorr(62.25,241.5,282,441.75)/O=1/F=0/T,Graph_mleandtheoMetrics(431.25,241.5,651.75,441.75)/O=1/F=0/T
    ModifyLayout mag=1, units=1
EndMacro


// function getMovieSim
//----------------------------------------
function getMovieSim(basename,startval,endval)
string basename
variable startval
variable endval


string filenameW1="W1";
string filenameW2="W2";
string filenameW3="W3";
string filenameW4="W4";

string filenameCHxy1="CHxy1";
string filenameCHxy2="CHxy2";
string filenameCHxy3="CHxy3";
string filenameCHxy4="CHxy4";

string filenameCHxz1="CHxz1";
string filenameCHxz2="CHxz2";
string filenameCHxz3="CHxz3";
string filenameCHxz4="CHxz4";

string filenameCHyz1="CHyz1";
string filenameCHyz2="CHyz2";
string filenameCHyz3="CHyz3";
string filenameCHyz4="CHyz4";


string filenameIX="IX";
string filenameIY="IY";
string filenameIZ="IZ";;
string filenameXI="XI";
string filenameXX="XX";
string filenameXY="XY";
string filenameXZ="XZ";
string filenameYI="YI";
string filenameYX="YX";
string filenameYY="YY";
string filenameYZ="YZ";
string filenameZI="ZI";
string filenameZX="ZX";
string filenameZY="ZY";
string filenameZZ="ZZ";
string filenameFid="Fid";



 LoadMathematicaFile(filenameW1,basename,startval,endval)
 LoadMathematicaFile(filenameW2,basename,startval,endval)
 LoadMathematicaFile(filenameW3,basename,startval,endval)
 LoadMathematicaFile(filenameW4,basename,startval,endval)

 LoadMathematicaFile(filenameCHxy1,basename,startval,endval)
 LoadMathematicaFile(filenameCHxy2,basename,startval,endval)
 LoadMathematicaFile(filenameCHxy3,basename,startval,endval)
 LoadMathematicaFile(filenameCHxy4,basename,startval,endval)


 LoadMathematicaFile(filenameCHxz1,basename,startval,endval)
 LoadMathematicaFile(filenameCHxz2,basename,startval,endval)
 LoadMathematicaFile(filenameCHxz3,basename,startval,endval)
 LoadMathematicaFile(filenameCHxz4,basename,startval,endval)


 LoadMathematicaFile(filenameCHyz1,basename,startval,endval)
 LoadMathematicaFile(filenameCHyz2,basename,startval,endval)
 LoadMathematicaFile(filenameCHyz3,basename,startval,endval)
 LoadMathematicaFile(filenameCHyz4,basename,startval,endval)

 LoadMathematicaFile(filenameIX,basename,startval,endval)
 LoadMathematicaFile(filenameIY,basename,startval,endval)
 LoadMathematicaFile(filenameIZ,basename,startval,endval)

 LoadMathematicaFile(filenameXI,basename,startval,endval)
 LoadMathematicaFile(filenameXX,basename,startval,endval)
 LoadMathematicaFile(filenameXY,basename,startval,endval)
 LoadMathematicaFile(filenameXZ,basename,startval,endval)

 LoadMathematicaFile(filenameYI,basename,startval,endval)
 LoadMathematicaFile(filenameYX,basename,startval,endval)
 LoadMathematicaFile(filenameYY,basename,startval,endval)
 LoadMathematicaFile(filenameYZ,basename,startval,endval)

 LoadMathematicaFile(filenameZI,basename,startval,endval)
 LoadMathematicaFile(filenameZX,basename,startval,endval)
 LoadMathematicaFile(filenameZY,basename,startval,endval)
 LoadMathematicaFile(filenameZZ,basename,startval,endval)

 LoadMathematicaFile(filenameFid,basename,startval,endval)


  end





Window Graph_rawandsimWitnesses() : Graph
    PauseUpdate; Silent 1       // building window...
    Display /W=(5.25,263,225.75,465.5) rawWitn0vec,rawWitn1vec,rawWitn2vec,rawWitn3vec
    AppendToGraph rawFidvec,W1_Exp1,W2_Exp1,W3_Exp1,W4_Exp1,simFidvec,simW1vec,simW2vec
    AppendToGraph simW3vec,simW4vec
    ModifyGraph userticks(bottom)={TicksCHSHAngles,TickNamesCHSHAngles}
    ModifyGraph margin(top)=35,margin(right)=10
    ModifyGraph mode(rawWitn0vec)=3,mode(rawWitn1vec)=3,mode(rawWitn2vec)=3,mode(rawWitn3vec)=3
    ModifyGraph mode(rawFidvec)=3
    ModifyGraph marker(rawWitn0vec)=19,marker(rawWitn1vec)=19,marker(rawWitn2vec)=19
    ModifyGraph marker(rawWitn3vec)=19,marker(rawFidvec)=8
    ModifyGraph rgb(rawWitn0vec)=(65280,0,0),rgb(rawWitn1vec)=(65280,43520,0),rgb(rawWitn2vec)=(0,52224,0)
    ModifyGraph rgb(rawWitn3vec)=(0,16320,65280),rgb(rawFidvec)=(0,0,0),rgb(W2_Exp1)=(65280,43520,0)
    ModifyGraph rgb(W3_Exp1)=(0,52224,0),rgb(W4_Exp1)=(0,12800,52224),rgb(simFidvec)=(0,0,0)
    ModifyGraph rgb(simW2vec)=(65280,43520,0),rgb(simW3vec)=(0,52224,0),rgb(simW4vec)=(0,12800,52224)
    ModifyGraph msize(rawWitn0vec)=1.5,msize(rawWitn1vec)=1.5,msize(rawWitn2vec)=1.5
    ModifyGraph msize(rawWitn3vec)=1.5,msize(rawFidvec)=1.5
    ModifyGraph opaque(rawFidvec)=1
    ModifyGraph grid(left)=1
    ModifyGraph tick=2
    ModifyGraph zero(left)=2
    ModifyGraph mirror=1
    ModifyGraph fSize=10
    ModifyGraph btLen=2
    Label left "Witness"
    Label bottom "Rotation angle [deg]"
    SetAxis left -1.2,1.2
    Legend/C/N=text0/J/F=0/B=1/A=MC/X=7.12/Y=41.83/E=2 "\\Z08\\s(rawWitn0vec) \\Z08\\s(rawWitn1vec) \\Z08\\s(rawWitn2vec) \\Z08\\s(rawWitn3vec) rawWitn#vec"
    AppendText "\\s(simW1vec) \\s(simW2vec) \\s(simW3vec) \\s(simW4vec) simW#vec\r\\s(rawFidvec) \\s(simFidvec) raw and sim Fidelity"
EndMacro


Window Graph_rawandsimxz() : Graph
    PauseUpdate; Silent 1       // building window...
    Display /W=(230.25,39.5,450.75,242.75) rawCHSHxzxz0vec,rawCHSHxzxz1vec,rawCHSHxzxz2vec
    AppendToGraph rawCHSHxzxz3vec,simCHxz1vec,simCHxz2vec,simCHxz3vec,simCHxz4vec
    ModifyGraph userticks(left)={TicksCHSH,TickNamesCHSH}
    ModifyGraph margin(left)=37,margin(top)=35,margin(right)=10
    ModifyGraph mode(rawCHSHxzxz0vec)=3,mode(rawCHSHxzxz1vec)=3,mode(rawCHSHxzxz2vec)=3
    ModifyGraph mode(rawCHSHxzxz3vec)=3
    ModifyGraph marker(rawCHSHxzxz0vec)=19,marker(rawCHSHxzxz1vec)=19,marker(rawCHSHxzxz2vec)=19
    ModifyGraph marker(rawCHSHxzxz3vec)=19
    ModifyGraph rgb(rawCHSHxzxz0vec)=(65280,0,0),rgb(rawCHSHxzxz1vec)=(65280,32512,16384)
    ModifyGraph rgb(rawCHSHxzxz2vec)=(0,52224,0),rgb(rawCHSHxzxz3vec)=(0,34816,52224)
    ModifyGraph rgb(simCHxz2vec)=(65280,21760,0),rgb(simCHxz3vec)=(0,52224,0),rgb(simCHxz4vec)=(0,12800,52224)
    ModifyGraph msize(rawCHSHxzxz0vec)=1.5,msize(rawCHSHxzxz1vec)=1.5,msize(rawCHSHxzxz2vec)=1.5
    ModifyGraph msize(rawCHSHxzxz3vec)=1.5
    ModifyGraph grid(left)=1
    ModifyGraph tick=2
    ModifyGraph zero(left)=2
    ModifyGraph mirror=1
    ModifyGraph noLabel(left)=2
    ModifyGraph fSize=10
    ModifyGraph gridRGB(left)=(0,0,0)
    ModifyGraph btLen=2
    Label left "CHSH mean value "
    Label bottom "Rotation angle [deg]"
    SetAxis left -3,3
    Legend/C/N=text0/J/F=0/B=1/A=MC/X=7.48/Y=41.57/E=2
    AppendText "\\Z08\\s(rawCHSHxzxz0vec) \\Z08\\s(rawCHSHxzxz1vec) \\Z08\\s(rawCHSHxzxz2vec) \\Z08\\s(rawCHSHxzxz3vec) rawCHSHxzxz"
    AppendText "\\s(simCHxz1vec) \\s(simCHxz2vec) \\s(simCHxz3vec) \\s(simCHxz4vec) simCHxz#vec"
EndMacro


Window Graph_rawandsimyz() : Graph
    PauseUpdate; Silent 1       // building window...
    Display /W=(456,39.5,676.5,242) rawCHSHyzyz0vec,rawCHSHyzyz1vec,rawCHSHyzyz2vec,rawCHSHyzyz3vec
    AppendToGraph simCHyz1vec,simCHyz2vec,simCHyz3vec,simCHyz4vec
    ModifyGraph userticks(left)={TicksCHSH,TickNamesCHSH}
    ModifyGraph margin(left)=37,margin(top)=35,margin(right)=10
    ModifyGraph mode(rawCHSHyzyz0vec)=3,mode(rawCHSHyzyz1vec)=3,mode(rawCHSHyzyz2vec)=3
    ModifyGraph mode(rawCHSHyzyz3vec)=3
    ModifyGraph marker(rawCHSHyzyz0vec)=19,marker(rawCHSHyzyz1vec)=19,marker(rawCHSHyzyz2vec)=19
    ModifyGraph marker(rawCHSHyzyz3vec)=19
    ModifyGraph rgb(rawCHSHyzyz0vec)=(65280,0,0),rgb(rawCHSHyzyz1vec)=(65280,21760,0)
    ModifyGraph rgb(rawCHSHyzyz2vec)=(0,52224,0),rgb(rawCHSHyzyz3vec)=(0,34816,52224)
    ModifyGraph rgb(simCHyz2vec)=(65280,21760,0),rgb(simCHyz3vec)=(0,52224,0),rgb(simCHyz4vec)=(0,12800,52224)
    ModifyGraph msize(rawCHSHyzyz0vec)=1.5,msize(rawCHSHyzyz1vec)=1.5,msize(rawCHSHyzyz2vec)=1.5
    ModifyGraph msize(rawCHSHyzyz3vec)=1.5
    ModifyGraph grid(left)=1
    ModifyGraph tick=2
    ModifyGraph zero(left)=2
    ModifyGraph mirror=1
    ModifyGraph noLabel(left)=2
    ModifyGraph fSize=10
    ModifyGraph gridRGB(left)=(0,0,0)
    ModifyGraph btLen=2
    Label left "CHSH mean value "
    Label bottom "Rotation angle [deg]"
    SetAxis left -3,3
    Legend/C/N=text0/J/F=0/B=1/A=MC/X=5.10/Y=42.70/E=2
    AppendText "\\Z08\\s(rawCHSHyzyz0vec) \\Z08\\s(rawCHSHyzyz1vec) \\Z08\\s(rawCHSHyzyz2vec) \\Z08\\s(rawCHSHyzyz3vec) rawCHSHyzyz"
    AppendText "\\s(simCHyz1vec) \\s(simCHyz2vec) \\s(simCHyz3vec) \\s(simCHyz4vec) simCHyz#vec"
EndMacro

Window Graph_rawandsimxy() : Graph
    PauseUpdate; Silent 1       // building window...
    Display /W=(5.25,39.5,225.75,242.75) rawCHSHxyxy0vec,rawCHSHxyxy1vec,rawCHSHxyxy2vec
    AppendToGraph rawCHSHxyxy3vec,simCHxy1vec,simCHxy2vec,simCHxy3vec,simCHxy4vec
    ModifyGraph userticks(left)={TicksCHSH,TickNamesCHSH}
    ModifyGraph margin(left)=37,margin(top)=35,margin(right)=10
    ModifyGraph mode(rawCHSHxyxy0vec)=3,mode(rawCHSHxyxy1vec)=3,mode(rawCHSHxyxy2vec)=3
    ModifyGraph mode(rawCHSHxyxy3vec)=3
    ModifyGraph marker(rawCHSHxyxy0vec)=19,marker(rawCHSHxyxy1vec)=19,marker(rawCHSHxyxy2vec)=19
    ModifyGraph marker(rawCHSHxyxy3vec)=19
    ModifyGraph rgb(rawCHSHxyxy0vec)=(65280,0,0),rgb(rawCHSHxyxy1vec)=(65280,32512,16384)
    ModifyGraph rgb(rawCHSHxyxy2vec)=(0,52224,26368),rgb(rawCHSHxyxy3vec)=(0,34816,52224)
    ModifyGraph rgb(simCHxy2vec)=(65280,21760,0),rgb(simCHxy3vec)=(0,52224,0),rgb(simCHxy4vec)=(0,12800,52224)
    ModifyGraph msize(rawCHSHxyxy0vec)=1.5,msize(rawCHSHxyxy1vec)=1.5,msize(rawCHSHxyxy2vec)=1.5
    ModifyGraph msize(rawCHSHxyxy3vec)=1.5
    ModifyGraph grid(left)=1
    ModifyGraph tick=2
    ModifyGraph zero(left)=2
    ModifyGraph mirror=1
    ModifyGraph fSize=10
    ModifyGraph gridRGB(left)=(0,0,0)
    ModifyGraph btLen=2
    Label left "CHSH mean value "
    Label bottom "Rotation angle [deg]"
    SetAxis left -3,3
    Legend/C/N=text0/J/F=0/B=1/A=MC/X=6.46/Y=41.95/E=2
    AppendText "\\Z08\\s(rawCHSHxyxy0vec) \\Z08\\s(rawCHSHxyxy1vec) \\Z08\\s(rawCHSHxyxy2vec) \\Z08\\s(rawCHSHxyxy3vec) rawCHSHxyxy"
    AppendText "\\s(simCHxy1vec) \\s(simCHxy2vec) \\s(simCHxy3vec) \\s(simCHxy4vec) simCHxy#vec"
EndMacro

// Function: ExtractST2QPauli
// -------------------------- ---------------

function ExtractST2QPauli(filenum, basename, phase, datestr, opcode, donorm, whichplot,savenum)
    variable filenum
    string basename
    string phase
    string datestr
    variable opcode     // opcode==0: asume 1 point for each type of measurements,
                        // opcode==1: assume 2 points for each type of measurements
                        // opcode==2: assume 8 points for each of the 4 pi/2,pi/2 measurements.
    variable donorm     // donorm==1: normalize Betas to Beta_1; donorm==0 : do not normalize
    variable whichplot      // whichplot==0, display normwave, whichplot==1, display ST2Qvec. whichpolot==2, display ST2QPaulivec
    variable savenum


    variable calSegments=20;
    variable numSegmentsPerState=15;

    //import the experiment vector
    string inputwavename=fancynum2str(filenum)+"_"+phase+"2DPlot"+basename+"_"+datestr+"_ex1_v";
    wave inputwave=$inputwavename

    //prepare the normalized vector
    string normwavename=inputwavename+"n"
    duplicate/o inputwave $normwavename
    wave normwave=$normwavename


    //display inputwave

    wavestats/Q/R=[0,calSegments-1] inputwave
    //print V_avg
    variable offset=V_avg
    normwave-=offset

    variable Beta1, Beta2, Beta12
    variable dBeta1, dBeta2, dBeta12

    wavestats/Q/R=[0,calSegments/4-1] normwave
    variable M1=V_avg
    variable dM1=V_sdev/sqrt(calSegments/4);
    wavestats/Q/R=[calSegments/4,2*calSegments/4-1] normwave
    variable M2=V_avg
    variable dM2=V_sdev/sqrt(calSegments/4);
    wavestats/Q/R=[2*calSegments/4,3*calSegments/4-1] normwave
    variable M3=V_avg
    variable dM3=V_sdev/sqrt(calSegments/4);
    wavestats/Q/R=[3*calSegments/4,4*calSegments/4-1] normwave
    variable M4=V_avg
    variable dM4=V_sdev/sqrt(calSegments/4);

    Beta1=(M1+M3)/2;
    Beta2=((M1+M2)/2);
    Beta12=((M1+M4)/2);

    dBeta1=sqrt(dM1^2+dM3^2)/2;
    dBeta2=sqrt(dM1^2+dM2^2)/2;
    dBeta12=sqrt(dM1^2+dM4^2)/2;


    //print (M1+M3)/2,  -(M2+M4)/2
    //print (M1+M2)/2,  -(M3+M4)/2
    //print -(M1+M4)/2, (M2+M3)/2

    if(donorm==1)       //normalize
        variable factor=(Beta1);
        normwave/=factor;
        Beta1/=factor;
        Beta2/=factor;
        Beta12/=factor;
        dBeta1/=factor;
        dBeta2/=factor;
        dBeta12/=factor;
        offset/=factor
    endif


    // Copy the Beta and dBeta values into the BetaVec wave used by other functions, such as analyzeRFsweep
    WAVE BetaVec
    BetaVec[0]=Beta1;
    BetaVec[1]=Beta2;
    BetaVec[2]=Beta12;
    BetaVec[3]=dBeta1;
    BetaVec[4]=dBeta2;
    BetaVec[5]=dBeta12;

    NVAR bequiet

    if(bequiet==0)
        printf  "Beta1=%.4g +/- %.2g,   Beta2=%.4g +/- %.2g,    Beta12=%.4g +/- %.2g        (%.2f, %.2f, %.2f)   offset=%.4g\r", Beta1, dBeta1,  Beta2, dBeta2,   Beta12, dBeta12,   dBeta1/Beta1, dBeta2/Beta2, dBeta12/Beta12, offset
    endif

    make/o/n=(2) HL1=NaN;
setscale/I x, -2, 16, HL1
duplicate/o HL1, HL2
duplicate/o HL1, HL3
WAVE HL1, HL2,HL3

HL1=Beta2+(Beta1+Beta12)
HL2=Beta2-(Beta1+Beta12)
HL3=Beta1-(Beta2+Beta12)


    make/o/n=(19) ST2Qvec=NaN
    setscale/P x, -2, 1, ST2Qvec;

    ST2Qvec[0]=Beta1;
    ST2Qvec[1]=Beta2;
    ST2Qvec[2]=Beta12;

if(opcode==0)
    ST2Qvec[3,17]=normwave[calSegments+(p-3)]
elseif(opcode==1)
    ST2Qvec[3,17]=(normwave[calSegments+2*(p-3)]+normwave[calSegments+2*(p-3)+1])/2
else
    ST2Qvec[3]=     (normwave[calSegments+0]+normwave[calSegments+1])/2;
    ST2Qvec[4]=     (normwave[calSegments+2]+normwave[calSegments+3])/2;
    ST2Qvec[5]=     (normwave[calSegments+4]+normwave[calSegments+5])/2;
    ST2Qvec[6]=     (normwave[calSegments+6]+normwave[calSegments+7])/2;

    ST2Qvec[7]=     (normwave[calSegments+  8]+normwave[calSegments+  9]+normwave[calSegments+10]+normwave[calSegments+11]+normwave[calSegments+12]+normwave[calSegments+13]+normwave[calSegments+14]+normwave[calSegments+15])/8;
    ST2Qvec[8]=     (normwave[calSegments+16]+normwave[calSegments+17]+normwave[calSegments+18]+normwave[calSegments+19]+normwave[calSegments+20]+normwave[calSegments+21]+normwave[calSegments+22]+normwave[calSegments+23])/8;

    ST2Qvec[9]=     (normwave[calSegments+24]+normwave[calSegments+25])/2;
    ST2Qvec[10]=    (normwave[calSegments+26]+normwave[calSegments+27])/2;

    ST2Qvec[11]=    (normwave[calSegments+28]+normwave[calSegments+29]+normwave[calSegments+30]+normwave[calSegments+31]+normwave[calSegments+32]+normwave[calSegments+33]+normwave[calSegments+34]+normwave[calSegments+35])/8;
    ST2Qvec[12]=    (normwave[calSegments+36]+normwave[calSegments+37]+normwave[calSegments+38]+normwave[calSegments+39]+normwave[calSegments+40]+normwave[calSegments+41]+normwave[calSegments+42]+normwave[calSegments+43])/8;

    ST2Qvec[13]=    (normwave[calSegments+44]+normwave[calSegments+45])/2;
    ST2Qvec[14]=    (normwave[calSegments+46]+normwave[calSegments+47])/2;
    ST2Qvec[15]=    (normwave[calSegments+48]+normwave[calSegments+49])/2;
    ST2Qvec[16]=    (normwave[calSegments+50]+normwave[calSegments+51])/2;
    ST2Qvec[17]=    (normwave[calSegments+52]+normwave[calSegments+53])/2;
endif
ST2Qvec[18]=1

make/o/n=(16) tempvec
tempvec[]=ST2Qvec[3+p];


// Finally convert to measurements into the Pauli basis
make/o/n=(16,16) ObsFromPauli=0;

ObsFromPauli[15][0]=1;

ObsFromPauli[0][3]=Beta1;
ObsFromPauli[1][3]=-Beta1;
ObsFromPauli[2][3]=Beta1;
ObsFromPauli[3][2]=Beta1;
ObsFromPauli[4][2]=Beta1;
ObsFromPauli[5][2]=Beta1;
ObsFromPauli[6][2]=Beta1;
ObsFromPauli[7][1]=-Beta1;
ObsFromPauli[8][1]=-Beta1;
ObsFromPauli[9][1]=-Beta1;
ObsFromPauli[10][1]=    -Beta1;
ObsFromPauli[11][3]=    Beta1;
ObsFromPauli[12][3]=    -Beta1;
ObsFromPauli[13][3]=Beta1;
ObsFromPauli[14][3]=-Beta1;

ObsFromPauli[0][12]=Beta2
ObsFromPauli[1][12]=Beta2
ObsFromPauli[2][12]=-Beta2
ObsFromPauli[3][12]=Beta2
ObsFromPauli[4][8]=Beta2
ObsFromPauli[5][4]=-Beta2
ObsFromPauli[6][12]=-Beta2
ObsFromPauli[7][12]=Beta2
ObsFromPauli[8][8]=Beta2
ObsFromPauli[9][4]=-Beta2
ObsFromPauli[10][12]=-Beta2
ObsFromPauli[11][8]=Beta2
ObsFromPauli[12][8]=Beta2
ObsFromPauli[13][4]=-Beta2
ObsFromPauli[14][4]=-Beta2

ObsFromPauli[0][15]=Beta12
ObsFromPauli[1][15]=-Beta12
ObsFromPauli[2][15]=-Beta12
ObsFromPauli[3][14]=Beta12
ObsFromPauli[4][10]=Beta12
ObsFromPauli[5][6]=-Beta12
ObsFromPauli[6][14]=-Beta12
ObsFromPauli[7][13]=-Beta12
ObsFromPauli[8][9]=-Beta12
ObsFromPauli[9][5]=Beta12
ObsFromPauli[10][13]=Beta12
ObsFromPauli[11][11]=Beta12
ObsFromPauli[12][11]=-Beta12
ObsFromPauli[13][7]=-Beta12
ObsFromPauli[14][7]=Beta12

MatrixOP/O PauliFromObs=Inv(ObsFromPauli)
MatrixOP/O ST2QPauliVec= PauliFromObs x tempvec
WAVE ST2QPauliVec
setscale/P x, 1, 1, ST2QPauliVec


    // Save vectors  to the tomopath
    string ST2Qvecfilename1="ST2Qvec"+phase+".txt"
    string ST2Qvecfilename2="ST2Qvec"+".txt"
    Save/O/G/M="\r\n"/P=tomopath ST2Qvec as ST2Qvecfilename1
    Save/O/G/M="\r\n"/P=tomopath ST2Qvec as ST2Qvecfilename2

    if(savenum==1)
        ST2Qvecfilename1="ST2Qvec"+phase+"_"+num2str(filenum)+".txt"
        ST2Qvecfilename2="ST2Qvec"+"_"+num2str(filenum)+".txt"
        Save/O/G/M="\r\n"/P=tomopath ST2Qvec as ST2Qvecfilename1
        Save/O/G/M="\r\n"/P=tomopath ST2Qvec as ST2Qvecfilename2
    endif

    string  ST2QPaulivecfilename1="ST2QPaulivec"+phase+".txt"
    string  ST2QPaulivecfilename2="ST2QPaulivec"+".txt"
    Save/O/G/M="\r\n"/P=tomopath ST2QPaulivec as ST2QPaulivecfilename1
    Save/O/G/M="\r\n"/P=tomopath ST2QPaulivec as ST2QPaulivecfilename2

    if(savenum==1)
        ST2QPaulivecfilename1="ST2QPaulivec"+phase+"_"+num2str(filenum)+".txt"
        ST2QPaulivecfilename2="ST2QPaulivec"+"_"+num2str(filenum)+".txt"
        Save/O/G/M="\r\n"/P=tomopath ST2QPaulivec as ST2QPaulivecfilename1
        Save/O/G/M="\r\n"/P=tomopath ST2QPaulivec as ST2QPaulivecfilename2
    endif

    // Make copies of vectors with run specific names
    string ST2Qvecname="ST2Qvec_"+fancynum2str(filenum)+"_"+phase+"_"+basename+"_"+datestr;
    duplicate/O ST2Qvec $ST2Qvecname
    WAVE thisST2Qvec=$ST2Qvecname

    string ST2QPaulivecname="ST2QPaulivec_"+fancynum2str(filenum)+"_"+phase+"_"+basename+"_"+datestr;
    duplicate/O ST2QPaulivec $ST2QPaulivecname
    WAVE thisST2QPaulivec=$ST2QPaulivecname

    NVAR doplots
    if(doplots==1)
        display;
    endif
    if(doplots>0)
        if(whichplot==0)
            appendtograph  normwave
        elseif(whichplot==1)
            appendtograph thisST2Qvec
            label bottom "Raw measurement number"
            ModifyGraph zero(bottom)=1
            WAVE TicksST2Q
            WAVE TickNamesST2Q
            ModifyGraph userticks(bottom)={TicksST2Q,TickNamesST2Q}

        elseif(whichplot==2)
            appendtograph thisST2QPaulivec
            label bottom "Pauli measurement"
            WAVE TicksST2QP
            WAVE TickNamesST2QP
            ModifyGraph userticks(bottom)={TicksST2QP,TickNamesST2QP}
        elseif(whichplot==3)
            appendtograph thisST2QPaulivec
            label bottom "Pauli measurement"
            WAVE TicksST2QP2
            WAVE TickNamesST2QP2
            ModifyGraph userticks(bottom)={TicksST2QP2,TickNamesST2QP2}
        endif

        fixl(); autocolor(); gridon(2); dotize();
        ModifyGraph nticks(left)=5
        label left "Mean Value"
        if(donorm==1)
            SetAxis left -3,3
        endif
    endif




// Calculate what ideal measurements of observables would give with ground state as input
make/o/n=(19) CalVec=NaN;
setscale/P x, -2, 1, CalVec;
CalVec[0]=Beta1;
CalVec[1]=Beta2;
CalVec[2]=Beta12
CalVec[3]=Beta1+Beta2+Beta12;
CalVec[4]=-Beta1+Beta2-Beta12;
CalVec[5]=Beta1-Beta2-Beta12;
CalVec[6]=Beta2
CalVec[7]=0
CalVec[8]=0
CalVec[9]=-Beta2
CalVec[10]=Beta2
CalVec[11]=0
CalVec[12]=0
CalVec[13]=-Beta2
CalVec[14]=Beta1
CalVec[15]=-Beta1
CalVec[16]=Beta1
CalVec[17]=-Beta1
CalVec[18]=1


make/o/n=(3) PQ1vec=NaN
setscale/I x, 2, 4, PQ1vec
make/o/n=(3)PQ2vec=Nan
setscale/I x, 5, 7, PQ2vec
make/o/n=(9) PQ12vec=Nan
setscale/I x, 8, 16, PQ12vec

PQ1vec[0]=ST2QPaulivec[1];
PQ1vec[1]=ST2QPaulivec[2];
PQ1vec[2]=ST2QPaulivec[3];

PQ2vec[0]=ST2QPaulivec[4]
PQ2vec[1]=ST2QPaulivec[8]
PQ2vec[2]=ST2QPaulivec[12]

PQ12vec[0]=ST2QPaulivec[9]
PQ12vec[1]=ST2QPaulivec[13]
PQ12vec[2]=ST2QPaulivec[6]
PQ12vec[3]=ST2QPaulivec[14]
PQ12vec[4]=ST2QPaulivec[7]
PQ12vec[5]=ST2QPaulivec[11]
PQ12vec[6]=ST2QPaulivec[5]
PQ12vec[7]=ST2QPaulivec[10]
PQ12vec[8]=ST2QPaulivec[15]

end




// Function: ExtractStateTomo2Q
// ---------------------------------------------

function ExtractStateTomo2Q(filenum, basename, phase, datestr, opcode, donorm, whichplot)
    variable filenum
    string basename
    string phase
    string datestr
    variable opcode     // opcode==0: asume 1 point for each type of measurements,
                        // opcode==1: assume 2 points for each type of measurements
                        // opcode==2: assume 8 points for each of the 4 pi/2,pi/2 measurements.
    variable donorm     // donorm==1: normalize Betas to Beta_1; donorm==0 : do not normalize
    variable whichplot      // whichplot==1, display ST2Qvec, rather than normwave


    variable calSegments=20;
    variable numSegmentsPerState=15;

    //import the experiment vector
    string inputwavename=fancynum2str(filenum)+"_"+phase+"2DPlot"+basename+"_"+datestr+"_ex1_v";
    wave inputwave=$inputwavename

    //prepare the normalized vector
    string normwavename=inputwavename+"n"
    duplicate/o inputwave $normwavename
    wave normwave=$normwavename


    //display inputwave

    wavestats/Q/R=[0,calSegments-1] inputwave
    //print V_avg
    variable offset=V_avg
    normwave-=offset

    variable Beta1, Beta2, Beta12
    variable dBeta1, dBeta2, dBeta12

    wavestats/Q/R=[0,calSegments/4-1] normwave
    variable M1=V_avg
    variable dM1=V_sdev/sqrt(calSegments/4);
    wavestats/Q/R=[calSegments/4,2*calSegments/4-1] normwave
    variable M2=V_avg
    variable dM2=V_sdev/sqrt(calSegments/4);
    wavestats/Q/R=[2*calSegments/4,3*calSegments/4-1] normwave
    variable M3=V_avg
    variable dM3=V_sdev/sqrt(calSegments/4);
    wavestats/Q/R=[3*calSegments/4,4*calSegments/4-1] normwave
    variable M4=V_avg
    variable dM4=V_sdev/sqrt(calSegments/4);

    Beta1=(M1+M3)/2;
    Beta2=((M1+M2)/2);
    Beta12=((M1+M4)/2);

    dBeta1=sqrt(dM1^2+dM3^2)/2;
    dBeta2=sqrt(dM1^2+dM2^2)/2;
    dBeta12=sqrt(dM1^2+dM4^2)/2;


    //print (M1+M3)/2,  -(M2+M4)/2
    //print (M1+M2)/2,  -(M3+M4)/2
    //print -(M1+M4)/2, (M2+M3)/2

    if(donorm==1)       //normalize
        variable factor=(Beta1);
        normwave/=factor;
        Beta1/=factor;
        Beta2/=factor;
        Beta12/=factor;
        dBeta1/=factor;
        dBeta2/=factor;
        dBeta12/=factor;
        offset/=factor
    endif

    NVAR bequiet

    if(bequiet==0)
        printf  "Beta1=%.4g +/- %.2g,   Beta2=%.4g +/- %.2g,    Beta12=%.4g +/- %.2g        (%.2f, %.2f, %.2f)   offset=%.4g\r", Beta1, dBeta1,  Beta2, dBeta2,   Beta12, dBeta12,   dBeta1/Beta1, dBeta2/Beta2, dBeta12/Beta12, offset
    endif



    make/o/n=(3+numSegmentsPerState) ST2Qvec=NaN
    ST2Qvec[0]=Beta1;
    ST2Qvec[1]=Beta2;
    ST2Qvec[2]=Beta12;

if(opcode==0)
    ST2Qvec[3,3+numSegmentsPerState-1]=normwave[calSegments+(p-3)]
elseif(opcode==1)
    ST2Qvec[3,3+numSegmentsPerState-1]=(normwave[calSegments+2*(p-3)]+normwave[calSegments+2*(p-3)+1])/2
else
    ST2Qvec[3]=     (normwave[calSegments+0]+normwave[calSegments+1])/2;
    ST2Qvec[4]=     (normwave[calSegments+2]+normwave[calSegments+3])/2;
    ST2Qvec[5]=     (normwave[calSegments+4]+normwave[calSegments+5])/2;
    ST2Qvec[6]=     (normwave[calSegments+6]+normwave[calSegments+7])/2;

    ST2Qvec[7]=     (normwave[calSegments+  8]+normwave[calSegments+  9]+normwave[calSegments+10]+normwave[calSegments+11]+normwave[calSegments+12]+normwave[calSegments+13]+normwave[calSegments+14]+normwave[calSegments+15])/8;
    ST2Qvec[8]=     (normwave[calSegments+16]+normwave[calSegments+17]+normwave[calSegments+18]+normwave[calSegments+19]+normwave[calSegments+20]+normwave[calSegments+21]+normwave[calSegments+22]+normwave[calSegments+23])/8;

    ST2Qvec[9]=     (normwave[calSegments+24]+normwave[calSegments+25])/2;
    ST2Qvec[10]=    (normwave[calSegments+26]+normwave[calSegments+27])/2;

    ST2Qvec[11]=    (normwave[calSegments+28]+normwave[calSegments+29]+normwave[calSegments+30]+normwave[calSegments+31]+normwave[calSegments+32]+normwave[calSegments+33]+normwave[calSegments+34]+normwave[calSegments+35])/8;
    ST2Qvec[12]=    (normwave[calSegments+36]+normwave[calSegments+37]+normwave[calSegments+38]+normwave[calSegments+39]+normwave[calSegments+40]+normwave[calSegments+41]+normwave[calSegments+42]+normwave[calSegments+43])/8;

    ST2Qvec[13]=    (normwave[calSegments+44]+normwave[calSegments+45])/2;
    ST2Qvec[14]=    (normwave[calSegments+46]+normwave[calSegments+47])/2;
    ST2Qvec[15]=    (normwave[calSegments+48]+normwave[calSegments+49])/2;
    ST2Qvec[16]=    (normwave[calSegments+50]+normwave[calSegments+51])/2;
    ST2Qvec[17]=    (normwave[calSegments+52]+normwave[calSegments+53])/2;
endif



    wave inputwave=$inputwavename
    // Save Chi matrices to the tomopath
    string ST2Qvecfilename1="ST2Qvec"+phase+".txt"
    string ST2Qvecfilename2="ST2Qvec"+".txt"
    Save/O/G/M="\r\n"/P=tomopath ST2Qvec as ST2Qvecfilename1
    Save/O/G/M="\r\n"/P=tomopath ST2Qvec as ST2Qvecfilename2

    string ST2Qvecname="ST2Qvec_"+fancynum2str(filenum)+"_"+phase+"_"+basename+"_"+datestr;
    duplicate/O ST2Qvec $ST2Qvecname
    WAVE thisST2Qvec=$ST2Qvecname

    //setscale/P x,-2, 1, thisST2Qvec

    NVAR doplots
    if(doplots==1)
        display;
    endif
    if(doplots>0)
        if(whichplot==0)
            appendtograph  normwave
        else
            appendtograph thisST2Qvec
        endif
        fixl(); autocolor(); gridon(2); dotize();
        ModifyGraph nticks(left)=5
        label bottom "Experiment Number"
        label left "Amplitude"
        if(donorm==1)
            SetAxis left -3,3
        endif
    endif


    // Copy the Beta and dBeta values into the BetaVec wave used by other functions, such as analyzeRFsweep
    WAVE BetaVec
    BetaVec[0]=Beta1;
    BetaVec[1]=Beta2;
    BetaVec[2]=Beta12;
    BetaVec[3]=dBeta1;
    BetaVec[4]=dBeta2;
    BetaVec[5]=dBeta12;

end



// Function: getCHSHrawFAST
// -------------------------------------------
function getCHSHrawFAST(filenumstart, filenumend, dn, basename, phase, datestr,type,signloc)
    variable filenumstart
    variable filenumend
    variable dn
    string basename
    string phase
    string datestr
    string type
    variable signloc

    variable numRuns=1+floor(abs(filenumend-filenumstart)/dn)

    // Make copies of vectors with run-specific names
    string  ST2QvecPname="ST2QvecP_"+fancynum2str(filenumstart)+"_"+phase+"_"+basename+"_"+datestr;
    string  ST2QvecMname="ST2QvecM_"+fancynum2str(filenumstart)+"_"+phase+"_"+basename+"_"+datestr;
    WAVE    thisST2QvecP=$ST2QvecPname
    WAVE    thisST2QvecM=$ST2QvecMname

    duplicate/o thisST2QvecP ST2QvecPavg;
    duplicate/o thisST2QvecM ST2QvecMavg;
    WAVE ST2QvecPavg
    WAVE ST2QvecMavg
    ST2QvecPavg=0;
    ST2QvecMavg=0;

    variable Beta1
    variable Beta2
    variable Beta12


    variable i=0
    variable currfilenum
    Do
        currfilenum=filenumstart+i*dn;
        ST2QvecPname="ST2QvecP_"+fancynum2str(currfilenum)+"_"+phase+"_"+basename+"_"+datestr;
        ST2QvecMname="ST2QvecM_"+fancynum2str(currfilenum)+"_"+phase+"_"+basename+"_"+datestr;
        WAVE    thisST2QvecP=$ST2QvecPname
        WAVE    thisST2QvecM=$ST2QvecMname
        ST2QvecPavg[]+=thisST2QVecP[p]
        ST2QvecMavg[]+=thisST2QVecM[p]
        i+=1
    While(i<numRuns)
    ST2QvecPavg/=numRuns;
    ST2QvecMavg/=numRuns;

    Beta1=ST2QvecPavg[0];
    Beta2=ST2QvecPavg[1];
    Beta12=ST2QvecPavg[2];

    make/o/n=(16) ST2QPauliVec=Nan;
    ST2QPauliVec[0]=1;                                                                                                      //ii
    ST2QPauliVec[1]=1/2*(ST2QvecPavg(8)+ST2QvecPavg(11))/(-2*Beta1)+1/2*(ST2QvecMavg(8)+ST2QvecMavg(11))/(2*Beta1);         //xi
    ST2QPauliVec[2]=1/2*(ST2QvecPavg(4)+ST2QvecPavg(7))/(2*Beta1)+1/2*(ST2QvecMavg(4)+ST2QvecMavg(7))/(-2*Beta1);           //yi
    ST2QPauliVec[3]=1/2*(ST2QvecPavg(1)+ST2QvecPavg(3))/(2*Beta1)+1/2*(ST2QvecMavg(1)+ST2QvecMavg(3))/(2*Beta1);                //zi
    ST2QPauliVec[4]=1/2*(ST2QvecPavg(14)+ST2QvecPavg(15))/(-2*Beta2)+1/2*(ST2QvecPavg(14)+ST2QvecPavg(15))/(2*Beta2);           //ix
    ST2QPauliVec[5]=(ST2QvecPavg(10)+ST2QvecMavg(10))/(2*Beta12);                                                           //xx
    ST2QPauliVec[6]=(ST2QvecPavg(6)+ST2QvecMavg(6))/(-2*Beta12);                                                                //yx
    ST2QPauliVec[7]=1/2*(-ST2QvecPavg(14)+ST2QvecPavg(15))/(2*Beta12)+1/2*(ST2QvecMavg(14)-ST2QvecMavg(15))/(2*Beta12);     //zx
    ST2QPauliVec[8]=1/2*(ST2QvecPavg(12)+ST2QvecPavg(13))/(2*Beta2)+1/2*(ST2QvecMavg(12)+ST2QvecMavg(13))/(-2*Beta2)            //iy
    ST2QPauliVec[9]=(ST2QvecPavg(9)+ST2QvecMavg(9))/(-2*Beta12);                                                                //xy
    ST2QPauliVec[10]=(ST2QvecPavg(5)+ST2QvecMavg(5))/(2*Beta12);                                                            //yy
    ST2QPauliVec[11]=1/2*(ST2QvecPavg(12)-ST2QvecPavg(13))/(2*Beta12)+1/2*(-ST2QvecMavg(12)+ST2QvecMavg(13))/(2*Beta12);        //zy
    ST2QPauliVec[12]=1/2*(ST2QvecPavg(1)+ST2QvecPavg(2))/(2*Beta2)+1/2*(ST2QvecMavg(1)+ST2QvecMavg(2))/(2*Beta2)                //iz

    ST2QPauliVec[13]=1/2*(-ST2QvecPavg(8)+ST2QvecPavg(11))/(2*Beta12)+1/2*(ST2QvecMavg(8)-ST2QvecMavg(11))/(2*Beta12);      //xz
    //ST2QPauliVec[13]=(ST2QvecPavg(8)+ST2QvecMavg(11))/(-2*Beta12);                                                            //xz
    //ST2QPauliVec[13]=(ST2QvecPavg(11)+ST2QvecMavg(8))/(2*Beta12);                                                         //xz


    ST2QPauliVec[14]=1/2*(ST2QvecPavg(4)-ST2QvecPavg(7))/(2*Beta12)+1/2*(-ST2QvecMavg(4)+ST2QvecMavg(7))/(2*Beta12);            //yz

    ST2QPauliVec[15]=1/2*(ST2QvecPavg(2)+ST2QvecPavg(3))/(-2*Beta12)+1/2*(ST2QvecMavg(2)+ST2QvecMavg(3))/(-2*Beta12);           //zz
    //ST2QPauliVec[15]=(ST2QvecPavg(2)+ST2QvecPavg(3))/(-2*Beta12)  //zz
    //ST2QPauliVec[15]=(ST2QvecMavg(2)+ST2QvecMavg(3))/(-2*Beta12)  //zz

    //ST2QPauliVec*=Beta12;

    string ST2QPaulivecname="ST2QPaulivecH_000"+"_"+phase+"_"+basename+"_"+datestr;
    duplicate/o ST2QPauliVec $ST2QPaulivecname

    return getCHSHraw(datestr, basename, phase,0, type,signloc,"H")

end

// Function: getCHSHrawvecFAST
// ----------------------------------------------
function getCHSHrawvecFAST(datestr, base, phase,startrun, endrun, dn, startval, endval, type,signloc, numavgs)
string datestr
string base
string phase
variable startrun
variable endrun
variable dn
variable startval
variable endval
string type
variable signloc
variable numavgs


string Xlabel="Rotation angle (deg)"
if (startval==0 && endval==0)
    startval=startrun;
    endval=endrun;
    Xlabel="File number"
endif

variable startnum=min(startrun,endrun);
variable endnum=max(startrun,endrun);
variable numruns=floor((endnum-startnum+1)/dn/numavgs);


make/o/n=(16)/D WeightVec=0;
string CHSHstr;

string thisST2QPaulivecname

if(stringmatch("xyxy",type)==1)
    WeightVec[5]=(-1)^(signloc==0)
    WeightVec[9]=(-1)^(signloc==1)
    WeightVec[6]=(-1)^(signloc==2)
    WeightVec[10]=(-1)^(signloc==3)
    CHSHstr="xy"+num2str(signloc);
elseif(stringmatch("xzxz",type)==1)
    WeightVec[5]=(-1)^(signloc==0)
    WeightVec[13]=(-1)^(signloc==1)
    WeightVec[7]=(-1)^(signloc==2)
    WeightVec[15]=(-1)^(signloc==3)
    CHSHstr="xz"+num2str(signloc);
 elseif(stringmatch("yzyz",type)==1)
    WeightVec[10]=(-1)^(signloc==0)
    WeightVec[14]=(-1)^(signloc==1)
    WeightVec[11]=(-1)^(signloc==2)
    WeightVec[15]=(-1)^(signloc==3)
    CHSHstr="yz"+num2str(signloc);
 elseif(stringmatch("conc",type)==1)
    CHSHstr="co";
endif


string CHSHvecname="rawCHSH"+CHSHstr+"_"+datestr+"_"+phase+"_"+fancynum2str(startnum)+"_"+fancynum2str(endnum)+"_"+num2str(numavgs);
make/o/n=(numruns)  $CHSHvecname=NaN
wave CHSHvec=$CHSHvecname
setscale/P x, startval, dn*numavgs, CHSHvec

variable i=0;
variable currnum;

Do
    currnum=startnum+i*numavgs*dn;
    CHSHvec[i]= getCHSHrawFAST(currnum,currnum+dn*(numavgs-1), dn, base, phase,datestr, type,signloc)
    i+=1
While(i<numRuns);

string CHSHvecname2="rawCHSH"+CHSHstr+"vec"
duplicate/o CHSHvec $CHSHvecname2
wave genericCHSHvec=$CHSHvecname2

// plot according to doplots
NVAR doplots
if(doplots==1)
    display
endif
if(doplots>=1)
    appendtograph genericCHSHvec
    fixg();
    label bottom Xlabel;
    //lbllchsh();
    gridoff();
    SetAxis left -3,3
    SetAxis bottom startval, endval
    ModifyGraph userticks(left)={ticksCHSH,ticknamesCHSH}
    gridon(1)
    ModifyGraph gridRGB(left)=(0,0,0)
    ModifyGraph margin(right)=10

endif

end


//function: getallCHSHrawvecsFAST
// --------------------------------------------------------
function getallCHSHrawvecsFAST(datestr, base, phase, startrun,endrun, dn, startval, endval, numavgs)
string datestr
string base
string phase
variable startrun
variable endrun
variable dn
variable startval
variable endval
variable numavgs



NVAR doplots
variable olddoplots=doplots

doplots=0;
//getCHSHrawvecFAST(datestr,base,phase,startrun,endrun,dn,startval,endval,"xyxy",0,numavgs)
//getCHSHrawvecFAST(datestr,base,phase,startrun,endrun,dn,startval,endval,"xyxy",1,numavgs)
//getCHSHrawvecFAST(datestr,base,phase,startrun,endrun,dn,startval,endval,"xyxy",2,numavgs)
//getCHSHrawvecFAST(datestr,base,phase,startrun,endrun,dn,startval,endval,"xyxy",3,numavgs)

getCHSHrawvecFAST(datestr,base,phase,startrun,endrun,dn,startval,endval,"xzxz",0,numavgs)
getCHSHrawvecFAST(datestr,base,phase,startrun,endrun,dn,startval,endval,"xzxz",1,numavgs)
getCHSHrawvecFAST(datestr,base,phase,startrun,endrun,dn,startval,endval,"xzxz",2,numavgs)
getCHSHrawvecFAST(datestr,base,phase,startrun,endrun,dn,startval,endval,"xzxz",3,numavgs)

//getCHSHrawvecFAST(datestr,base,phase,startrun,endrun,dn,startval,endval,"yzyz",0,numavgs)
//getCHSHrawvecFAST(datestr,base,phase,startrun,endrun,dn,startval,endval,"yzyz",1,numavgs)
//getCHSHrawvecFAST(datestr,base,phase,startrun,endrun,dn,startval,endval,"yzyz",2,numavgs)
//getCHSHrawvecFAST(datestr,base,phase,startrun,endrun,dn,startval,endval,"yzyz",3,numavgs)

//getCHSHrawvecFAST(datestr,base,phase,startrun,endrun,dn,startval,endval,"conc",3,numavgs)

doplots=olddoplots

end




// Function: FastCHSH
// ------------------------------
function FastCHSH(filenum, basename, phase, datestr, opcode)
    variable filenum
    string basename
    string phase
    string datestr
    variable opcode     // opcode==0: CHSH type xy
                        // opcode==1: CHSH type xz
                        // opcode==2: CHSH type yz

    variable whichplot=2;

    variable calSegments=8;
    //variable numExpsTotal=calSegments+14*2;

    //import the experiment vector
    string inputwavename=fancynum2str(filenum)+"_"+phase+"2DPlot"+basename+"_"+datestr+"_ex1_v";
    wave inputwave=$inputwavename


    variable Beta1, Beta2, Beta12
    variable dBeta1, dBeta2, dBeta12
    variable M1, M2, M3, M4
    variable dM1, dM2, dM3, dM4
    variable offset

    //prepare an offset/normalized vector
    duplicate/o inputwave normwave
    WAVE normwave

    wavestats/Q/R=[0,calSegments-1] inputwave
    offset=V_avg
    normwave-=offset

    // define some strings
    string ST2Qvecfilename1
    string ST2Qvecfilename2
    string ST2QPaulivecfilename1
    string ST2QPaulivecfilename2
    string ST2QvecPname
    string ST2QPaulivecPname
    string ST2QvecMname
    string ST2QPaulivecMname
    string ST2QPaulivecHname


    NVAR doplots
    NVAR bequiet

    WAVE Betavec

    WAVE ST2QPauliVec

    wavestats/Q/R=[0,calSegments/4-1] normwave
    M1=V_avg
    dM1=V_sdev/sqrt(calSegments/4);
    wavestats/Q/R=[calSegments/4,2*calSegments/4-1] normwave
    M2=V_avg
    dM2=V_sdev/sqrt(calSegments/4);
    wavestats/Q/R=[2*calSegments/4,3*calSegments/4-1] normwave
    M3=V_avg
    dM3=V_sdev/sqrt(calSegments/4);
    wavestats/Q/R=[3*calSegments/4,4*calSegments/4-1] normwave
    M4=V_avg
    dM4=V_sdev/sqrt(calSegments/4);

    Beta1=(M1+M3)/2;
    Beta2=((M1+M2)/2);
    Beta12=((M1+M4)/2);

    dBeta1=sqrt(dM1^2+dM3^2)/2;
    dBeta2=sqrt(dM1^2+dM2^2)/2;
    dBeta12=sqrt(dM1^2+dM4^2)/2;

    // Copy the Beta and dBeta values into the BetaVec wave used by other functions, such as analyzeRFsweep
    BetaVec[0]=Beta1;
    BetaVec[1]=Beta2;
    BetaVec[2]=Beta12;
    BetaVec[3]=dBeta1;
    BetaVec[4]=dBeta2;
    BetaVec[5]=dBeta12;

    //normwave+=18;

    if(bequiet==0)
        printf  " from CHSH Run : Beta1=%.4g +/- %.2g,  Beta2=%.4g +/- %.2g,    Beta12=%.4g +/- %.2g    (%.2f, %.2f, %.2f)   offset=%.4g\r", Beta1, dBeta1,  Beta2, dBeta2,   Beta12, dBeta12,   dBeta1/Beta1, dBeta2/Beta2, dBeta12/Beta12, offset
    endif


    make/o/n=(19) ST2Qvec=NaN
    setscale/P x, -2, 1, ST2Qvec;

    ST2Qvec[0]=Beta1;
    ST2Qvec[1]=Beta2;
    ST2Qvec[2]=Beta12;

    if(opcode==0)
        ST2Qvec[3]=     nan;
        ST2Qvec[4]=     (normwave[calSegments+0]+normwave[calSegments+1])/2;
        ST2Qvec[5]=     (normwave[calSegments+2]+normwave[calSegments+3])/2;
        ST2Qvec[6]=     nan;
        ST2Qvec[7]=     nan;
        ST2Qvec[8]=     nan;
        ST2Qvec[9]=     nan;
        ST2Qvec[10]=    (normwave[calSegments+4]+normwave[calSegments+5])/2;
        ST2Qvec[11]=    nan;
        ST2Qvec[12]=    (normwave[calSegments+6]+normwave[calSegments+7])/2
        ST2Qvec[13]=    (normwave[calSegments+8]+normwave[calSegments+9])/2;
           ST2Qvec[14]= nan;
        ST2Qvec[15]=    nan;
        ST2Qvec[16]=    (normwave[calSegments+10]+normwave[calSegments+11])/2;
        ST2Qvec[17]=    (normwave[calSegments+12]+normwave[calSegments+13])/2;
        ST2Qvec[18]=1
    endif

    make/o/n=(16) tempvec
    tempvec[]=ST2Qvec[3+p];

    // Finally convert to measurements into the Pauli basis
    make/o/n=(16,16) ObsFromPauli=0;

    ObsFromPauli[15][0]=1;

    ObsFromPauli[0][3]=Beta1;
    ObsFromPauli[1][3]=-Beta1;
    ObsFromPauli[2][3]=Beta1;
    ObsFromPauli[3][2]=Beta1;
    ObsFromPauli[4][2]=Beta1;
    ObsFromPauli[5][2]=Beta1;
    ObsFromPauli[6][2]=Beta1;
    ObsFromPauli[7][1]=-Beta1;
    ObsFromPauli[8][1]=-Beta1;
    ObsFromPauli[9][1]=-Beta1;
    ObsFromPauli[10][1]=    -Beta1;
    ObsFromPauli[11][3]=    Beta1;
    ObsFromPauli[12][3]=    -Beta1;
    ObsFromPauli[13][3]=Beta1;
    ObsFromPauli[14][3]=-Beta1;

    ObsFromPauli[0][12]=Beta2
    ObsFromPauli[1][12]=Beta2
    ObsFromPauli[2][12]=-Beta2
    ObsFromPauli[3][12]=Beta2
    ObsFromPauli[4][8]=Beta2
    ObsFromPauli[5][4]=-Beta2
    ObsFromPauli[6][12]=-Beta2
    ObsFromPauli[7][12]=Beta2
    ObsFromPauli[8][8]=Beta2
    ObsFromPauli[9][4]=-Beta2
    ObsFromPauli[10][12]=-Beta2
    ObsFromPauli[11][8]=Beta2
    ObsFromPauli[12][8]=Beta2
    ObsFromPauli[13][4]=-Beta2
    ObsFromPauli[14][4]=-Beta2

    ObsFromPauli[0][15]=Beta12
    ObsFromPauli[1][15]=-Beta12
    ObsFromPauli[2][15]=-Beta12
    ObsFromPauli[3][14]=Beta12
    ObsFromPauli[4][10]=Beta12
    ObsFromPauli[5][6]=-Beta12
    ObsFromPauli[6][14]=-Beta12
    ObsFromPauli[7][13]=-Beta12
    ObsFromPauli[8][9]=-Beta12
    ObsFromPauli[9][5]=Beta12
    ObsFromPauli[10][13]=Beta12
    ObsFromPauli[11][11]=Beta12
    ObsFromPauli[12][11]=-Beta12
    ObsFromPauli[13][7]=-Beta12
    ObsFromPauli[14][7]=Beta12

    MatrixOP/O PauliFromObs=Inv(ObsFromPauli)
    MatrixOP/O ST2QPauliVec= PauliFromObs x tempvec
    setscale/P x, 1, 1, ST2QPauliVec


    // Make copies of vectors with run-specific names
    ST2QvecPname="ST2QvecP_"+fancynum2str(filenum)+"_"+phase+"_"+basename+"_"+datestr;
    duplicate/O ST2Qvec $ST2QvecPname
    WAVE thisST2QvecP=$ST2QvecPname

    ST2QPaulivecPname="ST2QPaulivecP_"+fancynum2str(filenum)+"_"+phase+"_"+basename+"_"+datestr;
    duplicate/O ST2QPaulivec $ST2QPaulivecPname
    WAVE thisST2QPaulivecP=$ST2QPaulivecPname

    make/o/n=(30) PQ1vecD=NaN
    make/o/n=(30) PQ2vecD=Nan
    make/o/n=(30) PQ12vecD=Nan

    setscale/P x, 3,1, PQ1vecD
    setscale/P x, 3,1, PQ2vecD
    setscale/P x, 3,1, PQ12vecD

    PQ1vecD[0]=ST2QPaulivec[1];
    PQ1vecD[2]=ST2QPaulivec[2];
    PQ1vecD[4]=ST2QPaulivec[3];

    PQ2vecD[6]=ST2QPaulivec[4]
    PQ2vecD[8]=ST2QPaulivec[8]
    PQ2vecD[10]=ST2QPaulivec[12]

    PQ12vecD[12]=ST2QPaulivec[9]
    PQ12vecD[14]=ST2QPaulivec[13]
    PQ12vecD[16]=ST2QPaulivec[6]
    PQ12vecD[18]=ST2QPaulivec[14]
    PQ12vecD[20]=ST2QPaulivec[7]
    PQ12vecD[22]=ST2QPaulivec[11]
    PQ12vecD[24]=ST2QPaulivec[5]
    PQ12vecD[26]=ST2QPaulivec[10]
    PQ12vecD[28]=ST2QPaulivec[15]

    duplicate/o PQ1vecD PQ1vecDP
    duplicate/o PQ2vecD PQ2vecDP
    duplicate/o PQ12vecD PQ12vecDP

    PQ1vecD=NaN;
    PQ2vecD=NaN;
    PQ12vecD=NaN;
    duplicate/o PQ1vecD PQ1vecDM
    duplicate/o PQ2vecD PQ2vecDM
    duplicate/o PQ12vecD PQ12vecDM


    make/o/n=(19) ST2Qvec=NaN
    setscale/P x, -2, 1, ST2Qvec;

    ST2Qvec[0]=Beta1;
    ST2Qvec[1]=Beta2;
    ST2Qvec[2]=Beta12;

    if(opcode==0)
        ST2Qvec[3]=     nan;
        ST2Qvec[4]=     (normwave[calSegments+14]+normwave[calSegments+15])/2;
        ST2Qvec[5]=     (normwave[calSegments+16]+normwave[calSegments+17])/2;
        ST2Qvec[6]=     nan;
        ST2Qvec[7]=     nan;
        ST2Qvec[8]=     nan;
        ST2Qvec[9]=     nan;
        ST2Qvec[10]=    (normwave[calSegments+18]+normwave[calSegments+19])/2;
        ST2Qvec[11]=    nan;
        ST2Qvec[12]=    (normwave[calSegments+20]+normwave[calSegments+21])/2
        ST2Qvec[13]=    (normwave[calSegments+22]+normwave[calSegments+23])/2;
           ST2Qvec[14]= nan;
        ST2Qvec[15]=    nan;
        ST2Qvec[16]=    (normwave[calSegments+24]+normwave[calSegments+25])/2;
        ST2Qvec[17]=    (normwave[calSegments+26]+normwave[calSegments+27])/2;
        ST2Qvec[18]=1
    endif


        make/o/n=(16) tempvec
        tempvec[]=ST2Qvec[3+p];

        // Finally convert to measurements into the Pauli basis
        make/o/n=(16,16) ObsFromPauli=0;

        ObsFromPauli[15][0]=1;

        ObsFromPauli[0][3]=Beta1;
        ObsFromPauli[1][3]=-Beta1;
        ObsFromPauli[2][3]=Beta1;
        ObsFromPauli[3][2]=-Beta1;
        ObsFromPauli[4][2]=-Beta1;
        ObsFromPauli[5][2]=-Beta1;
        ObsFromPauli[6][2]=-Beta1;
        ObsFromPauli[7][1]=Beta1;
        ObsFromPauli[8][1]=Beta1;
        ObsFromPauli[9][1]=Beta1;
        ObsFromPauli[10][1]=Beta1;
        ObsFromPauli[11][3]=Beta1;
        ObsFromPauli[12][3]=    -Beta1;
        ObsFromPauli[13][3]=Beta1;
        ObsFromPauli[14][3]=-Beta1;

        ObsFromPauli[0][12]=Beta2
        ObsFromPauli[1][12]=Beta2
        ObsFromPauli[2][12]=-Beta2
        ObsFromPauli[3][12]=Beta2
        ObsFromPauli[4][8]=-Beta2
        ObsFromPauli[5][4]=Beta2
        ObsFromPauli[6][12]=-Beta2
        ObsFromPauli[7][12]=Beta2
        ObsFromPauli[8][8]=-Beta2
        ObsFromPauli[9][4]=Beta2
        ObsFromPauli[10][12]=-Beta2
        ObsFromPauli[11][8]=-Beta2
        ObsFromPauli[12][8]=-Beta2
        ObsFromPauli[13][4]=Beta2
        ObsFromPauli[14][4]=Beta2

        ObsFromPauli[0][15]=Beta12
        ObsFromPauli[1][15]=-Beta12
        ObsFromPauli[2][15]=-Beta12
        ObsFromPauli[3][14]=-Beta12
        ObsFromPauli[4][10]=Beta12
        ObsFromPauli[5][6]=-Beta12
        ObsFromPauli[6][14]=Beta12
        ObsFromPauli[7][13]=Beta12
        ObsFromPauli[8][9]=-Beta12
        ObsFromPauli[9][5]=Beta12
        ObsFromPauli[10][13]=-Beta12
        ObsFromPauli[11][11]=-Beta12
        ObsFromPauli[12][11]=Beta12
        ObsFromPauli[13][7]=Beta12
        ObsFromPauli[14][7]=-Beta12

        MatrixOP/O PauliFromObs=Inv(ObsFromPauli)
        MatrixOP/O ST2QPauliVec= PauliFromObs x tempvec
        setscale/P x, 1, 1, ST2QPauliVec


        // Make copies of vectors with run specific names
        ST2QvecMname="ST2QvecM_"+fancynum2str(filenum)+"_"+phase+"_"+basename+"_"+datestr;
        duplicate/O ST2Qvec $ST2QvecMname
        WAVE thisST2QvecM=$ST2QvecMname

        ST2QPaulivecMname="ST2QPaulivecM_"+fancynum2str(filenum)+"_"+phase+"_"+basename+"_"+datestr;
        duplicate/O ST2QPaulivec $ST2QPaulivecMname
        WAVE thisST2QPaulivecM=$ST2QPaulivecMname

        PQ1vecD[1]=ST2QPaulivec[1];
        PQ1vecD[3]=ST2QPaulivec[2];
        PQ1vecD[5]=ST2QPaulivec[3];

        PQ2vecD[7]=ST2QPaulivec[4]
        PQ2vecD[9]=ST2QPaulivec[8]
        PQ2vecD[11]=ST2QPaulivec[12]

        PQ12vecD[13]=ST2QPaulivec[9]
        PQ12vecD[15]=ST2QPaulivec[13]
        PQ12vecD[17]=ST2QPaulivec[6]
        PQ12vecD[19]=ST2QPaulivec[14]
        PQ12vecD[21]=ST2QPaulivec[7]
        PQ12vecD[23]=ST2QPaulivec[11]
        PQ12vecD[25]=ST2QPaulivec[5]
        PQ12vecD[27]=ST2QPaulivec[10]
        PQ12vecD[29]=ST2QPaulivec[15]

        duplicate/o PQ1vecD PQ1vecDM
        duplicate/o PQ2vecD PQ2vecDM
        duplicate/o PQ12vecD PQ12vecDM

ST2QPauliVec=NaN
ST2QPauliVec[0]=1;                                                                                                      //ii
ST2QPauliVec[1]=1/2*(thisST2QvecP(8)+thisST2QvecP(11))/(-2*Beta1)+1/2*(thisST2QvecM(8)+thisST2QvecM(11))/(2*Beta1);         //xi
ST2QPauliVec[2]=1/2*(thisST2QvecP(4)+thisST2QvecP(7))/(2*Beta1)+1/2*(thisST2QvecM(4)+thisST2QvecM(7))/(-2*Beta1);           //yi
ST2QPauliVec[3]=1/2*(thisST2QvecP(1)+thisST2QvecP(3))/(2*Beta1)+1/2*(thisST2QvecM(1)+thisST2QvecM(3))/(2*Beta1);            //zi
ST2QPauliVec[4]=1/2*(thisST2QvecP(14)+thisST2QvecP(15))/(-2*Beta2)+1/2*(thisST2QvecM(14)+thisST2QvecM(15))/(2*Beta2);       //ix
ST2QPauliVec[5]=(thisST2QvecP(10)+thisST2QvecM(10))/(2*Beta12);                                                         //xx
ST2QPauliVec[6]=(thisST2QvecP(6)+thisST2QvecM(6))/(-2*Beta12);                                                          //yx
ST2QPauliVec[7]=1/2*(-thisST2QvecP(14)+thisST2QvecP(15))/(2*Beta12)+1/2*(thisST2QvecM(14)-thisST2QvecM(15))/(2*Beta12);     //zx
ST2QPauliVec[8]=1/2*(thisST2QvecP(12)+thisST2QvecP(13))/(2*Beta2)+1/2*(thisST2QvecM(12)+thisST2QvecM(13))/(-2*Beta2)        //iy
ST2QPauliVec[9]=(thisST2QvecP(9)+thisST2QvecM(9))/(-2*Beta12);                                                          //xy
ST2QPauliVec[10]=(thisST2QvecP(5)+thisST2QvecM(5))/(2*Beta12);                                                          //yy
ST2QPauliVec[11]=1/2*(thisST2QvecP(12)-thisST2QvecP(13))/(2*Beta12)+1/2*(-thisST2QvecM(12)+thisST2QvecM(13))/(2*Beta12);    //zy
ST2QPauliVec[12]=1/2*(thisST2QvecP(1)+thisST2QvecP(2))/(2*Beta2)+1/2*(thisST2QvecM(1)+thisST2QvecM(2))/(2*Beta2)            //iz
ST2QPauliVec[13]=1/2*(-thisST2QvecP(8)+thisST2QvecP(11))/(2*Beta12)+1/2*(thisST2QvecM(8)-thisST2QvecM(11))/(2*Beta12);      //xz
ST2QPauliVec[14]=1/2*(thisST2QvecP(4)-thisST2QvecP(7))/(2*Beta12)+1/2*(-thisST2QvecM(4)+thisST2QvecM(7))/(2*Beta12);            //yz
ST2QPauliVec[15]=1/2*(thisST2QvecP(2)+thisST2QvecP(3))/(-2*Beta12)+1/2*(thisST2QvecM(2)+thisST2QvecM(3))/(-2*Beta12);       //zz

//ST2QPauliVec*=Beta12;

    make/o/n=(16) PQ1vecH=NaN
    make/o/n=(16) PQ2vecH=Nan
    make/o/n=(16) PQ12vecH=Nan

    PQ1vecH[1]=ST2QPaulivec[1];
    PQ1vecH[2]=ST2QPaulivec[2];
    PQ1vecH[3]=ST2QPaulivec[3];

    PQ2vecH[4]=ST2QPaulivec[4]
    PQ2vecH[5]=ST2QPaulivec[8]
    PQ2vecH[6]=ST2QPaulivec[12]

    PQ12vecH[7]=ST2QPaulivec[9]
    PQ12vecH[8]=ST2QPaulivec[13]
    PQ12vecH[9]=ST2QPaulivec[6]
    PQ12vecH[10]=ST2QPaulivec[14]
    PQ12vecH[11]=ST2QPaulivec[7]
    PQ12vecH[12]=ST2QPaulivec[11]
    PQ12vecH[13]=ST2QPaulivec[5]
    PQ12vecH[14]=ST2QPaulivec[10]
    PQ12vecH[15]=ST2QPaulivec[15]

    setscale /p x, 1, 1, PQ1vecH
    setscale /p x, 1, 1, PQ2vecH
    setscale /p x, 1, 1, PQ12vecH


        // Make copies of vectors with run specific name
        ST2QPaulivecHname="ST2QPaulivecH_"+fancynum2str(filenum)+"_"+phase+"_"+basename+"_"+datestr;
        duplicate/O ST2QPaulivec $ST2QPaulivecHname
        WAVE thisST2QPaulivecH=$ST2QPaulivecHname


    if(doplots==1)
        display;
    endif


    if(doplots>0)
        if(whichplot==0)
            appendtograph  normwave
            appendtograph normwavem
            label bottom "Experiment number"
        elseif(whichplot==1)
            appendtograph thisST2QvecP
            appendtograph thisST2QvecM
            label bottom "Raw measurement number"
            ModifyGraph zero(bottom)=1
            WAVE TicksST2Q
            WAVE TickNamesST2Q
            ModifyGraph userticks(bottom)={TicksST2Q,TickNamesST2Q}
        elseif(whichplot==2)
            appendtograph thisST2QPaulivecP
            appendtograph thisST2QPaulivecM
            appendtograph thisST2QPaulivecH
            label bottom "Pauli measurement"
            SetAxis left -1.3,1.3
            WAVE TicksST2QP
            WAVE TickNamesST2QP
            ModifyGraph userticks(bottom)={TicksST2QP,TickNamesST2QP}
        endif

        fixl(); autocolor(); gridon(2); dotize();
        fixg();
        ModifyGraph nticks(left)=5
        label left "Mean Value"
    endif
end




Window Graph_Pvec2Q() : Graph
    PauseUpdate; Silent 1       // building window...
    Display /W=(174.75,41.75,569.25,239.75) Pvec2QP,Pvec2QM,Pvec2QH
    ModifyGraph userticks(bottom)={TicksST2QP,TickNamesST2QP}
    ModifyGraph margin(top)=15,margin(right)=15
    ModifyGraph mode=4
    ModifyGraph marker=19
    ModifyGraph rgb(Pvec2QP)=(65280,0,0),rgb(Pvec2QM)=(21760,65280,0),rgb(Pvec2QH)=(0,43520,65280)
    ModifyGraph msize=2
    ModifyGraph grid=1
    ModifyGraph tick=2
    ModifyGraph zero(left)=1
    ModifyGraph mirror=1
    ModifyGraph fSize=10
    ModifyGraph gridRGB=(0,0,0)
    ModifyGraph btLen=2
    Label left "Mean Value"
    Label bottom "Pauli operator"
    SetAxis left -1.3,1.3
    Legend/C/N=text0/J/F=0/B=1/A=MC/X=-23.19/Y=-19.78/E=2 "\\s(Pvec2QP) Pvec2QP\r\\s(Pvec2QM) Pvec2QM\r\\s(Pvec2QH) Pvec2QH"
EndMacro

