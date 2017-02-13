#pragma rtGlobals=1     // Use modern global access method.

// Function: setupPauliTheo2Q
// -----------------------------------------
function setupPauliTheo2Q()
make/o/n=(4)/C PsiT_2Q
toground2Q();
make/o/n=(4,4)/C RhoT_2Q;
make/o/n=(2) CurrVals_PauliTheo2Q=NaN

MakePaulioperators2Q();

SetupQPT2Q();


make/o/n=4 CHSHxy=NaN;
make/o/n=4 CHSHxz=NaN;
make/o/n=4 CHSHyx=NaN;
make/o/n=4 CHSHyz=NaN;
make/o/n=4 CHSHzx=NaN;
make/o/n=4 CHSHzy=NaN;
make/o/n=4 Witness=NaN;

make/o/n=(3,3) MichelQmat=NaN;

UpdateRhoTheo2Q();
UpdatePauliTheo2Q();


make/o/n=7 TicksCHSH
 TicksCHSH[0]=-2*sqrt(2);
 TicksCHSH[1]=-2;
 TicksCHSH[2]=-sqrt(2);
 TicksCHSH[3]=0;
 TicksCHSH[4]=sqrt(2);
 TicksCHSH[5]=2;
 TicksCHSH[6]=2*sqrt(2);;

make/o/n=(7,2)/T TickNamesCHSH
setDimLabel 1,1,'Tick Type',TickNamesCHSH
TickNamesCHSH[0][1]="Major"
TickNamesCHSH[1][1]="Major"
TickNamesCHSH[2][1]="Major"
TickNamesCHSH[3][1]="Major"
TickNamesCHSH[4][1]="Major"
TickNamesCHSH[5][1]="Major"
TickNamesCHSH[6][1]="Major"

TickNamesCHSH[0][0]="-2.82"
TickNamesCHSH[1][0]="-2.00"
TickNamesCHSH[2][0]="-1.41"
TickNamesCHSH[3][0]="0"
TickNamesCHSH[4][0]="1.41"
TickNamesCHSH[5][0]="2.00"
TickNamesCHSH[6][0]="2.82"


make/o/n=17 TicksCHSHAngles
 TicksCHSHAngles[0]=-360;
 TicksCHSHAngles[1]=-315
 TicksCHSHAngles[2]=-270
 TicksCHSHAngles[3]=-225
 TicksCHSHAngles[4]=-180
 TicksCHSHAngles[5]=-135
 TicksCHSHAngles[6]=-90
 TicksCHSHAngles[7]=-45
 TicksCHSHAngles[8]=0
 TicksCHSHAngles[9]=45
 TicksCHSHAngles[10]=90
 TicksCHSHAngles[11]=135
 TicksCHSHAngles[12]=180
 TicksCHSHAngles[13]=225
 TicksCHSHAngles[14]=270
 TicksCHSHAngles[15]=315
 TicksCHSHAngles[16]=360

 make/o/n=(17,2)/T TickNamesCHSHAngles
setDimLabel 1,1,'Tick Type',TickNamesCHSHAngles
TickNamesCHSHAngles[0][1]="Major"
TickNamesCHSHAngles[1][1]="Minor"
TickNamesCHSHAngles[2][1]="Major"
TickNamesCHSHAngles[3][1]="Minor"
TickNamesCHSHAngles[4][1]="Major"
TickNamesCHSHAngles[5][1]="Minor"
TickNamesCHSHAngles[6][1]="Major"
TickNamesCHSHAngles[7][1]="Minor"
TickNamesCHSHAngles[8][1]="Major"
TickNamesCHSHAngles[9][1]="Minor"
TickNamesCHSHAngles[10][1]="Major"
TickNamesCHSHAngles[11][1]="Minor"
TickNamesCHSHAngles[12][1]="Major"
TickNamesCHSHAngles[13][1]="Minor"
TickNamesCHSHAngles[14][1]="Major"
TickNamesCHSHAngles[15][1]="Minor"
TickNamesCHSHAngles[16][1]="Major"

 TickNamesCHSHAngles[0][0]="-360";
 TickNamesCHSHAngles[1][0]="";
 TickNamesCHSHAngles[2][0]="-270";
 TickNamesCHSHAngles[3][0]="";
 TickNamesCHSHAngles[4][0]="-180";
 TickNamesCHSHAngles[5][0]="";
 TickNamesCHSHAngles[6][0]="-90";
 TickNamesCHSHAngles[7][0]="";
 TickNamesCHSHAngles[8][0]="0";
 TickNamesCHSHAngles[9][0]="";
 TickNamesCHSHAngles[10][0]="90";
 TickNamesCHSHAngles[11][0]="";
 TickNamesCHSHAngles[12][0]="180";
 TickNamesCHSHAngles[13][0]="";
 TickNamesCHSHAngles[14][0]="270";
 TickNamesCHSHAngles[15][0]="";
 TickNamesCHSHAngles[16][0]="360";



 make/o/n=16 TicksPauliRod
 TicksPauliRod[0]=1
 TicksPauliRod[1]=2
 TicksPauliRod[2]=3
 TicksPauliRod[3]=4
 TicksPauliRod[4]=5
 TicksPauliRod[5]=6
 TicksPauliRod[6]=7
 TicksPauliRod[7]=8
 TicksPauliRod[8]=9
 TicksPauliRod[9]=10
 TicksPauliRod[10]=11
 TicksPauliRod[11]=12
 TicksPauliRod[12]=13
 TicksPauliRod[13]=14
 TicksPauliRod[14]=15
 TicksPauliRod[15]=16

 make/o/n=(16,2)/T TickNamesPauliRod
setDimLabel 1,1,'Tick Type',TickNamesPauliRod
TickNamesPauliRod[0][1]="Major"
TickNamesPauliRod[1][1]="Major"
TickNamesPauliRod[2][1]="Major"
TickNamesPauliRod[3][1]="Major"
TickNamesPauliRod[4][1]="Major"
TickNamesPauliRod[5][1]="Major"
TickNamesPauliRod[6][1]="Major"
TickNamesPauliRod[7][1]="Major"
TickNamesPauliRod[8][1]="Major"
TickNamesPauliRod[9][1]="Major"
TickNamesPauliRod[10][1]="Major"
TickNamesPauliRod[11][1]="Major"
TickNamesPauliRod[12][1]="Major"
TickNamesPauliRod[13][1]="Major"
TickNamesPauliRod[14][1]="Major"
TickNamesPauliRod[15][1]="Major"

TickNamesPauliRod[0][0]="ii";
 TickNamesPauliRod[1][0]="xi";
 TickNamesPauliRod[2][0]="yi";
 TickNamesPauliRod[3][0]="zi";
 TickNamesPauliRod[4][0]="ix";
 TickNamesPauliRod[5][0]="iy";
 TickNamesPauliRod[6][0]="iz";
 TickNamesPauliRod[7][0]="xx";
 TickNamesPauliRod[8][0]="xy";
 TickNamesPauliRod[9][0]="xz";
 TickNamesPauliRod[10][0]="yx";
 TickNamesPauliRod[11][0]="yy";
 TickNamesPauliRod[12][0]="yz";
 TickNamesPauliRod[13][0]="zx";
 TickNamesPauliRod[14][0]="zy";
 TickNamesPauliRod[15][0]="zz";

end


// Function: SetupQPT2Q
// --------------------------------
function SetupQPT2Q()

make/o/n=(36)/T TomoAxisQ1;
make/o/n=(36) TomoAngleQ1;
make/o/n=(36)/T TomoAxisQ2;
make/o/n=(36) TomoAngleQ2;

TomoAxisQ1[0,5]="ii"
TomoAxisQ1[6,11]="xi"
TomoAxisQ1[12,17]="xi"
TomoAxisQ1[18,23]="xi"
TomoAxisQ1[24,29]="yi"
TomoAxisQ1[30,36]="yi"

TomoAngleQ1[0,5]=0;
TomoAngleQ1[6,11]=pi;
TomoAngleQ1[12,17]=pi/2;
TomoAngleQ1[18,23]=-pi/2;
TomoAngleQ1[24,29]=pi/2;
TomoAngleQ1[30,35]=-pi/2;

TomoAxisQ2[0]="ii"
TomoAxisQ2[1]="ix"
TomoAxisQ2[2]="ix"
TomoAxisQ2[3]="ix"
TomoAxisQ2[4]="iy"
TomoAxisQ2[5]="iy"

TomoAngleQ2[0]=0;
TomoAngleQ2[1]=pi;
TomoAngleQ2[2]=pi/2;
TomoAngleQ2[3]=-pi/2;
TomoAngleQ2[4]=pi/2;
TomoAngleQ2[5]=-pi/2;

TomoAxisQ2[6,11]=TomoAxisQ2[p-6];
TomoAxisQ2[12,17]=TomoAxisQ2[p-12];
TomoAxisQ2[18,23]=TomoAxisQ2[p-18];
TomoAxisQ2[24,29]=TomoAxisQ2[p-24];
TomoAxisQ2[30,35]=TomoAxisQ2[p-30];

TomoAngleQ2[6,11]=TomoAngleQ2[p-6];
TomoAngleQ2[12,17]=TomoAngleQ2[p-12];
TomoAngleQ2[18,23]=TomoAngleQ2[p-18];
TomoAngleQ2[24,29]=TomoAngleQ2[p-24];
TomoAngleQ2[30,35]=TomoAngleQ2[p-30];

end


// Function: toground2Q
// -------------------------------
// sets the theoretical two-qubit wave function to |00>
function toground2Q()
WAVE/C PsiT_2Q
PsiT_2Q={1,0,0,0};
updaterhotheo2Q();
end

// Function: toGE2Q
// -----------------------
// sets the theoretical two-qubit wave function to |11>
function toGE2Q()
WAVE/C PsiT_2Q
PsiT_2Q={0,1,0,0};
updaterhotheo2Q();
end

// Function: toEG2Q
// -----------------------
// sets the theoretical two-qubit wave function to |11>
function toEG2Q()
WAVE/C PsiT_2Q
PsiT_2Q={0,0,1,0};
updaterhotheo2Q();
end

// Function: toEE2Q
// -----------------------
// sets the theoretical two-qubit wave function to |11>
function toEE2Q()
WAVE/C PsiT_2Q
PsiT_2Q={0,0,0,1};
updaterhotheo2Q();
end



// Function: PauliMovie
// -------------------------------
function PauliMovie(initid,rotL, rotR,  startdeg, enddeg,  postrot, postdeg, numframes, sec, savemovie, moviename,xval1,xval2)
string initid    //id for initial state
string rotL     //rotation axis for left qubit
string rotR     //rotation axis for right qubit
variable startdeg // initial rotation angle, in degrees
variable enddeg // final rotation angle, in degrees
string postrot
variable postdeg
variable numframes // total number of frames
variable sec    //wait between frames
variable savemovie
string  moviename
variable xval1
variable xval2

variable startrad=startdeg/180*pi;
variable rad=(enddeg-startdeg)/180*pi/(numframes-1);
variable postrad=postdeg/180*pi

if(xval1==xval2)
        xval1=startdeg
        xval2=enddeg
endif


make/o/n=(3) P1vec_theo
setscale/I x, 2, 4, P1vec_theo
make/o/n=(3) P2vec_theo
setscale/I x, 5, 7, P2vec_theo
make/o/n=(9) P12vec_theo
setscale/I x, 8, 16, P12vec_theo

make/o/n=(4,numFrames) theoCHSHxyxy=Nan
setscale/I y, xval1, xval2, theoCHSHxyxy
make/o/n=(4,numFrames) theoCHSHxzxz=Nan
setscale/I y, xval1, xval2, theoCHSHxzxz
make/o/n=(4,numFrames) theoCHSHyzyz=Nan
setscale/I y, xval1, xval2, theoCHSHyzyz

make/o/n=(4,numFrames) theoWitness=Nan
setscale/I y, xval1, xval2, theoWitness



make/o/n=(numFrames) theoCHSHxz=Nan
setscale/I x, xval1, xval2, theoCHSHxz

make/o/n=(numFrames) theoCHSHmax=Nan
setscale/I x, xval1, xval2, theoCHSHmax
make/o/n=(numFrames) theoConcurrence=Nan
setscale/I x, xval1, xval2, theoConcurrence


make/o/n=(16,numFrames) theoPauliRod=Nan
setscale/I y, xval1, xval2, theoPauliRod

WAVE PauliVec_Theo

WAVE CHSHxy
WAVE CHSHxz
WAVE CHSHyz
WAVE Witness

WAVE CurrVals_PauliTheo


setRhoTheo2Q(initid);

UnitaryEvolution2Q(rotL+"i", startrad);
UnitaryEvolution2Q("i"+rotR, startrad);

//UnitaryEvolution("iy",Pi/2);
//UnitaryEvolution("C2",0);
//UnitaryEvolution("zi",0/180*Pi);
//UnitaryEvolution("iz",0/180*Pi);

UnitaryEvolution2Q(postrot,postrad);


UpdateRhoTheo2Q(); UpdatePauliTheo2Q();
theoCHSHxyxy[][0]=CHSHxy[p];
theoCHSHxzxz[][0]=CHSHxz[p];
theoCHSHyzyz[][0]=CHSHyz[p];
theoWitness[][0]=Witness[p];
theoCHSHxz[0]=2*PauliVec_Theo[8];
theoCHSHmax[0]=CurrVals_PauliTheo[0];
theoConcurrence[0]=CurrVals_PauliTheo[1];
theoPauliRod[][0]=getPauliTheoMeas2Q(p+1,dofancy=0);
doupdate;


if(savemovie==1)
    NewMovie/O/P=figurepath as MovieName
    AddMovieFrame
endif


variable i=1;
Do

    UnitaryEvolution2Q(postrot,-postrad);

//  UnitaryEvolution2Q("zi",0/180*Pi);
//  UnitaryEvolution2Q("iz",0/180*Pi);
//  UnitaryEvolution2Q("C2",0);
//  UnitaryEvolution2Q("iy",-Pi/2);

    UnitaryEvolution2Q(rotL+"i", rad);
    UnitaryEvolution2Q("i"+rotR, rad);

//  UnitaryEvolution2Q("iy",Pi/2);
//  UnitaryEvolution2Q("C2",0);
//  UnitaryEvolution2Q("zi",0/180*Pi);
//  UnitaryEvolution2Q("iz",0/180*Pi);

    UnitaryEvolution2Q(postrot,postrad);


    UpdateRhoTheo2Q();
    theoCHSHmax[i]=UpdatePauliTheo2Q();
    theoCHSHxyxy[][i]=CHSHxy[p];
    theoCHSHxzxz[][i]=CHSHxz[p];
    theoCHSHyzyz[][i]=CHSHyz[p];
    theoWitness[][i]=Witness[p];

    theoCHSHxz[i]=2*PauliVec_Theo[8];

    theoCHSHmax[i]=CurrVals_PauliTheo[0];
    theoConcurrence[i]=CurrVals_PauliTheo[1];
    theoPauliRod[][i]=getPauliTheoMeas2Q(p+1,dofancy=0);
    doupdate;

    if(savemovie==1)
        AddMovieFrame
    endif
    wait(sec);

    i+=1
While(i<numFrames);

if(savemovie==1)
    CloseMovie
endif

end

// Function: UnitaryEvolution2Q
// ------------------------------------------
function UnitaryEvolution2Q(id, rad)
string id
variable rad

WAVE/C PsiT_2Q

make/o/c/n=(4,4) Umat=0
variable      cr=cos(rad/2);
variable      sr=sin(rad/2);
variable/c   erp=cmplx(cr,sr)
variable/c   erm=cmplx(cr,-sr)


if(stringmatch(id, "CNOT"))
    Umat[0][0]=1;
    Umat[1][1]=1;
    Umat[2][3]=1;
    Umat[3][2]=1;
endif


if(stringmatch(id, "C0"))
    Umat[0][0]=-1;
    Umat[1][1]=1;
    Umat[2][2]=1;
    Umat[3][3]=1;
endif

if(stringmatch(id, "C1"))
    Umat[0][0]=1;
    Umat[1][1]=-1;
    Umat[2][2]=1;
    Umat[3][3]=1;
endif

if(stringmatch(id, "C2"))
    Umat[0][0]=1;
    Umat[1][1]=1;
    Umat[2][2]=-1;
    Umat[3][3]=1;
endif

if(stringmatch(id, "C3"))
    Umat[0][0]=1;
    Umat[1][1]=1;
    Umat[2][2]=1;
    Umat[3][3]=-1;
endif

if(stringmatch(id, "ii"))
    Umat[0][0]=1;
    Umat[1][1]=1;
    Umat[2][2]=1;
    Umat[3][3]=1;
endif

if(stringmatch(id, "ix"))
    Umat[0][0]=cmplx(cr,0);
    Umat[0][1]=cmplx(0,-sr);
    Umat[1][0]=cmplx(0,-sr);
    Umat[1][1]=cmplx(cr,0);
    Umat[2][2]=cmplx(cr,0);
    Umat[2][3]=cmplx(0,-sr);
    Umat[3][2]=cmplx(0,-sr);
    Umat[3][3]=cmplx(cr,0);
endif
if(stringmatch(id, "iy"))
    Umat[0][0]=cmplx(cr,0);
    Umat[0][1]=cmplx(-sr,0);
    Umat[1][0]=cmplx(sr,0);
    Umat[1][1]=cmplx(cr,0);
    Umat[2][2]=cmplx(cr,0);
    Umat[2][3]=cmplx(-sr,0);
    Umat[3][2]=cmplx(sr,0);
    Umat[3][3]=cmplx(cr,0);
endif

if(stringmatch(id, "iz"))
    Umat[0][0]=erm;
    Umat[1][1]=erp;
    Umat[2][2]=erm;
    Umat[3][3]=erp;
endif

if(stringmatch(id, "xi"))
    Umat[0][0]=cmplx(cr,0);
    Umat[1][1]=cmplx(cr,0);
    Umat[2][2]=cmplx(cr,0);
    Umat[3][3]=cmplx(cr,0);

    Umat[0][2]=cmplx(0,-sr);
    Umat[1][3]=cmplx(0,-sr);
    Umat[2][0]=cmplx(0,-sr);
    Umat[3][1]=cmplx(0,-sr);
endif

if(stringmatch(id, "xx"))
    Umat[0][0]=cmplx(cr,0);
    Umat[1][1]=cmplx(cr,0);
    Umat[2][2]=cmplx(cr,0);
    Umat[3][3]=cmplx(cr,0);

    Umat[0][3]=cmplx(0,-sr);
    Umat[1][2]=cmplx(0,-sr);
    Umat[2][1]=cmplx(0,-sr);
    Umat[3][0]=cmplx(0,-sr);
endif

if(stringmatch(id, "xy"))
    Umat[0][0]=cmplx(cr,0);
    Umat[1][1]=cmplx(cr,0);
    Umat[2][2]=cmplx(cr,0);
    Umat[3][3]=cmplx(cr,0);

    Umat[0][3]=cmplx(-sr,0);
    Umat[1][2]=cmplx(sr,0);
    Umat[2][1]=cmplx(-sr,0);
    Umat[3][0]=cmplx(sr,0);
endif

if(stringmatch(id, "xz"))
    Umat[0][0]=cmplx(cr,0);
    Umat[1][1]=cmplx(cr,0);
    Umat[2][2]=cmplx(cr,0);
    Umat[3][3]=cmplx(cr,0);

    Umat[0][2]=cmplx(0,-sr);
    Umat[1][3]=cmplx(0,sr);
    Umat[2][0]=cmplx(0,-sr);
    Umat[3][1]=cmplx(0,sr);
endif

if(stringmatch(id, "yi"))
    Umat[0][0]=cmplx(cr,0);
    Umat[1][1]=cmplx(cr,0);
    Umat[2][2]=cmplx(cr,0);
    Umat[3][3]=cmplx(cr,0);

    Umat[0][2]=cmplx(-sr,0);
    Umat[1][3]=cmplx(-sr,0);
    Umat[2][0]=cmplx(sr,0);
    Umat[3][1]=cmplx(sr,0);
endif
if(stringmatch(id, "yx"))
    Umat[0][0]=cmplx(cr,0);
    Umat[1][1]=cmplx(cr,0);
    Umat[2][2]=cmplx(cr,0);
    Umat[3][3]=cmplx(cr,0);

    Umat[0][3]=cmplx(-sr,0);
    Umat[1][2]=cmplx(-sr,0);
    Umat[2][1]=cmplx(sr,0);
    Umat[3][0]=cmplx(sr,0);
endif

if(stringmatch(id, "yy"))
    Umat[0][0]=cmplx(cr,0);
    Umat[1][1]=cmplx(cr,0);
    Umat[2][2]=cmplx(cr,0);
    Umat[3][3]=cmplx(cr,0);

    Umat[0][3]=cmplx(0,sr);
    Umat[1][2]=cmplx(0,-sr);
    Umat[2][1]=cmplx(0,-sr);
    Umat[3][0]=cmplx(0,sr);
endif
if(stringmatch(id, "yz"))
    Umat[0][0]=cmplx(cr,0);
    Umat[1][1]=cmplx(cr,0);
    Umat[2][2]=cmplx(cr,0);
    Umat[3][3]=cmplx(cr,0);

    Umat[0][2]=cmplx(-sr,0);
    Umat[1][3]=cmplx(sr,0);
    Umat[2][0]=cmplx(sr,0);
    Umat[3][1]=cmplx(-sr,0);
endif

if(stringmatch(id, "zi"))
    Umat[0][0]=erm;
    Umat[1][1]=erm;
    Umat[2][2]=erp;
    Umat[3][3]=erp;
endif
if(stringmatch(id, "zx"))
    Umat[0][0]=cmplx(cr,0);
    Umat[0][1]=cmplx(0,-sr);
    Umat[1][0]=cmplx(0,-sr);
    Umat[1][1]=cmplx(cr,0);
    Umat[2][2]=cmplx(cr,0);
    Umat[2][3]=cmplx(0,sr);
    Umat[3][2]=cmplx(0,sr);
    Umat[3][3]=cmplx(cr,0);
endif
if(stringmatch(id, "zy"))
    Umat[0][0]=cmplx(cr,0);
    Umat[0][1]=cmplx(-sr,0);
    Umat[1][0]=cmplx(sr,0);
    Umat[1][1]=cmplx(cr,0);
    Umat[2][2]=cmplx(cr,0);
    Umat[2][3]=cmplx(sr,0);
    Umat[3][2]=cmplx(-sr,0);
    Umat[3][3]=cmplx(cr,0);
endif
if(stringmatch(id, "zz"))
    Umat[0][0]=erm;
    Umat[1][1]=erp;
    Umat[2][2]=erp;
    Umat[3][3]=erm;
endif

WAVE/C PsiT_2Q
WAVE/C RhoT_2Q
MatrixOp/C/O temp= Umat x PsiT_2Q
PsiT_2Q=temp
MatrixOp/C/O temp2= Umat x RhoT_2Q x Umat^h;
RhoT_2Q=temp2
end

// Function: updatePauliTheo2Q()
// -----------------------------------------
function updatePauliTheo2Q([Pvectheoname])
string Pvectheoname

if(paramisdefault(Pvectheoname))
Pvectheoname="PauliSetT_2Q"
endif

make/o/n=(16) $Pvectheoname=NaN
setscale/i x, 1, 16, $Pvectheoname
WAVE Pvectheo=$Pvectheoname

make/o/n=(3) P1vecT=NaN
setscale/I x, 1, 3, P1vecT
make/o/n=(3) P2vecT=NaN
setscale/I x, 4, 6, P2vecT
make/o/n=(9) P12vecT=NaN
setscale/I x, 7, 15, P12vecT

Pvectheo=getPauliTheoMeas2Q(x,dofancy=1);

P1vecT[]=Pvectheo[1+p];
P2vecT[]=Pvectheo[4+p];
P12vecT[]=Pvectheo[7+p];

//variable CHSHxy1= -Pvectheo[13]+Pvectheo[7]+Pvectheo[9]+Pvectheo[14];
//variable CHSHxy2= Pvectheo[13]-Pvectheo[7]+Pvectheo[9]+Pvectheo[14];
//variable CHSHxy3= Pvectheo[13]+Pvectheo[7]-Pvectheo[9]+Pvectheo[14];
//variable CHSHxy4= Pvectheo[13]+Pvectheo[7]+Pvectheo[9]-Pvectheo[14];
//
//variable CHSHxz1= -Pvectheo[13]+Pvectheo[8]+Pvectheo[11]+Pvectheo[15];
//variable CHSHxz2= Pvectheo[13]-Pvectheo[8]+Pvectheo[11]+Pvectheo[15];
//variable CHSHxz3= Pvectheo[13]+Pvectheo[8]-Pvectheo[11]+Pvectheo[15];
//variable CHSHxz4= Pvectheo[13]+Pvectheo[8]+Pvectheo[11]-Pvectheo[15];
//
//variable CHSHyx1= -Pvectheo[14]+Pvectheo[9]+Pvectheo[7]+Pvectheo[13];
//variable CHSHyx2= Pvectheo[14]-Pvectheo[9]+Pvectheo[7]+Pvectheo[13];
//variable CHSHyx3= Pvectheo[14]+Pvectheo[9]-Pvectheo[7]+Pvectheo[13];
//variable CHSHyx4= Pvectheo[14]+Pvectheo[9]+Pvectheo[7]-Pvectheo[13];
//
//variable CHSHyz1= -Pvectheo[14]+Pvectheo[10]+Pvectheo[12]+Pvectheo[15];
//variable CHSHyz2= Pvectheo[14]-Pvectheo[10]+Pvectheo[12]+Pvectheo[15];
//variable CHSHyz3= Pvectheo[14]+Pvectheo[10]-Pvectheo[12]+Pvectheo[15];
//variable CHSHyz4= Pvectheo[14]+Pvectheo[10]+Pvectheo[12]-Pvectheo[15];
//
//variable CHSHzx1= -Pvectheo[15]+Pvectheo[11]+Pvectheo[8]+Pvectheo[13];
//variable CHSHzx2= Pvectheo[15]-Pvectheo[11]+Pvectheo[8]+Pvectheo[13];
//variable CHSHzx3= Pvectheo[15]+Pvectheo[11]-Pvectheo[8]+Pvectheo[13];
//variable CHSHzx4= Pvectheo[15]+Pvectheo[11]+Pvectheo[8]-Pvectheo[13];
//
//
//variable CHSHzy1= -Pvectheo[15]+Pvectheo[12]+Pvectheo[10]+Pvectheo[14];
//variable CHSHzy2= Pvectheo[15]-Pvectheo[12]+Pvectheo[10]+Pvectheo[14];
//variable CHSHzy3= Pvectheo[15]+Pvectheo[12]-Pvectheo[10]+Pvectheo[14];
//variable CHSHzy4= Pvectheo[15]+Pvectheo[12]+Pvectheo[10]-Pvectheo[14];
//
//
//Pvectheo=getPauliTheoMeas2Q(x,dofancy=0);
//variable Witness0=    -2*1/4*(Pvectheo[0]-Pvectheo[5]+Pvectheo[10]-Pvectheo[15])
//variable Witness1=    -2*1/4*(Pvectheo[0]+Pvectheo[5]-Pvectheo[10]-Pvectheo[15])
//variable Witness2=    -2*1/4*(Pvectheo[0]-Pvectheo[5]-Pvectheo[10]+Pvectheo[15])
//variable Witness3=    -2*1/4*(Pvectheo[0]+Pvectheo[5]+Pvectheo[10]+Pvectheo[15])
//Pvectheo=getPauliTheoMeas2Q(x,dofancy=1);
//
//WAVE PsiT_2Q
//WAVE CurrVals_PauliT_2Q
//
//WAVE CHSHxy
//WAVE CHSHxz
//WAVE CHSHyx
//WAVE CHSHyz
//WAVE CHSHzx
//WAVE CHSHzy
//WAVE Witness
//
//
//CHSHxy[0]=CHSHxy1
//CHSHxy[1]=CHSHxy2
//CHSHxy[2]=CHSHxy3
//CHSHxy[3]=CHSHxy4
//
//CHSHxz[0]=CHSHxz1
//CHSHxz[1]=CHSHxz2
//CHSHxz[2]=CHSHxz3
//CHSHxz[3]=CHSHxz4
//
//CHSHyx[0]=CHSHyx1
//CHSHyx[1]=CHSHyx2
//CHSHyx[2]=CHSHyx3
//CHSHyx[3]=CHSHyx4
//
//
//CHSHyz[0]=CHSHyz1
//CHSHyz[1]=CHSHyz2
//CHSHyz[2]=CHSHyz3
//CHSHyz[3]=CHSHyz4
//
//
//CHSHzx[0]=CHSHzx1
//CHSHzx[1]=CHSHzx2
//CHSHzx[2]=CHSHzx3
//CHSHzx[3]=CHSHzx4
//
//CHSHzy[0]=CHSHzy1
//CHSHzy[1]=CHSHzy2
//CHSHzy[2]=CHSHzy3
//CHSHzy[3]=CHSHzy4
//
//Witness[0]=Witness0
//Witness[1]=Witness1
//Witness[2]=Witness2
//Witness[3]=Witness3
//
//WAVE MichelQmat
//
//MichelQMat[0][0,2]={{Pvectheo[13]},{Pvectheo[7]},{Pvectheo[8]}}
//MichelQMat[1][0,2]={{Pvectheo[9]},{Pvectheo[14]},{Pvectheo[10]}}
//MichelQMat[2][0,2]={{Pvectheo[11]},{Pvectheo[12]},{Pvectheo[15]}}
//MatrixOp/O tempmat=(MichelQmat)^t x MichelQmat
//MatrixEigenV/S=1 tempmat
//WAVE W_eigenvalues
//make/o/n=(3) tempvec
//tempvec=Real(W_eigenvalues)
//sort/R  tempvec tempvec
//
//variable maxCHSH=2*sqrt(tempvec[0]+tempvec[1])
//variable concurrence=2*sqrt(magsqr(PsiT_2Q[0]*PsiT_2Q[3]-PsiT_2Q[1]*PsiT_2Q[2]))
//
//CurrVals_PauliT_2Q[0]=maxCHSH
//CurrVals_PauliT_2Q[1]=concurrence
//
//return maxCHSH;

end


// Function: getPauliTheoMeas2Q(Paulinum)
// --------------------------------------------------------
function getPauliTheoMeas2Q(Paulinum, [dofancy, Rhoname])
variable Paulinum
variable dofancy
string Rhoname

if(paramisdefault(dofancy))
    dofancy=0;
endif;
if(paramisdefault(Rhoname))
    Rhoname="RhoT_2Q"
endif

WAVE/C/D RhoT_2Q=$Rhoname

if(dofancy==1)
    if(Paulinum==1)
        WAVE thisSig=Sig_ii
    elseif(Paulinum==2)
        WAVE thisSig=Sig_xi
    elseif(Paulinum==3)
        WAVE thisSig=Sig_yi
    elseif(Paulinum==4)
        WAVE thisSig=Sig_zi
    elseif(Paulinum==5)
        WAVE thisSig=Sig_ix
    elseif(Paulinum==6)
        WAVE thisSig=Sig_iy
    elseif(Paulinum==7)
        WAVE thisSig=Sig_iz
    elseif(Paulinum==8)
        WAVE thisSig=Sig_xx
    elseif(Paulinum==9)
        WAVE thisSig=Sig_xy
    elseif(Paulinum==10)
        WAVE thisSig=Sig_xz
    elseif(Paulinum==11)
        WAVE thisSig=Sig_yx
    elseif(Paulinum==12)
        WAVE thisSig=Sig_yy
    elseif(Paulinum==13)
        WAVE thisSig=Sig_yz
    elseif(Paulinum==14)
        WAVE thisSig=Sig_zx
    elseif(Paulinum==15)
        WAVE thisSig=Sig_zy
    elseif(Paulinum==16)
        WAVE thisSig=Sig_zz
    endif
endif

if(dofancy==0)
    if(Paulinum==1)
        WAVE thisSig=Sig_ii
    elseif(Paulinum==2)
        WAVE thisSig=Sig_xi
    elseif(Paulinum==3)
        WAVE thisSig=Sig_yi
    elseif(Paulinum==4)
        WAVE thisSig=Sig_zi
    elseif(Paulinum==5)
        WAVE thisSig=Sig_ix
    elseif(Paulinum==6)
        WAVE thisSig=Sig_xx
    elseif(Paulinum==7)
        WAVE thisSig=Sig_yx
    elseif(Paulinum==8)
        WAVE thisSig=Sig_zx
    elseif(Paulinum==9)
        WAVE thisSig=Sig_iy
    elseif(Paulinum==10)
        WAVE thisSig=Sig_xy
    elseif(Paulinum==11)
        WAVE thisSig=Sig_yy
    elseif(Paulinum==12)
        WAVE thisSig=Sig_zy
    elseif(Paulinum==13)
        WAVE thisSig=Sig_iz
    elseif(Paulinum==14)
        WAVE thisSig=Sig_xz
    elseif(Paulinum==15)
        WAVE thisSig=Sig_yz
    elseif(Paulinum==16)
        WAVE thisSig=Sig_zz
    endif
endif

MatrixOp/O/C   Meas =Trace(thisSig x RhoT_2Q)

return Real(Meas[0][0])
end


//Function: updateRhoTheo2Q()
// -------------------------------------------
// updates RhoT_2Q using current value of PsiT_2Q
function updateRhoTheo2Q()

WAVE/C PsiT_2Q
WAVE/C RhoT_2Q
MatrixOp/O/C  RhoT_2Q =PsiT_2Q x PsiT_2Q^h
end


//Function: setRhoTheo2Q(StateId)
// ------------------------------------------------
// sets RhoT_2Q according to the specifications of StateID
function setRhoTheo2Q(StateId)
string StateId
WAVE/C RhoT_2Q
WAVE/C PsiT_2Q


if(stringmatch(StateId,"00"))
    make/O/C/N=(4,4) RhoT_2Q=NaN
    make/O/C/N=(4) PsiT_2Q=NaN
    PsiT_2Q[0]=1;
    PsiT_2Q[1]=0
    PsiT_2Q[2]=0
    PsiT_2Q[3]=0;
endif

if(stringmatch(StateId,"01"))
    make/O/C/N=(4,4) RhoT_2Q=NaN
    make/O/C/N=(4) PsiT_2Q=NaN
    PsiT_2Q[0]=0;
    PsiT_2Q[1]=1;
    PsiT_2Q[2]=0;
    PsiT_2Q[3]=0;
endif

if(stringmatch(StateId,"10"))
    make/O/C/N=(4,4) RhoT_2Q=NaN
    make/O/C/N=(4) PsiT_2Q=NaN
    PsiT_2Q[0]=0;
    PsiT_2Q[1]=0;
    PsiT_2Q[2]=1;
    PsiT_2Q[3]=0;
endif

if(stringmatch(StateId,"11"))
    make/O/C/N=(4,4) RhoT_2Q=NaN
    make/O/C/N=(4) PsiT_2Q=NaN
    PsiT_2Q[0]=0;
    PsiT_2Q[1]=0;
    PsiT_2Q[2]=0;
    PsiT_2Q[3]=1;
endif


if(stringmatch(StateId,"XX"))
    make/O/C/N=(4,4) RhoT_2Q=NaN
    make/O/C/N=(4) PsiT_2Q=NaN;
    PsiT_2Q[0]=cmplx(1/2,0);
    PsiT_2Q[1]=cmplx(0,-1/2);
    PsiT_2Q[2]=cmplx(0,-1/2);
    PsiT_2Q[3]=cmplx(-1/2,0);
endif

if(stringmatch(StateId,"XY"))
    make/O/C/N=(4,4) RhoT_2Q=NaN
    make/O/C/N=(4) PsiT_2Q=NaN;
    PsiT_2Q[0]=cmplx(1/2,0);
    PsiT_2Q[1]=cmplx(1/2,0);
    PsiT_2Q[2]=cmplx(0,-1/2);
    PsiT_2Q[3]=cmplx(0,-1/2);
endif

if(stringmatch(StateId,"YX"))
    make/O/C/N=(4,4) RhoT_2Q=NaN
    make/O/C/N=(4) PsiT_2Q=NaN;
    PsiT_2Q[0]=cmplx(1/2,0);
    PsiT_2Q[1]=cmplx(0,-1/2);
    PsiT_2Q[2]=cmplx(1/2,0);
    PsiT_2Q[3]=cmplx(0,-1/2);
endif

if(stringmatch(StateId,"YY"))
    make/O/C/N=(4,4) RhoT_2Q=NaN
    make/O/C/N=(4) PsiT_2Q=1/2
endif

if(stringmatch(StateId,"B0"))
    make/O/C/N=(4,4) RhoT_2Q=NaN
    make/O/C/N=(4) PsiT_2Q=NaN
    PsiT_2Q[0]=1/sqrt(2);
    PsiT_2Q[1]=0
    PsiT_2Q[2]=0
    PsiT_2Q[3]=1/sqrt(2);
endif

if(stringmatch(StateId,"B1"))
    make/O/C/N=(4,4) RhoT_2Q=NaN
    make/O/C/N=(4) PsiT_2Q=NaN
    PsiT_2Q[0]=1/sqrt(2);
    PsiT_2Q[1]=0
    PsiT_2Q[2]=0
    PsiT_2Q[3]=-1/sqrt(2);
endif

if(stringmatch(StateId,"B2"))
    make/O/C/N=(4,4) RhoT_2Q=NaN
    make/O/C/N=(4) PsiT_2Q=NaN
    PsiT_2Q[0]=0
    PsiT_2Q[1]=1/sqrt(2);
    PsiT_2Q[2]=1/sqrt(2);
    PsiT_2Q[3]=0
endif

if(stringmatch(StateId,"B3"))
    make/O/C/N=(4,4) RhoT_2Q=NaN
    make/O/C/N=(4) PsiT_2Q=NaN
    PsiT_2Q[0]=0
    PsiT_2Q[1]=1/sqrt(2);
    PsiT_2Q[2]=-1/sqrt(2);
    PsiT_2Q[3]=0
endif

MatrixOp/O/C  RhoT_2Q =PsiT_2Q x PsiT_2Q^h
updateRhoTheo2Q()

end


// Function: MakePauliOperators2Q()
// ----------------------------------------------------
function makePaulioperators2Q()

// projection operators
make/o/n=(4,4)/C Proj_00=0;
Proj_00[0][0]=cmplx(1,0);
make/o/n=(4,4)/C Proj_01=0;
Proj_01[1][1]=cmplx(1,0);
make/o/n=(4,4)/C Proj_10=0;
Proj_10[2][2]=cmplx(1,0);
make/o/n=(4,4)/C Proj_11=0;
Proj_11[3][3]=cmplx(1,0);

// creation operators
make/o/n=(4,4)/C Sig_pi=0
Sig_pi[2][0]=cmplx(1,0);
Sig_pi[3][1]=cmplx(1,0);
make/o/n=(4,4)/C Sig_ip=0
Sig_ip[1][0]=cmplx(1,0);
Sig_ip[3][2]=cmplx(1,0);


// annihilation operators
make/o/n=(4,4)/C Sig_mi=0
Sig_mi[0][2]=cmplx(1,0);
Sig_mi[1][3]=cmplx(1,0);
make/o/n=(4,4)/C Sig_im=0
Sig_im[0][1]=cmplx(1,0);
Sig_im[2][3]=cmplx(1,0);



make/o/n=(4,4)/C Sig_ii=0
Sig_ii[0][0]=cmplx(1,0);
Sig_ii[1][1]=cmplx(1,0);
Sig_ii[2][2]=cmplx(1,0);
Sig_ii[3][3]=cmplx(1,0);
make/o/n=(4,4)/C Sig_xi=0
Sig_xi[0][2]=cmplx(1,0);
Sig_xi[1][3]=cmplx(1,0);
Sig_xi[2][0]=cmplx(1,0);
Sig_xi[3][1]=cmplx(1,0);
make/o/n=(4,4)/C Sig_yi=0
Sig_yi[0][2]=cmplx(0,-1);
Sig_yi[1][3]=cmplx(0,-1);
Sig_yi[2][0]=cmplx(0,1);
Sig_yi[3][1]=cmplx(0,1);
make/o/n=(4,4)/C Sig_zi=0
Sig_zi[0][0]=cmplx(1,0);
Sig_zi[1][1]=cmplx(1,0);
Sig_zi[2][2]=cmplx(-1,0);
Sig_zi[3][3]=cmplx(-1,0);


make/o/n=(4,4)/C Sig_ix=0
Sig_ix[0][1]=cmplx(1,0);
Sig_ix[1][0]=cmplx(1,0);
Sig_ix[2][3]=cmplx(1,0);
Sig_ix[3][2]=cmplx(1,0);
make/o/n=(4,4)/C Sig_xx=0
Sig_xx[0][3]=cmplx(1,0);
Sig_xx[1][2]=cmplx(1,0);
Sig_xx[2][1]=cmplx(1,0);
Sig_xx[3][0]=cmplx(1,0);
make/o/n=(4,4)/C Sig_yx=0
Sig_yx[0][3]=cmplx(0,-1);
Sig_yx[1][2]=cmplx(0,-1);
Sig_yx[2][1]=cmplx(0,1);
Sig_yx[3][0]=cmplx(0,1);
make/o/n=(4,4)/C Sig_zx=0
Sig_zx[0][1]=cmplx(1,0);
Sig_zx[1][0]=cmplx(1,0);
Sig_zx[2][3]=cmplx(-1,0);
Sig_zx[3][2]=cmplx(-1,0);

make/o/n=(4,4)/C Sig_iy=0
Sig_iy[0][1]=cmplx(0,-1);
Sig_iy[1][0]=cmplx(0,1);
Sig_iy[2][3]=cmplx(0,-1);
Sig_iy[3][2]=cmplx(0,1);
make/o/n=(4,4)/C Sig_xy=0
Sig_xy[0][3]=cmplx(0,-1);
Sig_xy[1][2]=cmplx(0,1);
Sig_xy[2][1]=cmplx(0,-1);
Sig_xy[3][0]=cmplx(0,1);
make/o/n=(4,4)/C Sig_yy=0
Sig_yy[0][3]=cmplx(-1,0);
Sig_yy[1][2]=cmplx(1,0);
Sig_yy[2][1]=cmplx(1,0);
Sig_yy[3][0]=cmplx(-1,0);
make/o/n=(4,4)/C Sig_zy=0
Sig_zy[0][1]=cmplx(0,-1);
Sig_zy[1][0]=cmplx(0,1);
Sig_zy[2][3]=cmplx(0,1);
Sig_zy[3][2]=cmplx(0,-1);

make/o/n=(4,4)/C Sig_iz=0
Sig_iz[0][0]=cmplx(1,0);
Sig_iz[1][1]=cmplx(-1,0);
Sig_iz[2][2]=cmplx(1,0);
Sig_iz[3][3]=cmplx(-1,0);
make/o/n=(4,4)/C Sig_xz=0
Sig_xz[0][2]=cmplx(1,0);
Sig_xz[1][3]=cmplx(-1,0);
Sig_xz[2][0]=cmplx(1,0);
Sig_xz[3][1]=cmplx(-1,0);
make/o/n=(4,4)/C Sig_yz=0
Sig_yz[0][2]=cmplx(0,-1);
Sig_yz[1][3]=cmplx(0,1);
Sig_yz[2][0]=cmplx(0,1);
Sig_yz[3][1]=cmplx(0,-1);
make/o/n=(4,4)/C Sig_zz=0
Sig_zz[0][0]=cmplx(1,0);
Sig_zz[1][1]=cmplx(-1,0);
Sig_zz[2][2]=cmplx(-1,0);
Sig_zz[3][3]=cmplx(1,0);
end

Window Graph_PauliSetTheo() : Graph
    PauseUpdate; Silent 1       // building window...
    Display /W=(570.75,44.75,1064.25,258.5) P1vecT,P2vecT,P12vecT
    ModifyGraph userticks(bottom)={TicksPauliRod,TickNamesPauliRod}
    ModifyGraph margin(top)=20,margin(right)=10
    ModifyGraph mode=5
    ModifyGraph rgb(P2vecT)=(0,12800,52224),rgb(P12vecT)=(36864,14592,58880)
    ModifyGraph hbFill=12
    ModifyGraph offset(P1vecT)={-0.5,0},offset(P2vecT)={-0.5,0},offset(P12vecT)={-0.5,0}
    ModifyGraph grid(left)=1
    ModifyGraph tick=2
    ModifyGraph zero(left)=2
    ModifyGraph mirror=1
    ModifyGraph fSize=12
    ModifyGraph btLen=2
    Label left "Mean Value"
    Label bottom "Pauli measurement"
    SetAxis left -1.1,1.1
    SetAxis bottom 1,17
    Legend/C/N=text0/J/F=0/B=1/A=MC/X=16.37/Y=46.88/E=2 "\\s(P1vecT) P1vecT   \\s(P2vecT) P2vecT    \\s(P12vecT) P12vecT"
EndMacro

Window Graph_AllCHSH() : Graph
    PauseUpdate; Silent 1       // building window...
    Display /W=(489.75,41,947.25,279.5) TwoLine,mTwoLine,TwoRtTwoLine,mTwoRtTwoLine,CHSHxy
    AppendToGraph CHSHxz,CHSHyx,CHSHyz,CHSHzx,CHSHzy
    ModifyGraph margin(top)=30,margin(right)=10
    ModifyGraph mode(CHSHxy)=5,mode(CHSHxz)=5,mode(CHSHyx)=5,mode(CHSHyz)=5,mode(CHSHzx)=5
    ModifyGraph mode(CHSHzy)=5
    ModifyGraph lSize(TwoLine)=2,lSize(mTwoLine)=2,lSize(TwoRtTwoLine)=2,lSize(mTwoRtTwoLine)=2
    ModifyGraph lStyle(TwoLine)=3,lStyle(mTwoLine)=3
    ModifyGraph rgb(TwoLine)=(8704,8704,8704),rgb(mTwoLine)=(8704,8704,8704),rgb(TwoRtTwoLine)=(8704,8704,8704)
    ModifyGraph rgb(mTwoRtTwoLine)=(8704,8704,8704),rgb(CHSHxz)=(65280,43520,0),rgb(CHSHyx)=(0,52224,0)
    ModifyGraph rgb(CHSHyz)=(0,39168,39168),rgb(CHSHzx)=(0,12800,52224),rgb(CHSHzy)=(26368,0,52224)
    ModifyGraph hbFill(CHSHxy)=6,hbFill(CHSHxz)=6,hbFill(CHSHyx)=6,hbFill(CHSHyz)=6
    ModifyGraph hbFill(CHSHzx)=6,hbFill(CHSHzy)=6
    ModifyGraph offset(CHSHxz)={4,0},offset(CHSHyx)={8,0},offset(CHSHyz)={12,0},offset(CHSHzx)={16,0}
    ModifyGraph offset(CHSHzy)={20,0}
    ModifyGraph tick=2
    ModifyGraph zero(left)=2
    ModifyGraph mirror=1
    ModifyGraph fSize=10
    ModifyGraph btLen=2
    Label left "CHSH mean value"
    Label bottom "CHSH operator"
    SetAxis left -3,3
    SetAxis bottom 0,24
    Legend/N=text0/J/F=0/B=1/A=MC/X=2.95/Y=44.37/E=2 "\\s(CHSHxy) CHSHxy \\s(CHSHxz) CHSHxz \\s(CHSHyx) CHSHyx \\s(CHSHyz) CHSHyz \\s(CHSHzx) CHSHzx \\s(CHSHzy) CHSHzy"
EndMacro

Window Graph_theoCHSHxyxy() : Graph
    PauseUpdate; Silent 1       // building window...
    Display /W=(198,41,440.25,248) theoCHSHxyxy[0][*],theoCHSHxyxy[1][*],theoCHSHxyxy[2][*]
    AppendToGraph theoCHSHxyxy[3][*],theoCHSHmax
    ModifyGraph userticks(left)={TicksCHSH,TickNamesCHSH}
    ModifyGraph margin(left)=35,margin(top)=30,margin(right)=10
    ModifyGraph mode=4
    ModifyGraph marker=19
    ModifyGraph rgb(theoCHSHxyxy#1)=(65280,32512,16384),rgb(theoCHSHxyxy#2)=(0,52224,0)
    ModifyGraph rgb(theoCHSHxyxy#3)=(0,26112,39168),rgb(theoCHSHmax)=(0,0,0)
    ModifyGraph msize=1.5
    ModifyGraph grid(left)=1
    ModifyGraph tick=2
    ModifyGraph zero(left)=2
    ModifyGraph mirror=1
    ModifyGraph fSize=10
    ModifyGraph lblMargin(left)=5
    ModifyGraph gridRGB(left)=(4352,4352,4352)
    ModifyGraph btLen=2
    Label left "CHSH mean value"
    Label bottom "Rotation angle (degrees) "
    SetAxis left -3,3
    Legend/C/N=text0/J/F=0/B=1/A=MC/X=3.66/Y=44.24/E=2 "\\s(theoCHSHxyxy) \\s(theoCHSHxyxy#1) \\s(theoCHSHxyxy#2) \\s(theoCHSHxyxy#3) \\s(theoCHSHmax) theoCHSHxyxy"
EndMacro


Window Graph_theoCHSHxzxz() : Graph
    PauseUpdate; Silent 1       // building window...
    Display /W=(452.25,41,696,248) theoCHSHxzxz[0][*],theoCHSHxzxz[1][*],theoCHSHxzxz[2][*]
    AppendToGraph theoCHSHxzxz[3][*],theoCHSHmax
    ModifyGraph userticks(left)={TicksCHSH,TickNamesCHSH}
    ModifyGraph margin(left)=35,margin(top)=30,margin(right)=10
    ModifyGraph mode=4
    ModifyGraph marker=19
    ModifyGraph rgb(theoCHSHxzxz#1)=(65280,32512,16384),rgb(theoCHSHxzxz#2)=(0,52224,0)
    ModifyGraph rgb(theoCHSHxzxz#3)=(0,26112,39168),rgb(theoCHSHmax)=(0,0,0)
    ModifyGraph msize=1.5
    ModifyGraph offset(theoCHSHxzxz)={0,0.01}
    ModifyGraph grid(left)=1
    ModifyGraph tick=2
    ModifyGraph zero(left)=2
    ModifyGraph mirror=1
    ModifyGraph noLabel(left)=2
    ModifyGraph fSize=10
    ModifyGraph lblMargin(left)=5
    ModifyGraph gridRGB(left)=(4352,4352,4352)
    ModifyGraph btLen=2
    Label left "CHSH mean value"
    Label bottom "Rotation angle (degrees)  "
    SetAxis left -3,3
    Cursor/P A theoCHSHxzxz#1 16
    ShowInfo
    Legend/C/N=text0/J/F=0/B=1/A=MC/X=2.44/Y=44.60/E=2 "\\s(theoCHSHxzxz) \\s(theoCHSHxzxz#1) \\s(theoCHSHxzxz#2) \\s(theoCHSHxzxz#3) \\s(theoCHSHmax) theoCHSHxzxz"
EndMacro

Window Graph_theoCHSHyzyz() : Graph
    PauseUpdate; Silent 1       // building window...
    Display /W=(708,41,953.25,248) theoCHSHyzyz[0][*],theoCHSHyzyz[1][*],theoCHSHyzyz[2][*]
    AppendToGraph theoCHSHyzyz[3][*],theoCHSHmax
    ModifyGraph userticks(left)={TicksCHSH,TickNamesCHSH}
    ModifyGraph margin(left)=35,margin(top)=30,margin(right)=10
    ModifyGraph mode=4
    ModifyGraph marker=19
    ModifyGraph rgb(theoCHSHyzyz#1)=(65280,32512,16384),rgb(theoCHSHyzyz#2)=(0,52224,0)
    ModifyGraph rgb(theoCHSHyzyz#3)=(0,26112,39168),rgb(theoCHSHmax)=(0,0,0)
    ModifyGraph msize=1.5
    ModifyGraph grid(left)=1
    ModifyGraph tick=2
    ModifyGraph zero(left)=2
    ModifyGraph mirror=1
    ModifyGraph noLabel(left)=2
    ModifyGraph fSize=10
    ModifyGraph lblMargin(left)=5
    ModifyGraph gridRGB(left)=(4352,4352,4352)
    ModifyGraph btLen=2
    Label left "CHSH mean value"
    Label bottom "Rotation angle (degrees) "
    SetAxis left -3,3
    Legend/C/N=text0/J/F=0/B=1/A=MC/X=3.17/Y=43.53/E=2 "\\s(theoCHSHyzyz) \\s(theoCHSHyzyz#1) \\s(theoCHSHyzyz#2) \\s(theoCHSHyzyz#3) \\s(theoCHSHmax) theoCHSHyzyz"
EndMacro


Window Graph_theoMetrics() : Graph
    PauseUpdate; Silent 1       // building window...
    Display /W=(197.25,275,440.25,483.5) theoConcurrence
    ModifyGraph margin(left)=35,margin(top)=30,margin(right)=10
    ModifyGraph mode=4
    ModifyGraph marker=19
    ModifyGraph rgb=(65280,0,0)
    ModifyGraph msize=1.5
    ModifyGraph grid(left)=1
    ModifyGraph tick=2
    ModifyGraph zero(left)=2
    ModifyGraph mirror=1
    ModifyGraph fSize=10
    ModifyGraph btLen=2
    Label left "Concurrence"
    Label bottom "Rotation angle (degrees)"
    SetAxis left -0.05,1.05
    Legend/C/N=text0/J/F=0/B=1/A=MC/X=3.63/Y=44.96/E=2 "\\Z08\\s(theoConcurrence) theoConcurrence"
EndMacro


Window Layout_PauliTheo() : Layout
    PauseUpdate; Silent 1       // building window...
    Layout/C=1/W=(59.25,41.75,936,671) graph_theoMetrics(269.25,257.25,512.25,465.75)/O=1/F=0/T
    Append Graph_theoCHSHyz(480.75,39.75,726,246.75)/O=1/F=0/T,Graph_theoCHSHxz(268.5,39.75,512.25,246.75)/O=1/F=0/T
    Append Graph_theoCHSHxy(59.25,39.75,301.5,246.75)/O=1/F=0/T
    ModifyLayout mag=1, units=1
EndMacro



function appendTheoPauliRod()
appendtograph theoPauliRod[0][];
appendtograph theoPauliRod[1][];
appendtograph theoPauliRod[2][];
appendtograph theoPauliRod[3][];
appendtograph theoPauliRod[4][];
appendtograph theoPauliRod[5][];
appendtograph theoPauliRod[6][];
appendtograph theoPauliRod[7][];
appendtograph theoPauliRod[8][];
appendtograph theoPauliRod[9][];
appendtograph theoPauliRod[10][];
appendtograph theoPauliRod[11][];
appendtograph theoPauliRod[12][];
appendtograph theoPauliRod[13][];
appendtograph theoPauliRod[14][];
appendtograph theoPauliRod[15][];
autocolor(sets=2,inter=0,doffset=2)
end



// function UncondParity2Q
//-------------------------------------
function UncondParity2Q()

WAVE/C/D RhoT_2Q
WAVE/C/D Proj_00
WAVE/C/D Proj_01
WAVE/C/D Proj_10
WAVE/C/D Proj_11
MatrixOp/O/C   Proj_odd= Proj_01+Proj_10;
MatrixOp/O/C   Proj_even= Proj_00+Proj_11;

MatrixOp/O/C   RhoT_2Q= Proj_odd x RhoT_2Q x  Proj_odd + Proj_even x RhoT_2Q x  Proj_even
end



// function Dissipator2Q
//--------------------------------
function Dissipator2Q(id, [prefactor])
string id
variable/D prefactor

if(paramisdefault(prefactor))
    prefactor=1;
endif

WAVE/C/D RhoT_2Q

if(stringmatch(id,"zi")||stringmatch(id,"ZI"))
    WAVE/C/D A=Sig_zi;
elseif(stringmatch(id,"iz")||stringmatch(id,"IZ"))
    WAVE/C/D A=Sig_iz;
elseif(stringmatch(id,"pi")||stringmatch(id,"PI"))
    WAVE/C/D A=Sig_pi;
elseif(stringmatch(id,"mi")||stringmatch(id,"MI"))
    WAVE/C/D A=Sig_mi;
elseif(stringmatch(id,"im")||stringmatch(id,"IM"))
    WAVE/C/D A=Sig_im;
else
    WAVE/C/D A=Sig_ii;
endif

MatrixOp/O/C   Adag= A^h
MatrixOp/O/C   dRhoT_2Q= A x RhoT_2Q x Adag - Adag x A x RhoT_2Q /2  - RhoT_2Q x Adag x A  /2
RhoT_2Q+=cmplx(prefactor,0) * dRhoT_2Q
end


// function Measurement2Q
//--------------------------------
function Measurement2Q(id, [prefactor])
string id
variable/D prefactor

if(paramisdefault(prefactor))
    prefactor=1;
endif

WAVE/C/D RhoT_2Q

if(stringmatch(Id, "homodyne"))
    WAVE/C/D A=ProjAlpha
elseif(stringmatch(id,"zi")||stringmatch(id,"ZI"))
    WAVE/C/D A=Sig_zi;
elseif(stringmatch(id,"iz")||stringmatch(id,"IZ"))
    WAVE/C/D A=Sig_iz;
elseif(stringmatch(id,"pi")||stringmatch(id,"PI"))
    WAVE/C/D A=Sig_pi;
elseif(stringmatch(id,"mi")||stringmatch(id,"MI"))
    WAVE/C/D A=Sig_mi;
elseif(stringmatch(id,"im")||stringmatch(id,"IM"))
    WAVE/C/D A=Sig_im;
else
    WAVE/C/D A=Sig_ii;
endif

MatrixOp/O/C   foo=Trace((A+A^h) x RhoT_2Q)
MatrixOp/O/C   dRhoT_2Q= A x RhoT_2Q + RhoT_2Q x A^h
RhoT_2Q+=cmplx(prefactor,0) * (dRhoT_2Q  - foo[0][0]* RhoT_2Q)

end

//Function: setstate2Q
// ------------------------------
function setstate2Q(StateId,val, [theta, phi])
string StateId
variable val        // degrees
variable theta      // radians
variable phi        // radians

if(paramisdefault(theta))
    theta=pi/2
endif

if(paramisdefault(phi))
    phi=0
endif

if(stringmatch(StateId,"QPT")||stringmatch(StateId,"QPT"))

    WAVE/T TomoAxisQ1
    WAVE/T TomoAxisQ2
    WAVE TomoAngleQ1
    WAVE TomoAngleQ2

    toground2Q()
    unitaryevolution2Q(TomoAxisQ1[val],TomoAngleQ1[val]);
    unitaryevolution2Q(TomoAxisQ2[val],TomoAngleQ2[val]);
endif

// |00>.
if(stringmatch(StateId,"gg")||stringmatch(StateId,"00"))
    toground2Q();
endif
// |01>
if(stringmatch(StateId,"ge")||stringmatch(StateId,"01"))
    make/O/C/D/N=(4) PsiT_2Q=0
    PsiT_2Q[1]=1;
endif
// |10>
if(stringmatch(StateId,"eg")||stringmatch(StateId,"10"))
    make/O/C/D/N=(4) PsiT_2Q=0
    PsiT_2Q[2]=1;
endif
// |11>
if(stringmatch(StateId,"ee")||stringmatch(StateId,"11"))
    make/O/C/D/N=(4) PsiT_2Q=0
    PsiT_2Q[3]=1;
endif

// |+x,+x>.
if(stringmatch(StateId,"Y90pY90p")||stringmatch(StateId,"y90py90p"))
    make/O/C/D/N=(4) PsiT_2Q=1/2;
endif

// Bell, even, +>.
if(stringmatch(StateId,"BEp")||stringmatch(StateId,"bep"))
    make/O/C/D/N=(4) PsiT_2Q=0;
    PsiT_2Q[0]=1/sqrt(2);
    PsiT_2Q[3]=1/sqrt(2);
endif
// Bell, even, ->.
if(stringmatch(StateId,"BEm")||stringmatch(StateId,"bem"))
    make/O/C/D/N=(4) PsiT_2Q=0;
    PsiT_2Q[0]=1/sqrt(2);
    PsiT_2Q[3]=-1/sqrt(2);
endif

// Bell, odd, +>.
if(stringmatch(StateId,"BOp")||stringmatch(StateId,"BOp"))
    make/O/C/D/N=(4) PsiT_2Q=0;
    PsiT_2Q[1]=1/sqrt(2);
    PsiT_2Q[2]=1/sqrt(2);
endif

// Bell, odd, ->. Also the singlet.
if(stringmatch(StateId,"BOm")||stringmatch(StateId,"BOm")||stringmatch(StateId,"singlet"))
    make/O/C/D/N=(4) PsiT_2Q=0;
    PsiT_2Q[1]=1/sqrt(2);
    PsiT_2Q[2]=-1/sqrt(2);
endif

updaterhotheo2Q();
end


















// Function: getConcurrence
// ----------------------------------------
// Follows Wooters PRL 1998.
function getConcurrence([rho])
WAVE/C rho

WAVE/C RhoT_2Q

if(paramisdefault(rho))
    make/o/n=(4,4)/c rho
    rho=RhoT_2Q
endif

WAVE/C/D Sig_yi
WAVE/C/D Sig_iy
MatrixOp/O/C tempmat=rho x  Sig_yi x Sig_iy x ((rho^t)^h)  x Sig_yi x Sig_iy
MatrixEigenV tempmat
WAVE W_eigenvalues
make/o/n=(4) MyEigen
MyEigen[]=real(W_eigenvalues[p]);
sort MyEigen MyEigen
MyEigen[]=sqrt(max(0,MyEigen[p]));

//print myeigen

variable conc=max(0, MyEigen[3]-MyEigen[2]-MyEigen[1]-Myeigen[0]);

return conc
end

// Function: getPureConcurrence
// ----------------------------------------------
function getPureConcurrence(thisPsi)
WAVE/C thisPsi

variable conc=2*cabs(thisPsi[0]*thisPsi[3]-thisPsi[1]*thisPsi[2])
return conc

end

// Function: getCost
// ----------------------------
function getCost(K)
variable K

WAVE/C RhoT_2Q
WAVE ProbVec
WAVE AlphaMat

variable numP=dimsize(ProbVec,0);

make/o/n=(4)/c thisPsi
make/o/n=(4,4)/c thisRho=0;

variable i=0
Do
    thisPsi[0]=cmplx(AlphaMat[i][0],AlphaMat[i][1]);
    thisPsi[1]=cmplx(AlphaMat[i][2],AlphaMat[i][3]);
    thisPsi[2]=cmplx(AlphaMat[i][4],AlphaMat[i][5]);
    thisPsi[3]=cmplx(AlphaMat[i][6],AlphaMat[i][7]);

    matrixop/o/c tempC= thisPsi x thisPsi^h
    thisRho+=ProbVec[i]*tempC
    i+=1
While(i<numP);

variable conc=getconcurrence(rho=thisRho);

matrixop/o/c mat=(RhoT_2Q-thisRho);
matrixop/o/c tmat=mat^h x mat;
variable mismatch=real(tmat[0][0]+tmat[1][1]+tmat[2][2]+tmat[3][3]);

//print conc, mismatch

return conc + K * mismatch
end

// Function: initMC
// --------------------------
function initMC(numP)
variable numP
make/o/n=(numP) ProbVec;
make/o/n=(numP,8) AlphaMat=0;

Probvec=1/numP;
AlphaMat[][]=(q==2*p);



//AlphaMat[0][0]=0
//AlphaMat[0][1]=0
//AlphaMat[0][2]=0
//AlphaMat[0][3]=0
//AlphaMat[0][4]=0
//AlphaMat[0][5]=0
//AlphaMat[0][6]=0
//AlphaMat[0][7]=0

//AlphaMat[1][0]=0
//AlphaMat[1][1]=0
//AlphaMat[1][2]=0
//AlphaMat[1][3]=0
//AlphaMat[1][4]=0
//AlphaMat[1][5]=0
//AlphaMat[1][6]=1
//AlphaMat[1][7]=0
end

// Function: KickRho
// ---------------------------
function  KickRho(sdevp,sdeva)
variable sdevp
variable sdeva

WAVE ProbVec;
WAVE AlphaMat;

variable numP=dimsize(ProbVec,0);

variable i=0
Do
    ProbVec[i]+=max(0, ProbVec[i]+gnoise(sdevp));
    i+=1
While(i<numP)
variable thissum=sum(ProbVec)
ProbVec/=thissum;

i=0;
variable j=0;
variable thismagsq
Do
    j=0;
    thismagsq=0
    Do
        AlphaMat[i][j]+=gnoise(sdeva);
        thismagsq+=(AlphaMat[i][j])^2
        j+=1
    While(j<8);
    thismagsq=sqrt(thismagsq);
    AlphaMat[i][]/=thismagsq;
    i+=1
While(i<numP);

end


// Function: doonestep
// ------------------------------
function doonestep()
variable sdevp=0.1;
variable sdeva=0.1;
variable K=100;
kickrho(sdevp, sdeva);
print getcost(K);

end


// Function: doMC
// -----------------------
function doMC(numsteps,thisbeta)
variable numsteps
variable thisbeta

variable sdevp=0.0001;
variable sdeva=0.0001;
variable K=10000;
//variable thisbeta=1/1;

make/o/n=(numsteps) CostVec=NaN;

WAVE ProbVec
WAVE AlphaMat
duplicate/o ProbVec thisProbVec
duplicate/o AlphaMat thisAlphaMat

//initmc()
variable thiscost=getcost(K);
variable newcost;
variable i=0;
Do
    CostVec[i]=thiscost;
    thisProbVec=ProbVec;
    thisAlphaMat=AlphaMat;

    kickrho(sdevp, sdeva);

    newcost=getCost(K);

    if(newcost<thiscost)
        thiscost=newcost
    elseif(passupdate(thiscost, newcost,thisbeta)==1)
        thiscost=newcost
    else
        ProbVec=thisProbVec;
        AlphaMat=thisAlphaMat;
    endif
    i+=1
while(i<numsteps);

print thiscost
end

function passupdate(costold,costnew,thisbeta)
variable costold
variable costnew
variable thisbeta
variable boltzmann=e^(-thisbeta*(costnew-costold))
variable p=(1+enoise(1))/2;
variable val=0;
if(p < boltzmann)
    val=1;
endif
return val;

end









