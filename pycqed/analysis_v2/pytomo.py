
from cmath import *
import re, os, math, sys, time, subprocess, string, getopt
from numpy import zeros, eye
import uuid

i=j=1j
"""
File that prepares the input and reads the output from the .exe executable file.
The executable contains a Convex semidefinite programming code, that is a faster version of MLE.
This code was originally developed by NATHAN LANGFORD.

"""

#--------------------------------------------------------------------------------
def bases(dim,triangular=False):
    '''
    Generate the Gell-Mann bases as a list of non-zero values
    (they tend to be very sparse)

    if triangular==True, do only the upper triangular region
    '''

    invsqrtdim = 1/float(math.sqrt(dim))
    invsqrt2 = 1.0/math.sqrt(2)

    gellmann=[]
    for m in range(dim):
        for n in range(m+1):
            B=[]
            if m==0 and n==0:
                # I/sqrt(d)
                for ii in range(dim):
                    B.append((ii,ii,invsqrtdim))
                gellmann.append(B)  
            elif m==n:
                v=-1.0/math.sqrt(m*(m+1))
                for ii in range(m):
                    B.append((ii,ii,v))
                B.append((m,m,-m*v))
                gellmann.append(B)  
            else:
                if not triangular:
                    B.append((m,n,invsqrt2))
                B.append((n,m,invsqrt2))
                gellmann.append(B)
                
                B=[]
                if not triangular:
                    B.append((m,n,1j*invsqrt2))
                B.append((n,m,-1j*invsqrt2))
                gellmann.append(B)

    return gellmann

#--------------------------------------------------------------------------------
def readsol(filename,OPT):
    '''
    structure of solution file:

    csdp:
        vector x in one line
        (primal) matrix Z in sparse sdpa format
        (dual)   matrix X in sparse sdpa format 

    dsdp5:
        vector x in one line
        ... other stuff ...
    sdplr:
        vector -x one number per line
        ... other stuff ...
    '''

    x=[]

    # XXX make this more robust to initial comments?
    # first line contains x
    fp = open(filename)

    if OPT['solver']=="sdplr":

        # skip first line
        line=fp.readline()
        
        # now read list of numbers 1 per line
        # (next section has multiple numbers / line)
        while True:
            line = fp.readline().strip().split()
            if len(line)>1:
                break
            else:
                # sdplr gives -x
                x.append(-float(line[0]))

    elif OPT['solver']=="csdp" or OPT['solver']=="dsdp5":
        # solution on first line
        line=fp.readline()
        x = list(map(float,line.strip().split()))
    else:
        raise 'Unknown solver "%s"'%OPT['solver']

    fp.close()
 
    return x

def writesol(rho,filename):
    '''
    Write out reconstructed matrix. Real and imaginary parts are separate files for 
    ease of reading with other programs. The files are the comma separated entries
    for rho.
    '''   
 
    fpr = open(filename+'.rhor','w')
    fpi = open(filename+'.rhoi','w')

    for ii in range(rho.dim):
        lr = [];
        li = [];
        for jj in range(rho.dim):
            v = complex(rho[ii,jj])
            lr.append(str(v.real))
            li.append(str(v.imag))
        fpr.write(string.join(lr,', ')+'\n')
        fpi.write(string.join(li,', ')+'\n')

    fpr.close()
    fpi.close()
    
#--------------------------------------------------------------------------------
def pretty_print(rho,tol=1e-4,dp=3):
    '''
    print matrix in a nice way
    '''
    
    print() 
    for ii in range(rho.dim):
        values = []
        for jj in range(rho.dim):
            v = complex(rho[ii,jj])
            if abs(v)< tol:
                values.append("0")
            elif abs(v.imag) < tol:
                values.append("% g"%round(v.real,dp))
            elif abs(v.real) < tol:
                values.append("% g*i"%round(v.imag,dp))
            else:
                values.append("% g+% g*i"%(round(v.real,dp),round(v.imag,dp)))

        print(string.join(values,',\t'))
    print()

#--------------------------------------------------------------------------------
def writesdpa_state(data, observables, weights, filename, fixedweight, OPT):
    if fixedweight:
        return writesdpa_state_fw(data, observables, weights, filename, OPT)
    else:
        return writesdpa_state_ml(data, observables, weights, filename, OPT)
        
def writesdpa_state_fw(data, observables, weights, filename, OPT):
    '''
    The Problem
    ~~~~~~~~~~~
    The primal form:
    
    (P)    min c1*x1+c2*x2+...+cm*xm
           st  F1*x1+F2*x2+...+Fm*xn-F0=X>=0

    Here all of the matrices F0, F1, ..., Fm, and X are assumed to be
    symmetric of size n by n.  The constraints X>=0 mean that X
    must be positive semidefinite.  
    
    SDPA File Format
    ~~~~~~~~~~~~~~~~
    The file consists of six sections:
     
    1. Comments.  The file can begin with arbitrarily many lines of comments.
    Each line of comments must begin with '"' or '*'.  
    
    2. The first line after the comments contains m, the number of constraint
    matrices.  Additional text on this line after m is ignored.
     
    3. The second line after the comments contains nblocks, the number of 
    blocks in the block diagonal structure of the matrices.  Additional text
    on this line after nblocks is ignored.  
     
    4. The third line after the comments contains a vector of numbers that 
    give the sizes of the individual blocks.  
    Negative numbers may be used to indicate that a block is actually a diagonal
    submatrix.  Thus a block size of "-5" indicates a 5 by 5 block in which 
    only the diagonal elements are nonzero.  
    
    5. The fourth line after the comments contains the objective function
    vector c.  
     
    6. The remaining lines of the file contain entries in the constraint
    matrices, with one entry per line.  The format for each line is 
     
      <matno> <blkno> <i> <j> <entry>
     
    Here <matno> is the number of the matrix to which this entry belongs, 
    <blkno> specifies the block within this matrix, <i> and <j> specify a
    location within the block, and <entry> gives the value of the entry in
    the matrix.  Note that since all matrices are assumed to be symmetric, 
    only entries in the upper triangle of a matrix are given.  
    '''

    d = observables[0].shape[0]
    ncontraints = d**2+1  # d^2 - (identity component) + (slack variable) + N variable
    nblocks = 2 # fixed by algorithm
    dim1 = len(data)+1
    
    fp = open(filename,'w')

    fp.write('* File generated by tomoc on %s\n'%time.strftime("%c",time.localtime(time.time())))
    fp.write('%d\n%d\n%d %d\n'%(ncontraints,nblocks,dim1,2*d))
    fp.write('1.0 '+'0.0 '*(d**2)+'\n')

    gellmann = bases(d)
    gellmann_ut = bases(d,triangular=True)

    # F0 -------------------------------------------
    # block 1
    for ii in range(dim1-1):
        #trE = observables[ii].trace()/float(d)
        v = data[ii]
        if v != 0:
            fp.write('0 1 1 %d %f\n'%(ii+2,-v) )
            fp.write('0 1 %d %d %f\n'%(ii+2,ii+2,-v))
        
    # Ft -------------------------------------------
    fp.write('1 1 1 1 1.0\n')

    # Fmu -------------------------------------------
    for mu in range(0,d**2):
        if OPT['verbose']:
            print("writing matrix %d/%d ..."%(mu+2,d**2+2))
        # block 1
        Gmu = gellmann[mu]
        for kk in range(dim1-1):
            Ek = observables[kk] 
            v = 0
            for item in Gmu:
                v += item[-1]*Ek[item[1],item[0]]*weights[kk]

            # XXX should check imaginary component is small
            if isinstance(v,complex):
                v=v.real

            if v == 0:
                continue
            
            fp.write('%d 1 1 %d %f\n'%(mu+2,kk+2,-v))
            #fp.write('%d 1 %d %d %f\n'%(mu+2,kk+2,kk+2,v))

        # block 2
        # we want all of the imaginary part:
        #Gmu=gellmann[mu]
        for item in Gmu:
            Gi = complex(item[2]).imag
            if Gi != 0:
                fp.write('%d 2 %d %d %f\n'%(mu+2, item[0]+1, item[1]+1+d,-Gi))

        # but only want the upper triangular of the real part:
        Gmu=gellmann_ut[mu]
        for item in Gmu:
            Gr = complex(item[2]).real
            if Gr != 0:
                fp.write('%d 2 %d %d %f\n'%(mu+2, item[0]+1, item[1]+1,Gr))
                fp.write('%d 2 %d %d %f\n'%(mu+2, item[0]+1+d, item[1]+1+d,Gr))

    fp.close()

    return d

def writesdpa_state_ml(data, observables, weights, filename, OPT):
    '''
    The Problem
    ~~~~~~~~~~~
    The primal form:
    
    (P)    min c1*x1+c2*x2+...+cm*xm
           st  F1*x1+F2*x2+...+Fm*xn-F0=X>=0

    Here all of the matrices F0, F1, ..., Fm, and X are assumed to be
    symmetric of size n by n.  The constraints X>=0 mean that X
    must be positive semidefinite.  
    
    SDPA File Format
    ~~~~~~~~~~~~~~~~
    The file consists of six sections:
     
    1. Comments.  The file can begin with arbitrarily many lines of comments.
    Each line of comments must begin with '"' or '*'.  
    
    2. The first line after the comments contains m, the number of constraint
    matrices.  Additional text on this line after m is ignored.
     
    3. The second line after the comments contains nblocks, the number of 
    blocks in the block diagonal structure of the matrices.  Additional text
    on this line after nblocks is ignored.  
     
    4. The third line after the comments contains a vector of numbers that 
    give the sizes of the individual blocks.  
    Negative numbers may be used to indicate that a block is actually a diagonal
    submatrix.  Thus a block size of "-5" indicates a 5 by 5 block in which 
    only the diagonal elements are nonzero.  
    
    5. The fourth line after the comments contains the objective function
    vector c.  
     
    6. The remaining lines of the file contain entries in the constraint
    matrices, with one entry per line.  The format for each line is 
     
      <matno> <blkno> <i> <j> <entry>
     
    Here <matno> is the number of the matrix to which this entry belongs, 
    <blkno> specifies the block within this matrix, <i> and <j> specify a
    location within the block, and <entry> gives the value of the entry in
    the matrix.  Note that since all matrices are assumed to be symmetric, 
    only entries in the upper triangle of a matrix are given.  
    '''

    d = observables[0].shape[0]
    nconstraints = d**2+1  # d^2 - (identity component) + (slack variable) + N variable
    nblocks = 2 # fixed by algorithm
    dim1 = len(data)+1
    
    fp = open(filename,'w')

    fp.write('* File generated by tomoc on %s\n'%time.strftime("%c",time.localtime(time.time())))
    fp.write('%d\n%d\n%d %d\n'%(nconstraints,nblocks,dim1,2*d))
    fp.write('1.0 '+'0.0 '*(d**2)+'\n')

    gellmann = bases(d)
    gellmann_ut = bases(d,triangular=True)

    # F0 -------------------------------------------
    # block 1
    for ii in range(dim1-1):
        #trE = observables[ii].trace()/float(d)
        v = data[ii]
        if v != 0:
            fp.write('0 1 1 %d %f\n'%(ii+2,-v) )
        
    # Ft -------------------------------------------
    fp.write('1 1 1 1 1.0\n')

    # Fmu -------------------------------------------
    for mu in range(0,d**2):
        if OPT['verbose']:
            print("writing matrix %d/%d ..."%(mu+2,d**2+2))
        # block 1
        Gmu = gellmann[mu]
        for kk in range(dim1-1):
            Ek = observables[kk] 
            v = 0
            for item in Gmu:
                v += item[-1]*Ek[item[1],item[0]]*weights[kk]

            # XXX should check imaginary component is small
            if isinstance(v,complex):
                v=v.real

            if v == 0:
                continue
            
            fp.write('%d 1 1 %d %f\n'%(mu+2,kk+2,-v))
            fp.write('%d 1 %d %d %f\n'%(mu+2,kk+2,kk+2,v))

        # block 2
        # we want all of the imaginary part:
        #Gmu=gellmann[mu]
        for item in Gmu:
            Gi = complex(item[2]).imag
            if Gi != 0:
                fp.write('%d 2 %d %d %f\n'%(mu+2, item[0]+1, item[1]+1+d,-Gi))

        # but only want the upper triangular of the real part:
        Gmu=gellmann_ut[mu]
        for item in Gmu:
            Gr = complex(item[2]).real
            if Gr != 0:
                fp.write('%d 2 %d %d %f\n'%(mu+2, item[0]+1, item[1]+1,Gr))
                fp.write('%d 2 %d %d %f\n'%(mu+2, item[0]+1+d, item[1]+1+d,Gr))

    fp.close()

    return d

#--------------------------------------------------------------------------------
def writesdpa_process(data, inputs, observables, weights, filename, fixedweight, OPT):
    if fixedweight:
        return writesdpa_process_fw(data, inputs, observables, weights, filename, OPT)
    else:
        return writesdpa_process_ml(data, inputs, observables, weights, filename, OPT)
        
def writesdpa_process_fw(data, inputs, observables, weights, filename, OPT):
    '''
    The Problem
    ~~~~~~~~~~~
    The primal form:
    
    (P)    min c1*x1+c2*x2+...+cm*xm
           st  F1*x1+F2*x2+...+Fm*xn-F0=X>=0

    Here all of the matrices F0, F1, ..., Fm, and X are assumed to be
    symmetric of size n by n.  The constraints X>=0 mean that X
    must be positive semidefinite.  
    
    SDPA File Format
    ~~~~~~~~~~~~~~~~
    The file consists of six sections:
     
    1. Comments.  The file can begin with arbitrarily many lines of comments.
    Each line of comments must begin with '"' or '*'.  
    
    2. The first line after the comments contains m, the number of constraint
    matrices.  Additional text on this line after m is ignored.
     
    3. The second line after the comments contains nblocks, the number of 
    blocks in the block diagonal structure of the matrices.  Additional text
    on this line after nblocks is ignored.  
     
    4. The third line after the comments contains a vector of numbers that 
    give the sizes of the individual blocks.  
    Negative numbers may be used to indicate that a block is actually a diagonal
    submatrix.  Thus a block size of "-5" indicates a 5 by 5 block in which 
    only the diagonal elements are nonzero.  
    
    5. The fourth line after the comments contains the objective function
    vector c.  
     
    6. The remaining lines of the file contain entries in the constraint
    matrices, with one entry per line.  The format for each line is 
     
      <matno> <blkno> <i> <j> <entry>
     
    Here <matno> is the number of the matrix to which this entry belongs, 
    <blkno> specifies the block within this matrix, <i> and <j> specify a
    location within the block, and <entry> gives the value of the entry in
    the matrix.  Note that since all matrices are assumed to be symmetric, 
    only entries in the upper triangle of a matrix are given.  
    '''

    d = observables[0].shape[0]
    ncontraints = d**4-d+2  # d^4 - d (TP cond) + (slack variable) + N variable
    nblocks = 2 # fixed by algorithm
    dim1 = len(data)+1
    
    fp = open(filename,'w')

    fp.write('* File generated by protomoc on %s\n'%time.strftime("%c",time.localtime(time.time())))
    fp.write('%d\n%d\n%d %d\n'%(ncontraints,nblocks,dim1,2*d**2))
    fp.write('1 '+'0 '*(d**4)+'\n')

    gellmann = bases(d)
    gellmann_ut = bases(d,triangular=True)

    # F0 -------------------------------------------
    # block 1
    for mn in range(dim1-1):
        v = data[mn]
        if v != 0:
            fp.write('0 1 1 %d %f\n'%(mn+2,-v) )
            fp.write('0 1 %d %d %f\n'%(mn+2,mn+2,-v) )
        
    # Ft -------------------------------------------
    fp.write('1 1 1 1 1.0\n')

    # FN -------------------------------------------
    # block 1
    for mn in range(dim1-1):
        v = weights[mn]*observables[mn].trace()/float(d)
        if v != 0:
            fp.write('2 1 1 %d %f\n'%(mn+2,-v) )
            #fp.write('2 1 %d %d %f\n'%(mn+2,mn+2,v) )
            
    # block 2
    for jj in range(0,2*d**2):
        # to be real symmetric we have [Fr,-Fi;Fr,Fi] and want 
        # upper diagonal only, since this term is identity we can
        # range over 2d**2 instead
        fp.write('2 2 %d %d %f\n'%(jj+1,jj+1,1/float(d**2)) )

    # Fjj -------------------------------------------
    jk = 1  # this is the matrix number
    for jj in range(0,d**2):
        for kk in range(1,d**2):
            if OPT['verbose']:
                print("writing matrix %d/%d ..."%(jj*d**2+kk,d**4-d+1))
            # block 1
            Gj = gellmann[jj]
            Gk = gellmann[kk]
            for mn in range(dim1-1):
                Emn = observables[mn]
                Rmn = inputs[mn]
                TrEmnGk = 0
                TrRmnGj = 0
                # only need to do the non-zero terms
                # this is trace(Rmn^T Gj) = vec(Emn)^T vec(Gj)
                for Gjnz in Gj:
                    TrRmnGj += Rmn[Gjnz[0],Gjnz[1]]*Gjnz[-1]
                # this is trace(Emn Gj) = vec(Emn^T)^T vec(Gj)
                for Gknz in Gk:
                    TrEmnGk += Emn[Gknz[1],Gknz[0]]*Gknz[-1]

                # XXX should check imaginary component is small
                v = d*TrEmnGk*TrRmnGj*weights[mn] 
            
                # both TrEmnGk and TrRmnGj shoulb be real (Gjj hermitian)
                if isinstance(v,complex):
                    v=v.real

                if v == 0:
                    continue
            
                fp.write('%d 1 1 %d %f\n'%(jk+2,mn+2,-v))
                #fp.write('%d 1 %d %d %f\n'%(jk+2,mn+2,mn+2,v))

            # block 2
            for gj in Gj:
                for gk in Gk:
                    
                    # work out tensor product Gj . Gk
                    gjgk = (gj[0]*d+gk[0],gj[1]*d+gk[1],gj[2]*gk[2])

                    # need to split this up hermitian -> real symmetric
                    
                    # we only want the upper triangular of the real part:
                    if gjgk[1] >= gjgk[0]:
                        v =complex(gjgk[2]).real
                        if v != 0:
                            fp.write('%d 2 %d %d %f\n'%(jk+2, gjgk[0]+1, gjgk[1]+1,v))
                            fp.write('%d 2 %d %d %f\n'%(jk+2, gjgk[0]+1+d**2, gjgk[1]+1+d**2,v))

                            
                    #Gjj=gellmann_ut[jj]
                    #for item in Gjj:
                    #   Gr = complex(item[2]).real
                    #   if Gr != 0:
                    #       fp.write('%d 2 %d %d %f\n'%(jj+2, item[0]+1, item[1]+1,Gr))
                    #       fp.write('%d 2 %d %d %f\n'%(jj+2, item[0]+1+d, item[1]+1+d,Gr))

                    # but all of the imaginary part:
                    v =complex(gjgk[2]).imag
                    fp.write('%d 2 %d %d %f\n'%(jk+2, gjgk[0]+1, gjgk[1]+1+d**2,-v))
                    #Gjj=gellmann[jj]
                    #for item in Gjj:
                    #   Gi = complex(item[2]).imag
                    #   if Gi != 0:
                    #       fp.write('%d 2 %d %d %f\n'%(jj+2, item[0]+1, item[1]+1+d,-Gi))
            jk+=1
    fp.close()

    return d

def writesdpa_process_ml(data, inputs, observables, weights, filename, OPT):
    '''
    The Problem
    ~~~~~~~~~~~
    The primal form:
    
    (P)    min c1*x1+c2*x2+...+cm*xm
           st  F1*x1+F2*x2+...+Fm*xn-F0=X>=0

    Here all of the matrices F0, F1, ..., Fm, and X are assumed to be
    symmetric of size n by n.  The constraints X>=0 mean that X
    must be positive semidefinite.  
    
    SDPA File Format
    ~~~~~~~~~~~~~~~~
    The file consists of six sections:
     
    1. Comments.  The file can begin with arbitrarily many lines of comments.
    Each line of comments must begin with '"' or '*'.  
    
    2. The first line after the comments contains m, the number of constraint
    matrices.  Additional text on this line after m is ignored.
     
    3. The second line after the comments contains nblocks, the number of 
    blocks in the block diagonal structure of the matrices.  Additional text
    on this line after nblocks is ignored.  
     
    4. The third line after the comments contains a vector of numbers that 
    give the sizes of the individual blocks.  
    Negative numbers may be used to indicate that a block is actually a diagonal
    submatrix.  Thus a block size of "-5" indicates a 5 by 5 block in which 
    only the diagonal elements are nonzero.  
    
    5. The fourth line after the comments contains the objective function
    vector c.  
     
    6. The remaining lines of the file contain entries in the constraint
    matrices, with one entry per line.  The format for each line is 
     
      <matno> <blkno> <i> <j> <entry>
     
    Here <matno> is the number of the matrix to which this entry belongs, 
    <blkno> specifies the block within this matrix, <i> and <j> specify a
    location within the block, and <entry> gives the value of the entry in
    the matrix.  Note that since all matrices are assumed to be symmetric, 
    only entries in the upper triangle of a matrix are given.  
    '''

    d = observables[0].shape[0]
    ncontraints = d**4-d+2  # d^4 - d (TP cond) + (slack variable) + N variable
    nblocks = 2 # fixed by algorithm
    dim1 = len(data)+1
    
    fp = open(filename,'w')

    fp.write('* File generated by protomoc on %s\n'%time.strftime("%c",time.localtime(time.time())))
    fp.write('%d\n%d\n%d %d\n'%(ncontraints,nblocks,dim1,2*d**2))
    fp.write('1 '+'0 '*(d**4)+'\n')

    gellmann = bases(d)
    gellmann_ut = bases(d,triangular=True)

    # F0 -------------------------------------------
    # block 1
    for mn in range(dim1-1):
        v = data[mn]
        if v != 0:
            fp.write('0 1 1 %d %f\n'%(mn+2,-v) )
        
    # Ft -------------------------------------------
    fp.write('1 1 1 1 1.0\n')

    # FN -------------------------------------------
    # block 1
    for mn in range(dim1-1):
        v = weights[mn]*observables[mn].trace()/float(d)
        if v != 0:
            fp.write('2 1 1 %d %f\n'%(mn+2,-v) )
            fp.write('2 1 %d %d %f\n'%(mn+2,mn+2,v) )
            
    # block 2
    for jj in range(0,2*d**2):
        # to be real symmetric we have [Fr,-Fi;Fr,Fi] and want 
        # upper diagonal only, since this term is identity we can
        # range over 2d**2 instead
        fp.write('2 2 %d %d %f\n'%(jj+1,jj+1,1/float(d**2)) )

    # Fjj -------------------------------------------
    jk = 1  # this is the matrix number
    for jj in range(0,d**2):
        for kk in range(1,d**2):
            if OPT['verbose']:
                print("writing matrix %d/%d ..."%(jj*d**2+kk,d**4-d+1))
            # block 1
            Gj = gellmann[jj]
            Gk = gellmann[kk]
            for mn in range(dim1-1):
                Emn = observables[mn]
                Rmn = inputs[mn]
                TrEmnGk = 0
                TrRmnGj = 0
                # only need to do the non-zero terms
                # this is trace(Rmn^T Gj) = vec(Emn)^T vec(Gj)
                for Gjnz in Gj:
                    TrRmnGj += Rmn[Gjnz[0],Gjnz[1]]*Gjnz[-1]
                # this is trace(Emn Gj) = vec(Emn^T)^T vec(Gj)
                for Gknz in Gk:
                    TrEmnGk += Emn[Gknz[1],Gknz[0]]*Gknz[-1]

                # XXX should check imaginary component is small
                v = d*TrEmnGk*TrRmnGj*weights[mn] 
            
                # both TrEmnGk and TrRmnGj shoulb be real (Gjj hermitian)
                if isinstance(v,complex):
                    v=v.real

                if v == 0:
                    continue
            
                fp.write('%d 1 1 %d %f\n'%(jk+2,mn+2,-v))
                fp.write('%d 1 %d %d %f\n'%(jk+2,mn+2,mn+2,v))

            # block 2
            for gj in Gj:
                for gk in Gk:
                    
                    # work out tensor product Gj . Gk
                    gjgk = (gj[0]*d+gk[0],gj[1]*d+gk[1],gj[2]*gk[2])

                    # need to split this up hermitian -> real symmetric
                    
                    # we only want the upper triangular of the real part:
                    if gjgk[1] >= gjgk[0]:
                        v =complex(gjgk[2]).real
                        if v != 0:
                            fp.write('%d 2 %d %d %f\n'%(jk+2, gjgk[0]+1, gjgk[1]+1,v))
                            fp.write('%d 2 %d %d %f\n'%(jk+2, gjgk[0]+1+d**2, gjgk[1]+1+d**2,v))

                            
                    #Gjj=gellmann_ut[jj]
                    #for item in Gjj:
                    #   Gr = complex(item[2]).real
                    #   if Gr != 0:
                    #       fp.write('%d 2 %d %d %f\n'%(jj+2, item[0]+1, item[1]+1,Gr))
                    #       fp.write('%d 2 %d %d %f\n'%(jj+2, item[0]+1+d, item[1]+1+d,Gr))

                    # but all of the imaginary part:
                    v =complex(gjgk[2]).imag
                    fp.write('%d 2 %d %d %f\n'%(jk+2, gjgk[0]+1, gjgk[1]+1+d**2,-v))
                    #Gjj=gellmann[jj]
                    #for item in Gjj:
                    #   Gi = complex(item[2]).imag
                    #   if Gi != 0:
                    #       fp.write('%d 2 %d %d %f\n'%(jj+2, item[0]+1, item[1]+1+d,-Gi))
            jk+=1
    fp.close()

    return d

#--------------------------------------------------------------------------------
def reconstructrho_state(x, dim, OPT):
    '''
    reconstruct solution as density matrix rho
    '''

    # 1st value is the min objective function
    # 2nd value is the normalisation N
    N=float(x[1])*sqrt(dim)
    if OPT['verbose']:
        print("Normalisation = ", N)
    
    # get basis matrices
    gellmann=bases(dim)

    rho = zeros([dim, dim], dtype=complex)
    for ii in range(dim**2):
        for item in gellmann[ii]:
            rho[item[0], item[1]] +=  x[ii+1]*item[2]

    if OPT['normalised']:
        rho = rho/rho.trace()

    #rho = eye(dim)/float(dim)
    #for ii in range(1,dim**2):
    #    for item in gellmann[ii]:
    #        rho[item[0],item[1]] +=  x[ii+1]*item[2]/N

    return rho

def reconstructrho_process(x, dim, OPT):
    '''
    reconstruct solution as density matrix rho
    '''

    # 1st value is the min objective function
    # 2nd value is the normalisation N
    N=float(x[1])
    if OPT['verbose']:
        print("Normalisation = ", N)
    
    # get basis matrices
    gellmann=bases(dim)
 
    rho = eye(dim**2, dtype=complex)/float(dim**2)*N 
    
    jk = 1  # this is the matrix number
    for jj in range(0,dim**2):
        for kk in range(1,dim**2):
            Gj = gellmann[jj]
            Gk = gellmann[kk]
            for gj in Gj:
                for gk in Gk:
                    # work out tensor product Gj . Gk
                    gjgk = (gj[0]*dim+gk[0],gj[1]*dim+gk[1],gj[2]*gk[2])
                    rho[gjgk[0],gjgk[1]] +=  x[jk+1]*gjgk[2]
            jk+=1

    if OPT['normalised']:
        rho = rho/rho.trace()
    return rho

#--------------------------------------------------------------------------------
def tomo_state(data, observables, weights, filebase=None, fixedweight=True, tomo_options={}):

    OPT={'verbose':False, 'prettyprint':False, 'solver':'csdp', 'normalised':False}
    for o, a in list(tomo_options.items()):
        OPT[o] = a
    # default option defaults

    if filebase is None:
        filebase = 'temp' + str(uuid.uuid4())
    
    fullfilepath = os.path.abspath(globals()['__file__'])
    filedir,tmp = os.path.split(fullfilepath)
    orig_dir = os.getcwd()

    if OPT['verbose']:
        print("changing dir to: %s"% filedir)
    os.chdir(filedir)

    if OPT['verbose']:
        print("writing sdpa control file: %s.spda"%filebase)
    dim = writesdpa_state(data, observables, weights, filebase+'.sdpa', fixedweight, OPT)

    if OPT['solver']=='csdp':
        command = 'csdp %s.sdpa %s.sol'%(filebase,filebase)
        if OPT['verbose']:
            print("running: "+command)
        # else:
        #     status, out =  commands.getstatusoutput(command)
        os.system(command)

    elif OPT['solver']=='dsdp5':
        # WARNING: dsdp5 arbitrarily truncates -save path!
        command = 'dsdp5 %s.sdpa -save %s.sol'%(filebase,filebase)
        if OPT['verbose']:
            print("running: "+command)
        # else:
        #     status, out =  commands.getstatusoutput(command)
        os.system(command)

    elif OPT['solver']=='sdplr':
        command = 'sdplr %s.sdpa non_existent_file non_existent_file %s.sol'%(filebase,filebase)
        if OPT['verbose']:
            print("running: "+command)
        # else:
        #     status, out =  commands.getstatusoutput(command)
        os.system(command)
    else:
        raise "unknown solver: %s"%OPT["solver"]

    if OPT['verbose']:
        print("reading in solution from %s.sol"%filebase)
    x = readsol("%s.sol"%filebase, OPT)

    # len(x) = d**2+1  [ d^2 - (identity component) + (slack variable) + N variable ]

    rho = reconstructrho_state(x, dim, OPT)

    if OPT['prettyprint']:
        pretty_print(rho)

    # clean up temporary files
    os.remove("%s.sdpa"%filebase)
    os.remove("%s.sol"%filebase)

    os.chdir(orig_dir)

    return rho

def tomo_process(data, inputs, observables, weights, filebase=None, fixedweight=True, tomo_options={}):

    OPT={'verbose':False, 'prettyprint':False, 'solver':'csdp', 'normalised':False}
    for o, a in list(tomo_options.items()):
        OPT[o] = a
    # default option defaults

    if filebase is None:
        filebase = 'temp' + str(uuid.uuid4())
    
    fullfilepath = os.path.abspath(globals()['__file__'])
    filedir,tmp = os.path.split(fullfilepath)
    orig_dir = os.getcwd()

    if OPT['verbose']:
        print("changing dir to: %s"%dir)
    os.chdir(filedir)

    if OPT['verbose']:
        print("writing sdpa control file: %s.spda"%filebase)
    dim = writesdpa_process(data, inputs, observables, weights, filebase+'.sdpa', fixedweight, OPT)

    if OPT['solver']=='csdp':
        command = 'csdp %s.sdpa %s.sol'%(filebase,filebase)
        if OPT['verbose']:
            print("running: "+command)
        # else:
        #     status, out =  commands.getstatusoutput(command)
        os.system(command)

    elif OPT['solver']=='dsdp5':
        # WARNING: dsdp5 arbitrarily truncates -save path! 
        command = 'dsdp5 %s.sdpa -save %s.sol'%(filebase,filebase)
        if OPT['verbose']:
            print("running: "+command)
        # else:
        #     status, out =  commands.getstatusoutput(command)
        os.system(command)

    elif OPT['solver']=='sdplr':
        command = 'sdplr %s.sdpa non_existent_file non_existent_file %s.sol'%(filebase,filebase)
        if OPT['verbose']:
            print("running: "+command)
        # else:
        #     status, out =  commands.getstatusoutput(command)
        os.system(command)
    else:
        raise "unknown solver: %s"%OPT["solver"]
    

    if OPT['verbose']:
        print("reading in solution from %s.sol"%filebase)
    x = readsol("%s.sol"%filebase,OPT)

    # len(x) = d**4-d+2  [d^4 - d (TP cond) + (slack variable) + N variable ]

    rho = reconstructrho_process(x, dim, OPT)

    if OPT['prettyprint']:
        pretty_print(rho)

    # # clean up temporary files
    # os.remove("%s.sdpa"%filebase)
    # os.remove("%s.sol"%filebase)

    os.chdir(orig_dir)

    return rho

#--------------------------------------------------------------------------------
