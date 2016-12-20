from pycqed.analysis import measurement_analysis as ma
import numpy as np
import pylab
# %matplotlib inline
from matplotlib import pyplot as plt
from numpy.linalg import inv

def two_qubit_ssro_fidelity(label, fig_format='png'):
    #extracting data sets
    # qubits=['q0','q1', 'q2', 'q3']
    states=['00','01','10','11']
    list_qubits=['q0', 'q1']
    nr_states = len(states)
    namespace = globals()

    #ma.SSRO_Analysis(label =label, auto=True, channels=['w0'])
    data=ma.MeasurementAnalysis(auto=False, label=label)

    #ma.SSRO_Analysis(auto=False)
    #extract fit parameters for q0
    w0_data=data.get_values('w0')
    w1_data=data.get_values('w1')
    lengths=[]
    i=0
    for nr_state, state in enumerate(states):
        if i==0:
            namespace['w0_data_r{}'.format(state)]=[]
            namespace['w1_data_r{}'.format(state)]=[]

    for nr_state, state in enumerate(states):
        namespace['w0_data_sub_{}'.format(state)]=w0_data[nr_state::nr_states]
        namespace['w1_data_sub_{}'.format(state)]=w1_data[nr_state::nr_states]
        lengths.append(len(w0_data[nr_state::nr_states]))

    #capping off the maximum lengths
    min_len = np.min(lengths)
    for nr_state, state in enumerate(states):
        namespace['w0_data_sub_{}'.format(state)]= namespace['w0_data_sub_{}'.format(state)][0:min_len]
        namespace['w1_data_sub_{}'.format(state)]= namespace['w1_data_sub_{}'.format(state)][0:min_len]
    for nr_state, state in enumerate(states):
        namespace['w0_data_r{}'.format(state)]+=list(namespace['w0_data_sub_{}'.format(state)])
        namespace['w1_data_r{}'.format(state)]+=list(namespace['w1_data_sub_{}'.format(state)])
    # for state in states:
    #     namespace['w0_data_{}'.format(state)]=np.transpose(np.array(namespace['w0_data_{}'.format(state)]))
    #     namespace['w2_data_{}'.format(state)]=np.transpose(np.array(namespace['w2_data_{}'.format(state)]))

    for nr_state, state in enumerate(states):
         namespace['w0_data_{}'.format(state)]=namespace['w0_data_r{}'.format(state)]
         namespace['w1_data_{}'.format(state)]=namespace['w1_data_r{}'.format(state)]

    min_len_all=min_len/2
    for  state in states:
         namespace['w0_data_{}'.format(state)]=namespace['w0_data_r{}'.format(state)]
         namespace['w1_data_{}'.format(state)]=namespace['w1_data_r{}'.format(state)]


    #plot results for q0
    ma.SSRO_Analysis(label=label, auto=True, channels=['w0'], sample_0=0,
                                  sample_1=1, nr_samples=4, rotate=False)
    ana=ma.MeasurementAnalysis(labl=label,auto=False)
    ana.load_hdf5data()
    Fa=ana.data_file['Analysis']['SSRO_Fidelity'].attrs['F_a']
    Fd=ana.data_file['Analysis']['SSRO_Fidelity'].attrs['F_d']
    mu0_0=ana.data_file['Analysis']['SSRO_Fidelity'].attrs['mu0_0']
    mu1_0=ana.data_file['Analysis']['SSRO_Fidelity'].attrs['mu1_0']
    mu0_1=ana.data_file['Analysis']['SSRO_Fidelity'].attrs['mu0_1']
    mu1_1=ana.data_file['Analysis']['SSRO_Fidelity'].attrs['mu1_1']

    sigma0_0=ana.data_file['Analysis']['SSRO_Fidelity'].attrs['sigma0_0']
    sigma1_1=ana.data_file['Analysis']['SSRO_Fidelity'].attrs['sigma1_1']
    sigma0_1=ana.data_file['Analysis']['SSRO_Fidelity'].attrs['sigma0_1']
    sigma1_0=ana.data_file['Analysis']['SSRO_Fidelity'].attrs['sigma1_0']
    frac1_0=ana.data_file['Analysis']['SSRO_Fidelity'].attrs['frac1_0']
    frac1_1=ana.data_file['Analysis']['SSRO_Fidelity'].attrs['frac1_1']
    V_opt=ana.data_file['Analysis']['SSRO_Fidelity'].attrs['V_th_a']
    SNR=ana.data_file['Analysis']['SSRO_Fidelity'].attrs['SNR']

    n, bins0, patches = pylab.hist(namespace['w0_data_00'], bins=int(min_len_all/50),
                                          label = 'input state {}'.format(state),histtype='step',
                                          color='red', normed=True, visible=False)
    n, bins1, patches = pylab.hist(namespace['w0_data_01'], bins=int(min_len_all/50),
                                          label = 'input state {}'.format(state),histtype='step',
                                          color='red', normed=True, visible=False)
    pylab.clf()
    fig, axes = plt.subplots(figsize=(8, 5))
    colors=['blue', 'red', 'grey', 'magenta']
    markers=['o','o','o','v']

    for marker, color,state in zip(markers,colors,states):

        n, bins, patches = pylab.hist(namespace['w0_data_{}'.format(state)], bins=int(min_len_all/50),
                                      histtype='step',  normed=True, visible=False)
        pylab.plot(bins[:-1]+0.5*(bins[1]-bins[0]),n, color=color, linestyle='None',marker=marker,label = '|{}>'.format(state))

    y0 = (1-frac1_0)*pylab.normpdf(bins0, mu0_0, sigma0_0)+frac1_0*pylab.normpdf(bins0, mu1_0, sigma1_0)
    y1_0 = frac1_0*pylab.normpdf(bins0, mu1_0, sigma1_0)
    y0_0 = (1-frac1_0)*pylab.normpdf(bins0, mu0_0, sigma0_0)

    #building up the histogram fits for on measurements
    y1 = (1-frac1_1)*pylab.normpdf(bins1, mu0_1, sigma0_1)+frac1_1*pylab.normpdf(bins1, mu1_1, sigma1_1)
    y1_1 = frac1_1*pylab.normpdf(bins1, mu1_1, sigma1_1)
    y0_1 = (1-frac1_1)*pylab.normpdf(bins1, mu0_1, sigma0_1)


    pylab.semilogy(bins0, y0, 'b',linewidth=1.5, label='fit |00>')
    # pylab.semilogy(bins0, y1_0, 'b--', linewidth=3.5)
    # pylab.semilogy(bins0, y0_0, 'b--', linewidth=3.5)

    pylab.semilogy(bins1, y1, 'r',linewidth=1.5, label='fit |01>')
    #pylab.semilogy(bins1, y0_1, 'r--', linewidth=3.5)
    #pylab.semilogy(bins1, y1_1, 'r--', linewidth=3.5)
    (pylab.gca()).set_ylim(0.2e-6,1e-3)
    pdf_max=(max(max(y0),max(y1)))
    (pylab.gca()).set_ylim(pdf_max/100,2*pdf_max)

    axes.set_title('Histograms for q0')
    plt.xlabel('Integration result q0 (a.u.)')#, fontsize=14)
    plt.ylabel('Fraction of counts')#, fontsize=14)
    plt.axvline(V_opt, ls='--',
               linewidth=2, color='grey' ,label='SNR={0:.2f}\n $F_a$={1:.5f}\n $F_d$={2:.5f}'.format(SNR, Fa, Fd))
    plt.legend(frameon=False,loc='upper right')
    a=plt.xlim()
    plt.xlim(a[0],a[0]+(a[1]-a[0])*1.2)

    #plt.xlim(-1.3,3.8)
    #plt.savefig('w0_before_{}.pdf'.format(ana.timestamp.replace('/','_')),format='pdf')
    plt.savefig(ana.folder+'\\'+'histogram_w0.'+fig_format,format=fig_format)

    V_th = np.zeros(len(list_qubits))
    V_th[0] =V_opt

    #plot results for q1
    ma.SSRO_Analysis(label =label, auto=True, channels=['w1'], sample_0=0,
                                  sample_1=2, nr_samples=4, rotate=False)
    ana=ma.MeasurementAnalysis(label =label,auto=False)
    ana.load_hdf5data()
    Fa=ana.data_file['Analysis']['SSRO_Fidelity'].attrs['F_a']
    Fd=ana.data_file['Analysis']['SSRO_Fidelity'].attrs['F_d']
    mu0_0=ana.data_file['Analysis']['SSRO_Fidelity'].attrs['mu0_0']
    mu1_0=ana.data_file['Analysis']['SSRO_Fidelity'].attrs['mu1_0']
    mu0_1=ana.data_file['Analysis']['SSRO_Fidelity'].attrs['mu0_1']
    mu1_1=ana.data_file['Analysis']['SSRO_Fidelity'].attrs['mu1_1']

    sigma0_0=ana.data_file['Analysis']['SSRO_Fidelity'].attrs['sigma0_0']
    sigma1_1=ana.data_file['Analysis']['SSRO_Fidelity'].attrs['sigma1_1']
    sigma0_1=ana.data_file['Analysis']['SSRO_Fidelity'].attrs['sigma0_1']
    sigma1_0=ana.data_file['Analysis']['SSRO_Fidelity'].attrs['sigma1_0']
    frac1_0=ana.data_file['Analysis']['SSRO_Fidelity'].attrs['frac1_0']
    frac1_1=ana.data_file['Analysis']['SSRO_Fidelity'].attrs['frac1_1']
    V_opt=ana.data_file['Analysis']['SSRO_Fidelity'].attrs['V_th_a']
    SNR=ana.data_file['Analysis']['SSRO_Fidelity'].attrs['SNR']

    n, bins0, patches = pylab.hist(namespace['w1_data_00'], bins=int(min_len_all/50),
                                          label = 'input state {}'.format(state),histtype='step',
                                          color='red', normed=True, visible=False)
    n, bins1, patches = pylab.hist(namespace['w1_data_10'], bins=int(min_len_all/50),
                                          label = 'input state {}'.format(state),histtype='step',
                                          color='red', normed=True, visible=False)
    pylab.clf()
    fig, axes = plt.subplots(figsize=(8, 5))
    colors=['blue', 'red', 'grey', 'magenta']
    markers=['o','o','o','v']
    for marker, color,state in zip(markers,colors,states):

        n, bins, patches = pylab.hist(namespace['w1_data_{}'.format(state)], bins=int(min_len_all/50),
                                      histtype='step',  normed=True, visible=False)
        pylab.plot(bins[:-1]+0.5*(bins[1]-bins[0]),n, color=color, linestyle='None',marker=marker)

    y0 = (1-frac1_0)*pylab.normpdf(bins0, mu0_0, sigma0_0)+frac1_0*pylab.normpdf(bins0, mu1_0, sigma1_0)
    y1_0 = frac1_0*pylab.normpdf(bins0, mu1_0, sigma1_0)
    y0_0 = (1-frac1_0)*pylab.normpdf(bins0, mu0_0, sigma0_0)

    #building up the histogram fits for on measurements
    y1 = (1-frac1_1)*pylab.normpdf(bins1, mu0_1, sigma0_1)+frac1_1*pylab.normpdf(bins1, mu1_1, sigma1_1)
    y1_1 = frac1_1*pylab.normpdf(bins1, mu1_1, sigma1_1)
    y0_1 = (1-frac1_1)*pylab.normpdf(bins1, mu0_1, sigma0_1)


    pylab.semilogy(bins0, y0, 'b',linewidth=1.5, label='fit |00>')
    # pylab.semilogy(bins0, y1_0, 'b--', linewidth=3.5)
    # pylab.semilogy(bins0, y0_0, 'b--', linewidth=3.5)

    pylab.semilogy(bins1, y1, 'r',linewidth=1.5, label='fit |10>')
    #pylab.semilogy(bins1, y0_1, 'r--', linewidth=3.5)
    #pylab.semilogy(bins1, y1_1, 'r--', linewidth=3.5)
    (pylab.gca()).set_ylim(0.2e-6,1e-3)
    pdf_max=(max(max(y0),max(y1)))
    (pylab.gca()).set_ylim(pdf_max/100,2*pdf_max)

    axes.set_title('Histograms for q1')
    plt.xlabel('Integration result q1 (a.u.)')#, fontsize=14)
    plt.ylabel('Fraction of counts')#, fontsize=14)
    plt.axvline(V_opt, ls='--',
               linewidth=2, color='grey' ,label='SNR={0:.2f}\n $F_a$={1:.5f}\n $F_d$={2:.5f}'.format(SNR, Fa, Fd))
    plt.legend(frameon=False,loc='upper right')
    a=plt.xlim()
    plt.xlim(a[0],a[0]+(a[1]-a[0])*1.2)
    #plt.savefig('w0_before_{}.pdf'.format(ana.timestamp.replace('/','_')),format='pdf')
    plt.savefig(ana.folder+'\\'+'histogram_w1.'+fig_format,format=fig_format)

    V_th[1] =V_opt

    # calculating cross-talk matrix and inverting
    ground_state='00'
    weights=[0,1]
    cal_states=['01','10']
    ground_state='00'
    mu_0_vec = np.zeros(len(weights))
    for j, weight in enumerate(weights):
        mu_0_vec[j]=np.average(eval('w{}_data_{}'.format(weight, ground_state)))

    mu_matrix = np.zeros((len(cal_states), len(weights)))
    for i,state in enumerate(cal_states):
        for j, weight in enumerate(weights):
            mu_matrix[i, j]=np.average(eval('w{}_data_{}'.format(weight, state)))-mu_0_vec[j]

    mu_matrix_inv = inv(mu_matrix)
    V_th_cor = np.dot(mu_matrix_inv,V_th)
    return mu_matrix, V_th, mu_matrix_inv, V_th_cor