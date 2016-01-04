# physcon Physical constants
# Note: type physcon.help() (after import physcon)

from math import pi

# define light velocity, because we need it in calculations
cloc =299792458.

# dictionary of physical constants in SI units. Values CODATA 2010 http://www.physics.nist.gov/cuu/Constants/
# each item: [description (string), symbol (string), value (float), sd (float), relat. sd (float),
#             value(sd) unit (string), source (string)]
all={'lightvel':['velocity of light in vacuum','c',cloc,0., 0.,'299 792 458(ex) m/s', 'CODATA 2010'],
     'planck':["Planck's constant",'h',6.62606957e-34,2.9e-41,4.4e-8,'6.626 069 57(29) e-34 J s', 'CODATA 2010'],
     'dirac':["Dirac's constant = h/(2 pi)",'hbar',1.054571726e-34,4.7e-42,4.4e-8,'1.054 571 726(47) e-34 J s', 'CODATA 2010 '],
     'magn-const':['magnetic permeability of vacuum','mu_0',4.e-7*pi,0.,0.,'1.256 637 061... e-6 N A^-2',''],
     'elec-const':['dielectric permittivity of vacuum','eps_0',1.e7/(4*pi*cloc*cloc),0.,0.,'8.854 187 817... e-12 F/m',''],
     'gravit':['Newton constant of gravitation','G',6.67384e-11, 8.0e-15,1.2e-4,'6.673 84(80) e-11 m^3 kg^-1 s^-2','CODATA 2010'],
     'charge-e':['elementary charge','e',1.602176565e-19,3.5e-27,2.2e-8,'1.602 176 565(35) e-19 C','CODATA 2010'],
     'mass-e':['electron mass','m_e',9.10938291e-31,4.0e-38,4.4e-8,'9.109 382 91(40) e-31 kg','CODATA 2010'],
     'mass-e/u':['electron mass in u','m_e_u',5.4857990946e-4,2.2e-13,4.0e-10,'5.485 799 0946(22) u','CODATA 2010'],
     'mass-p':['proton mass','m_p',1.672621777e-27,7.4e-35,4.4e-8,'1.672 621 777(74) e-27 kg','CODATA 2010'],
     'mass-p/u':['proton mass in u','m_p_u',1.007276466812,9.0e-11,8.9e-11,'1.007 276 466 812(90) u','CODATA 2010'],
     'mass-n':['neutron mass','m_n',1.674927351e-27,7.4e-35,4.4e-8,'1.674 927 351(74) e-27 kg','CODATA 2010'],
     'mass-n/u':['neutron mass in u','m_n_u',1.00866491600,4.3e-10,4.2e-10,'1.008 664 916 00(43) u','CODATA 2010'],
     'mass-d':['deuteron mass','m_d',3.34358348e-27,1.5e-34,4.4e-8,'3.343 583 48(15) e-27 kg','CODATA 2010'],
     'mass-d/u':['deuteron mass in u','m_d_u',2.013553212712,7.7e-11,3.8e-11,'2.013 553 212 712(77) u','CODATA 2010'],
     'mass-mu':['muon mass','m_m',1.883531475e-28,9.6e-36,5.1e-8,'1.883 531 475(96) e-28 kg','CODATA 2010'],
     'mass-mu/u':['muon mass in u','m_m_u',0.1134289267,2.9e-9,2.5e-8,'0.113 428 9267(29) u','CODATA 2010'],
     'ratio-me/mp':['electron/proton mass ratio','ratio_memp',5.4461702178e-4,2.2e-13,4.1e-10,'5.446 170 2178(22) e-4','CODATA 2010'],
     'ratio-mp/me':['proton/electron mass ratio','ratio_mpme',1836.15267245,7.5e-7,4.1e-10,'1836.152 672 45(75)','CODATA 2010'],
     'amu':['unified atomic mass unit = 1/12 m(12C)','u',1.660538921e-27,7.3e-35,4.4e-8,'1.660 538 921(73) e-27 kg','CODATA 2010'],
     'avogadro':['Avogadro constant','N_A',6.02214129e23,2.7e16,4.4e-8,'6.022 141 29(27) e23 mol^-1','CODATA 2010'],
     'boltzmann':['Boltzmann constant','k_B',1.3806488e-23,1.3e-29,9.1e-7,'1.380 6488(13) e-23 J/K','CODATA 2010'],
     'gas':['molar gas constant = N_A k_B','R',8.3144621,7.5e-6,9.1e-7,'8.314 4621(75) J mol^-1 K^-1','CODATA 2010'],
     'faraday':['Faraday constant = N_A e','F',96485.3365,2.1e-3,2.2e-8,'96 485.3365(21) C/mol','CODATA 2010'],
     'bohrradius':['Bohr radius = 4 pi eps_0 hbar^2/(m_e e^2)','a_0',5.2917721092e-11,1.7e-20,3.2e-10,'0.529 177 210 92(17) e-10 m','CODATA 2010'],
     'magflux-qu':['magnetic flux quantum = h/(2 e)','Phi_0',2.067833758e-15,4.6e-23,2.2e-8,'2.067 833 758(46) Wb','CODATA 2010'],
     'conduct-qu':['conductance quantum = 2 e^2/h','G_0',7.7480917346e-5,2.5e-14,3.2e-10,'7.748 091 7346(25) e-5 S','CODATA 2010'],
     'josephson':['Josephson constant = 2 e/h','K_J',4.83597870e14, 1.1e7,2.2e-8,'4.835 978 70(11) e14 Hz/V','CODATA 2010'],
     'bohrmagn':['Bohr magneton = e hbar/(2 m_e)','mu_B',9.27400968e-24,2.0e-31,2.2e-8,'9.274 009 68(20) e-24 J/T','CODATA 2010'],
     'nuclmagn':['nuclear magneton = e hbar/(2 m_p)','mu_N',5.05078353e-27,1.1e-34,2.2e-8,'5.050 783 53(11) e-27 J/T','CODATA 2010'],
     'magnmom-e':['electron magnetic moment','mu_e',-9.28476430e-24,2.1e-31,2.2e-8,'-9.284 764 30(21) e-24 J/T','CODATA 2010'],
     'magnmom-p':['proton magnetic moment','mu_p',1.410606743e-26,3.3e-33,2.4e-8,'1.410 606 743(33) e-26 J/T','CODATA 2010'],
     'gfactor-e':['electron g-factor','g_e',-2.00231930436153,5.3e-13,2.6e-13,'-2.002 319 304 361 53(53)','CODATA 2010'],
     'gfactor-p':['proton g-factor','g_p',5.585694713, 4.6e-8,8.2e-9,'5.585 694 713(46)','CODATA 2010'],
     'alpha':['fine-structure constant = e^2/(4 pi eps_0 hbar c)','alpha',7.2973525698e-3,2.4e-12,3.2e-10,'7.297 352 5698(24) e-3','CODATA 2010'],
     'alpha-1':['inverse fine-structure constant = 4 pi eps_0 hbar c/e^2','',137.035999074,4.4e-8,3.2e-10,'137.035 999 074(44)','CODATA 2010'],
     'gyromagratio-p':['proton gyromagnetic ratio','gamma_p',2.675222005e8,6.3,2.4e-8,'2.675 222 005(63) e8 s^-1 T^-1','CODATA 2010'],
     'magres-p':['magnetic resonance frequency proton = gamma_p/(2*pi)','',4.25774806e7,1.0,2.4e-8,'42.577 4806(10) MHz/T','CODATA 2010'],
     'rydberg':['Rydberg constant = alpha^2 m_e c/(2 h)','R_infty',10973731.568539,5.5e-5,5.0e-12,'10 973 731.568 539(55) m^-1','CODATA 2010'],
     'stefan-boltzm':['Stefan-Boltzmann constant = pi^2 k^4/(60 hbar^3 c^2)','sigma',5.670373e-8,2.1e-13,3.6e-6,'5.670 373(21) e-8 W m^-2 K^-4','CODATA 2010']}


# many common values are also available as global constants:
global alpha,a_0,c,e,eps_0,F,G,g_e,g_p,gamma_p,h,hbar,k_B
global m_d,m_e,m_n,m_p,mu_B,mu_e,mu_N,mu_p,mu_0,N_A,R,sigma,u
alpha = all['alpha'][2]
a_0 =  all['bohrradius'][2]
c = cloc
e =  all['charge-e'][2]
eps_0 =  all['elec-const'][2]
F =  all['faraday'][2]
G =  all['gravit'][2]
g_e =  all['gfactor-e'][2]
g_p =  all['gfactor-p'][2]
gamma_p =  all['gyromagratio-p'][2]
h = all['planck'][2]
hbar = all['dirac'][2]
k_B =  all['boltzmann'][2]
m_d =  all['mass-d'][2]
m_e =  all['mass-e'][2]
m_n =  all['mass-n'][2]
m_p =  all['mass-p'][2]
mu_B =  all['bohrmagn'][2]
mu_e =  all['magnmom-e'][2]
mu_N =  all['nuclmagn'][2]
mu_p =  all['magnmom-p'][2]
mu_0 =  all['magn-const'][2]
N_A =  all['avogadro'][2]
R =  all['gas'][2]
sigma =  all['stefan-boltzm'][2]
u =  all['amu'][2]


def help():
    print('Available functions:')
    print('[note: key must be a string, within quotes!]' )
    print('  value(key) returns value (float)')
    print('  sd(key)    returns standard deviation (float)')
    print('  relsd(key) returns relative standard deviation (float)')
    print('  descr(key) prints description with units\n')
    print('Available global variables:')
    print('  alpha, a_0, c, e, eps_0, F, G, g_e, g_p, gamma_p, h, hbar, k_B')
    print('  m_d, m_e, m_n, m_p, mu_B, mu_e, mu_N, mu_p, mu_0, N_A, R, sigma, u\n')
    allkeys=sorted(all.keys())
    print('Available keys:')
    print(allkeys)

def value(key):
    return all[key][2]

def sd(key):
    return all[key][3]

def relsd(key):
    return all[key][4]

def descr(key):
    print(('Description of ',key,':'))
    print(('  Name:               ',all[key][0]))
    print(('  Symbol (if avail.): ',all[key][1]))
    print(('  Value:              ',all[key][2]))
    print(('  Standard deviation: ',all[key][3]))
    print(('  Relative stdev:     ',all[key][4]))
    print(('  value(sd) unit:     ',all[key][5]))
    print(('  Source:             ',all[key][6],'\n'))






