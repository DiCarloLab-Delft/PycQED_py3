kernel_list = ['corr0_2017-03-29_2_195314.txt',
               'corr0_2017-03-30_0_110407.txt',
               'corr0_2017-03-30_1_111929.txt',
               # 'corr0_2017-03-30_4_115338.txt',
               'corr0_2017-03-30_5_132742.txt']

               # 'corr0_2017-03-30_5_132742.txt',

               # 'corr0_2017-03-29_2_195314.txt']



import pycqed.instrument_drivers.meta_instrument.qubit_objects.Tektronix_driven_transmon as qbt
QRt = qbt.Tektronix_driven_transmon('QRt')
station.add_component(QRt)
gen.load_settings_ont


fids=[]
freqs = np.linspace(7.19e9, 7.192e9, 21)
for f in freqs:
    QR.f_RO(f)
    a = QR.measure_ssro(nr_shots=4000)
    fids.append(a.F_a)

plt.plot(freqs, fids)


fids=[]
freqs = np.linspace(7.084e9, 7.088e9, 11)
for f in freqs:
    QL.f_RO(f)
    a = QL.measure_ssro(nr_shots=4000)
    fids.append(a.F_a)

plt.plot(freqs, fids)
plt.show()