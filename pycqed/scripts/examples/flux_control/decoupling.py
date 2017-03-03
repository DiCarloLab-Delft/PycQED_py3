import numpy as np
import logging
"""
Flux control scan script.
Inputs:
        N               number of qubits
        f_center_vec    N-dim vector with frequencies where the qubits respond linearly.
        dac_center      N-dim vector with the biasing corresponding to f_center_vec.
        f_ranges        N-dim list with the frequency range for the resonator scans.
        dac_span_low    N-dim vector with the low span for the dac sweep.
        dac_span_high   N-dim vector with the high span for the dac sweep.
        use_min         N-dim vector with the flag for resonator fitting.
        f_span_low      N-dim vector with the low span for the qubit scans.
        f_span_high     N-dim vector with the high span for the qubit scans.
        M               number of points per qubit panel.
        pulsed          N-dim vector with the flag for pulsed(or not) spectroscopy.
        qubit_list      N-dim vector with the qubit objects.
        IVVI            Instrument controlling the IVVI rack.

        Order of inputs must match between all vectors.


"""


def flux_decoupling_scan(N, f_center_vec, dac_center, f_ranges, dac_span_low,
                         dac_span_high, use_min, f_span_low, f_span_high,
                         M, pulsed, qubit_list, IVVI):

    assert(N == len(qubit_list))

    dac_range = np.zeros(N)
    fq_range = np.zeros(N)
    dac_channel_list = np.zeros(N)

    for q in range(N):
        dac_range[q] = np.linspace(
            dac_span_low[q], dac_span_high[q], M) + dac_center[q]
        fq_range[q] = np.linspace(
            f_span_low[q], f_span_high[q], M) + f_center_vec[q]

    for i, q in enumerate(qubit_list):
        IVVI.set('dac%d' % q.dac_channel(), dac_center[i])
        dac_channel_list[i] = q.dac_channel()

    # goes sweeping dac by dac
    for dac_idx in dac_channel_list:
        # starts by rebiasing the positions to the linear center
        for i, q in enumerate(qubit_list):
            IVVI.set('dac%d' % q.dac_channel(), dac_center[i])
        # sweeps the dac channel
        for d in dac_range[dac_idx]:
            # sets the biasing
            IVVI.set('dac%d' % q.dac_channel(), dac_center[i])
            for i, q in enumerate(qubit_list):
                # scans this qubit
                q.find_resonator_frequency(
                    freqs=f_ranges[i], use_min=use_min[i])
                try:
                    q.find_frequency(freqs=fq_range[i], pulsed=pulsed[i])
                except Exception:
                    logging.warning(
                        'Qubit fit failed at Dac_%d=%.2f' % (dac_idx, d))
