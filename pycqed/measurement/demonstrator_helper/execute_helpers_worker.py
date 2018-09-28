
from pycqed.measurement import measurement_control
import os
import pycqed as pq
from qcodes.instrument.base import Instrument
import qcodes as qc

import pycqed.measurement.openql_experiments.generate_qi_cfg as gcfg_qi


from QIProductionWorkers.CCLightWorker import CCLightWorker

try:
    MC_demo = measurement_control.MeasurementControl(
        'QInfinity_MC', live_plot_enabled=True, verbose=True)

    datadir = os.path.abspath(os.path.join(
        os.path.dirname(pq.__file__), os.pardir, 'demonstrator_execute_data'))
    MC_demo.datadir(datadir)
    st = qc.station.Station()
    MC_demo.station = st
    st.add_component(MC_demo)

except KeyError:
    MC_demo = Instrument.find_instrument('QInfinity_MC')



config_fn ='D:\\GitHubRepos\\PycQED_py3\\pycqed\\measurement\\openql_experiments\\output\\cfg_CCL_QI.json'
gcfg_qi.generate_config(filename=config_fn,
                mw_pulse_duration=20,
                ro_duration=5000,
                flux_pulse_duration=40,
                init_duration=300000)


# worker = CCLightWorker(name='Starmon', nr_qubits=2,
#                        is_simulator=False, openql_config_path=config_fn)

worker = CCLightWorker(name='Octoqubit', nr_qubits=2,
                       is_simulator=False, openql_config_path=config_fn)