from lmfit.parameter import Parameter
from pycqed.utilities.general import SafeFormatter, format_value_string


base_str = 'my_test_values_{:.2f}_{:.3f}'
fmt = SafeFormatter()


def test_safe_formatter():

    fmt_string = fmt.format(base_str, 4, 4.32497)
    assert fmt_string == 'my_test_values_4.00_4.325'


def test_safe_formatter_missing():
    fmt_string = fmt.format(base_str, 4, None)
    assert fmt_string == 'my_test_values_4.00_~~'
    fmt_custom = SafeFormatter(missing='?')
    fmt_string = fmt_custom.format(base_str, 4, None)
    assert fmt_string == 'my_test_values_4.00_?'


def test_safe_formatter_bad_format():
    fmt_string = fmt.format(base_str, 4, 'myvalue')
    assert fmt_string == 'my_test_values_4.00_!!'

    fmt_custom = SafeFormatter(bad_fmt='!')
    fmt_string = fmt_custom.format(base_str, 4, 'myvalue')
    assert fmt_string == 'my_test_values_4.00_!'


def test_save_formatter_named_args():
    plot_title = fmt.format('{measurement}\n{timestamp}',
                timestamp='190101_001122',
                measurement='test')
    assert plot_title == 'test\n190101_001122'

def test_format_value_string():
    tau = Parameter('tau', value=5.123456)
    formatted_string = format_value_string('tau', tau)
    assert formatted_string == 'tau: 5.1235$\pm$NaN '

    tau.stderr = 0.03
    formatted_string = format_value_string('tau', tau)
    assert formatted_string == 'tau: 5.1235$\pm$0.0300 '
    tau.stderr = 0.03

def test_format_value_string_unit_aware():
    tau = Parameter('tau', value=5.123456e-6)
    formatted_string = format_value_string('tau', tau, unit='s')
    assert formatted_string == 'tau: 5.1235$\pm$NaN μs'

    tau.stderr = 0.03e-6
    formatted_string = format_value_string('tau', tau, unit='s')
    assert formatted_string == 'tau: 5.1235$\pm$0.0300 μs'
    tau.stderr = 0.03

