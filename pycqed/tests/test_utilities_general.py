from pycqed.utilities.general import SafeFormatter


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
