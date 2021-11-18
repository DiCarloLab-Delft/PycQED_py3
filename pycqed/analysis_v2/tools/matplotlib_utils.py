"""
Contains matplotlib utilities
"""
import matplotlib.pyplot as plt
import re
import logging

log = logging.getLogger(__name__)


def latex_friendly_str(text: str, escape_chars: str = "&%$#_{}~^\\<>"):
    """
    Checks if matplotlib is using latex and escapes sensitive characters
    Useful when activating latex rendering and all the analysis break
    due to timestamps format...

    Example:
        fig, ax = plt.subplots()
        fig.suptitle(latex_friendly_str("1234_2134"))
    """

    return text if not plt.rcParams["text.usetex"] else tex_escape(text, escape_chars)


def tex_escape(text, escape_chars: str = "&%$#_{}~^\\<>"):
    """
        :param text: a plain text message
        :return: the message escaped to appear correctly in LaTeX

        From: https://stackoverflow.com/a/25875504/9047715
    """
    conv = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\^{}",
        "\\": r"\textbackslash{}",
        "<": r"\textless{}",
        ">": r"\textgreater{}",
    }
    conv = {key: val for key, val in conv.items() if key in escape_chars}
    regex = re.compile(
        "|".join(
            re.escape(str(key))
            for key in sorted(conv.keys(), key=lambda item: -len(item))
        )
    )
    return regex.sub(lambda match: conv[match.group()], text)
