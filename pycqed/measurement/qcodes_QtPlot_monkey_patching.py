"""
Last update: 2020-02-05
Initial author: Victor Negirneac
This is a patch for qcodes that is inteded to do 2 things:
- Allow specifying a colormap for plotmon_2D
- Set a specific range in the plotmon_2D
It is general enough but the main goal was to have a circular colormap for
phase plotting with a fixed range 0.0 - 360.0 deg.

Tested and developed with qcodes 0.10.0

WARNING: It is very easy to break this script just by changing the order
of certain lines!

===============================================================================
If you ever have to change anything in here
===============================================================================

This code does "monkey patching". Not the ideal way to code and integrate other
packages but it is the fastest way to achive exactly what you want without
having to fork qcodes or do some other developments that require maintnence.
Even though it is moneky patching I put effort into making sure it is will be
working even if the qcodes changes a bit as long as key functions keep their
names and general behaviour.
The ideal way would be to make a pull request in qcodes and wait...
The naive way would be to read the source code and replace/insert code, or
worse, copy paste the entire module/class/function...
What is done here is in between those two extremes. If you have time still go
fot the ideal scenario so that more people have access to this features.

-------------------------------------------------------------------------------
Some useful notes and lines of code if you ever have to debug this:

With this you can go through all nodes and have an idea of how it is structured
`for node in ast.walk(parsedSourceCode)`

Most nodes have some attributes that can help when looking for a specific one.
Some nodes don't have any useful attributes. Use this to dig deeper
`ast.dump(node)`

Using ast.walk will go through all nodes and not necessarily follow the tree's
branches. You can somewhat "navigate" through the tree with:
`ast.parse(sourceCode).body[0].body[2]....`

You can use the following to print a modified tree:
from astunparse import Unparser
import io
buf = io.StringIO()
Unparser(parsedQtPlotSource, buf)
buf.seek(0)
print(buf.read())

You can do more fancy things using the `ast` module using ast.NodeTransformer
and ast.NodeVisitor

=== Usefull refs: ===
It all started here:
https://medium.com/@chipiga86/python-monkey-patching-like-a-boss-87d7ddb8098e
An example of `ast` in action
https://stackoverflow.com/questions/46388130/insert-a-node-into-an-abstract-syntax-tree

=== NBs: ===
- Don't import QtPlot before this file
- Namespaces can screw you up easily
- Order of imports matters!
- Reloading QtPlot will likely break this
- Don't import anything else from qcodes in this file, will likely break this

Happy monkey patching!
"""
import ast
import inspect

# Do not import anything else from qcodes before this, it breaks this code

# Below: patch the QtPlot method to allow for setting a fixed color scale range

import qcodes
from qcodes.plots.pyqtgraph import QtPlot


def dummy_func(hist, **kwargs):
    """
    This extra line allows to set a fixed colormap range when drawing the
    plot 2D for the first time, when inserted in the right place in the original
    code.
    """
    hist.setLevels(*(kwargs["zrange"] if "zrange" in kwargs else hist.getLevels()))
    return None


# Get source code
str_dummy_func = inspect.getsource(dummy_func)
# Parse code into a tree that allows proper modifications
parsed_dummy_func = ast.parse(str_dummy_func)

# Get the tree version of the code inside the dummy_func to be inserted
setLevelsNode = parsed_dummy_func.body[0].body[-2]  # Grab the line before the return

# Get source code of QtPlot and parse it into a tree
QtPlotSource = inspect.getsource(QtPlot)
parsedQtPlotSource = ast.parse(QtPlotSource)

# Search for the Class node
QtPlotClass = None
for node in ast.walk(parsedQtPlotSource):
    if isinstance(node, ast.ClassDef):
        if node.name == "QtPlot":
            QtPlotClass = node
            break

# Search for the method node
_draw_imageFunc = None
for node in ast.walk(QtPlotClass):
    if isinstance(node, ast.FunctionDef):
        if node.name == "_draw_image":
            _draw_imageFunc = node
            break

# Insert the new node and fix the tree to accomodate the changes
_draw_imageFunc.body.insert(-1, setLevelsNode)
ast.fix_missing_locations(parsedQtPlotSource)

# Compile and execute the code in the namespace of the "pyqtgraph" module
# such that on next import of QtPlot the patched version will be used
co = compile(parsedQtPlotSource, "<string>", "exec")
exec(co, qcodes.plots.pyqtgraph.__dict__)


# Patch the color scales

# The line below is the naive way of doing it but will not work consistenly
# qcodes.plots.colors.colorscales = qcodes_QtPlot_colors_override.colorscales

from pycqed.measurement import qcodes_QtPlot_colors_override as qc_cols_override

str_colorscales = "colorscales = " + repr(qc_cols_override.colorscales)
str_colorscales_raw = "colorscales_raw = " + repr(qc_cols_override.colorscales_raw)

parsed_colorscales = ast.parse(str_colorscales)
parsed_colorscales_raw = ast.parse(str_colorscales_raw)

co = compile(parsed_colorscales, "<string>", "exec")
exec(co, qcodes.plots.colors.__dict__)
co = compile(parsed_colorscales_raw, "<string>", "exec")
exec(co, qcodes.plots.colors.__dict__)


# On some systems this is also required probably because the colors get imported
# there as well and, depending on the python version, the reference doesn't
# change everywhere
co = compile(parsed_colorscales, "<string>", "exec")
exec(co, qcodes.plots.pyqtgraph.__dict__)
co = compile(parsed_colorscales_raw, "<string>", "exec")
exec(co, qcodes.plots.pyqtgraph.__dict__)