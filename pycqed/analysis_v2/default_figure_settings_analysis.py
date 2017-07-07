import matplotlib
def apply_default_figure_settings():
    my_font = {'sans-serif':'Arial'}
    # my_font = {'family':'sans-serif',
    #            'sans-serif':'DejaVu Sans',
    #            'weight':'normal'}
    my_font.update({'size':16})
    matplotlib.rc('font', **my_font)

# font = {'family'    :   'sans-serif',
#         'sans-serif':   ['Computer Modern Sans serif'],
#         'size'      :   11,
#         'weight'    : 'normal'
#        }

    my_mathtext = dict(fontset='dejavusans')
    matplotlib.rc('mathtext', **my_mathtext)

    my_axes = {'formatter.useoffset':False}
    matplotlib.rc('axes',**my_axes)

    my_xtick = {'major.pad':8,
                'minor.pad':8}
    matplotlib.rc('xtick',**my_xtick)

    # matplotlib.rc('text', usetex=True)

    # my_latex = {'preamble':[
    #     # r'\usepackage{tgheros}',    # helvetica font
    #     r'\usepackage{sansmath}',   # math-font matching  helvetica
    #     r'\sansmath'                # actually tell tex to use it!
    #     r'\usepackage{siunitx}',    # micro symbols
    #     r'\sisetup{detect-all}',    # force siunitx to use the fonts
    # ]}
    # matplotlib.rc('text.latex',**my_latex)

    # ax.ticklabel_format(useOffset=False)

    # figure = {'subplot.bottom': 0.1,
    #           'subplot.left': 0.15}
    # figure.subplot.hspace: 0.2
    # figure.subplot.left: 0.125
    # figure.subplot.right: 0.9
    # figure.subplot.top: 0.9
    # figure.subplot.wspace: 0.2}
    # matplotlib.rc('figure',**figure)

    # legend = {'frameon':False,
    #          'fontsize':'small'}
    legend = {'numpoints':1,
             'fontsize':14,
              'loc':'best',
              'framealpha':0.5}
    matplotlib.rc('legend', **legend)

def apply_figure_settings(settings_name, settings_dict):
    matplotlib.rc(settings_name, **settings_dict)
