import spips
import delta_cep_data # contains all the observations
import os

# -- load data from aux file
obs = delta_cep_data.data

# -- this is the model using splines
# Parameters description:
# 'DIAMAVG':      1.45573 , # Average angular diameter (mas)
# 'E(B-V)':       0.09109 , # reddenning
# 'EXCESS EXP':   0.4 , # exponential law for IR Excess
# 'EXCESS SLOPE': 0.06278 , # slope for IR excess
# 'EXCESS WL0':   1.2 , # starting WL, in um, for IR Excess
# 'METAL':        0.06 , # metalicity / solar
# 'MJD0':         48304.7362421 , # 0-phase
# 'P-FACTOR':     1.2687 , # projection factor
# 'PERIOD':       5.36626201 , # period in days
# 'PERIOD1':      -0.0086 , # period change, in s/yrs
# 'TEFF DVAL1':   -219.91 , # offset of split point to VAL0
# 'TEFF DVAL2':   163.7 , # offset of split point to VAL0
# 'TEFF DVAL3':   709.69 , # offset of split point to VAL0
# 'TEFF DVAL4':   571.61 , # offset of split point to VAL0
# 'TEFF PHI0':    0.8844 , # ref phase for spline comb
# 'TEFF POW':     1.5856 re, # spline comb spread (1 is regular)
# 'TEFF VAL0':    5704.094 , # first spline node value
# 'VRAD DVAL1':   14.002 , #
# 'VRAD DVAL2':   22.084 , #
# 'VRAD DVAL3':   18.935 , #
# 'VRAD DVAL4':   -1.221 , #
# 'VRAD DVAL5':   -13.712 , #
# 'VRAD PHI0':    0.84471 , #
# 'VRAD POW':     1.9098 , #
# 'VRAD VAL0':    -21.371 , #
# 'd_kpc':        0.274 , # distance in kilo-pc

p_splines = {'DIAMAVG':      1.45616 , # +/- 0.00105
            'E(B-V)':       0.0908 , # +/- 0.00192
            'EXCESS EXP':   0.4 ,
            'EXCESS SLOPE': 0.06187 , # +/- 0.0015
            'EXCESS WL0':   1.2 ,
            'METAL':        0.06 ,
            'MJD0':         48304.7362421 ,
            'P-FACTOR':     1.2712 , # +/- 0.0177
            'PERIOD':       5.36627863 , # +/- 5.5e-06
            'PERIOD1':      -0.0851 , # +/- 0.0293
            'TEFF DVAL1':   -221.253 , # +/- 4.327
            'TEFF DVAL2':   167.46 , # +/- 18.02
            'TEFF DVAL3':   711.8 , # +/- 13.76
            'TEFF DVAL4':   577.53 , # +/- 12.55
            'TEFF PHI0':    0.88491 , # +/- 0.00161
            'TEFF POW':     1.5952 , # +/- 0.0372
            'TEFF VAL0':    5702.675 , # +/- 6.041
            'VRAD DVAL1':   13.882 , # +/- 0.275
            'VRAD DVAL2':   22.02 , # +/- 0.162
            'VRAD DVAL3':   19.021 , # +/- 0.406
            'VRAD DVAL4':   -1.405 , # +/- 0.524
            'VRAD DVAL5':   -13.67 , # +/- 0.321
            'VRAD PHI0':    0.84753 , # +/- 0.00197
            'VRAD POW':     1.8918 , # +/- 0.0478
            'VRAD VAL0':    -21.379 , # +/- 0.17
            'd_kpc':        0.274 ,
            }


# Alternatively, the TEFF and VRAD profiles can be described using FOURIER parameters:
# 'TEFF A0':      5887.886 , # average Teff
# 'TEFF A1':      469.915 , # amplitude of first harmonic
# 'TEFF PHI1':    -0.3581 , # phase of first harmonic
# 'TEFF PHI2':    -0.2403 , # etc.
# 'TEFF PHI3':    0.3564 ,
# 'TEFF PHI4':    0.7853 ,
# 'TEFF PHI5':    1.71 ,
# 'TEFF R2':      0.39556 , # amp1/amp2
# 'TEFF R3':      0.15563 , # etc.
# 'TEFF R4':      0.06821 ,
# 'TEFF R5':      0.02028 ,

p_fourier = {'DIAMAVG':      1.45621 , # +/- 0.00103
        'E(B-V)':       0.09123 , # +/- 0.00189
        'EXCESS EXP':   0.4 ,
        'EXCESS SLOPE': 0.06187 , # +/- 0.00148
        'EXCESS WL0':   1.2 ,
        'METAL':        0.06 ,
        'MJD0':         48304.7362421 ,
        'P-FACTOR':     1.2743 , # +/- 0.0188
        'PERIOD':       5.36627457 , # +/- 5.15e-06
        'PERIOD1':      -0.0601 , # +/- 0.024
        'TEFF A0':      5887.088 , # +/- 5.621
        'TEFF A1':      470.031 , # +/- 3.244
        'TEFF PHI1':    -0.35815 , # +/- 0.00513
        'TEFF PHI2':    -0.2397 , # +/- 0.0145
        'TEFF PHI3':    0.357 , # +/- 0.0317
        'TEFF PHI4':    0.7916 , # +/- 0.0702
        'TEFF PHI5':    1.697 , # +/- 0.22
        'TEFF R2':      0.39535 , # +/- 0.00523
        'TEFF R3':      0.15611 , # +/- 0.00481
        'TEFF R4':      0.06832 , # +/- 0.00461
        'TEFF R5':      0.02047 , # +/- 0.00438
        'VRAD A0':      -18.4296 , # +/- 0.0707
        'VRAD A1':      15.483 , # +/- 0.115
        'VRAD PHI1':    2.18429 , # +/- 0.00697
        'VRAD PHI2':    -0.6246 , # +/- 0.0165
        'VRAD PHI3':    4.9257 , # +/- 0.0353
        'VRAD PHI4':    4.2857 , # +/- 0.0468
        'VRAD PHI5':    3.4539 , # +/- 0.0829
        'VRAD PHI6':    2.802 , # +/- 0.147
        'VRAD PHI7':    1.729 , # +/- 0.272
        'VRAD PHI8':    1.164 , # +/- 0.525
        'VRAD R2':      0.41709 , # +/- 0.006
        'VRAD R3':      0.21503 , # +/- 0.00474
        'VRAD R4':      0.1206 , # +/- 0.0065
        'VRAD R5':      0.05849 , # +/- 0.00673
        'VRAD R6':      0.03872 , # +/- 0.00503
        'VRAD R7':      0.01959 , # +/- 0.00465
        'VRAD R8':      0.00997 , # +/- 0.00538
        'd_kpc':        0.274 ,
        }


def fit(p=None):
    if p is None:
        p = p_splines
    # - list parameters which we do not wish to fit
    doNotFit= ['MJD0','METAL', 'd_kpc', 'EXCESS WL0', 'EXCESS EXP']
    fitOnly = None
    # - alternatively, we can list the only parameters we wich to fit
    # fitOnly = filter(lambda x: x.startswith('TEFF ') or x.startswith('VRAD '), p.keys())
    # obs = filter(lambda o: 'vrad' in o[1] or 'teff' in o[1], obs)
    #fitOnly=['P-FACTOR', 'DIAMAVG']
    f = spips.fit(obs, p, doNotFit=doNotFit, fitOnly=fitOnly,
            normalizeErrors='techniques', # 'observables' is the alternative
            ftol=5e-4, # stopping tolerance on chi2
            epsfcn=1e-8, # by how much parameters will vary
            maxfev=500, # maximum number of iterations
            maxCores=None, # max number of CPU cores, None will use all available
            starName='delta Cep',
            follow=['P-FACTOR'], # list here parameters you want to see during fit
            exportFits=True,
            )
    spips.dispCor(f) # show the correlation matrix between parameters
    return

def show(p=None):
    if p is None:
        p = p_splines
    Y = spips.model(obs, p, starName='delta Cep', verbose=True, plot=True)

def fitsDemo(mode='export', p=None):
    if p is None:
        p = p_splines

    if mode=='export':
        Y = spips.model(obs, p, starName='delta Cep',
                        exportFits=True, verbose=True)
    elif mode=='import':
        filename = 'delta_cep.fits'
        if os.path.exists(filename):
            _p, _obs = spips.importFits(filename)
            Y = spips.model(_obs, _p, starName='delta Cep', verbose=True,
                            plot=True)
        else:
            print 'ERROR:', filename, 'does not exist!'
    else:
        print "use: fitsDemo(mode='export')"
        print "  or fitsDemo(mode='import')"
