import spips
import delta_cep_data # contains all the observations
import os

# -- load data from aux file
obs = delta_cep_data.data

# -- this is the model using splines
p_splines = {'DIAMAVG':      1.45573 , # Average angular diameter (mas)
            'E(B-V)':       0.09109 , # reddenning
            'EXCESS EXP':   0.4 , # exponential law for IR Excess
            'EXCESS SLOPE': 0.06278 , # slope for IR excess
            'EXCESS WL0':   1.2 , # starting WL, in um, for IR Excess
            'METAL':        0.06 , # metalicity / solar
            'MJD0':         48304.7362421 , # 0-phase
            'P-FACTOR':     1.2687 , # projection factor
            'PERIOD':       5.36626201 , # period in days
            'PERIOD1':      -0.0086 , # period change, in s/yrs
            'TEFF DVAL1':   -219.91 , # offset of split point to VAL0
            'TEFF DVAL2':   163.7 , # offset of split point to VAL0
            'TEFF DVAL3':   709.69 , # offset of split point to VAL0
            'TEFF DVAL4':   571.61 , # offset of split point to VAL0
            'TEFF PHI0':    0.8844 , # ref phase for spline comb
            'TEFF POW':     1.5856 , # spline comb spread (1 is regular)
            'TEFF VAL0':    5704.094 , # first spline node value
            'VRAD DVAL1':   14.002 , #
            'VRAD DVAL2':   22.084 , #
            'VRAD DVAL3':   18.935 , #
            'VRAD DVAL4':   -1.221 , #
            'VRAD DVAL5':   -13.712 , #
            'VRAD PHI0':    0.84471 , #
            'VRAD POW':     1.9098 , #
            'VRAD VAL0':    -21.371 , #
            'd_kpc':        0.274 , # distance in kilo-pc
            }

# Alternatively, the TEFF and VRAD profiles can be described using FOURIER parameters:
p_fourier = {'DIAMAVG':      1.45593 ,
            'E(B-V)':       0.09146 ,
            'EXCESS EXP':   0.4 ,
            'EXCESS SLOPE': 0.06196 ,
            'EXCESS WL0':   1.2 ,
            'METAL':        0.06 ,
            'MJD0':         48304.7362421 ,
            'P-FACTOR':     1.2749 ,
            'PERIOD':       5.36627437 ,
            'PERIOD1':      -0.0614 ,
            'TEFF A0':      5887.886 , # average Teff
            'TEFF A1':      469.915 , # amplitude of first harmonic
            'TEFF PHI1':    -0.3581 , # phase of first harmonic
            'TEFF PHI2':    -0.2403 , # etc.
            'TEFF PHI3':    0.3564 ,
            'TEFF PHI4':    0.7853 ,
            'TEFF PHI5':    1.71 ,
            'TEFF R2':      0.39556 , # amp1/amp2
            'TEFF R3':      0.15563 , # etc.
            'TEFF R4':      0.06821 ,
            'TEFF R5':      0.02028 ,
            'VRAD A0':      -18.4305 ,
            'VRAD A1':      15.481 ,
            'VRAD PHI1':    2.18445 ,
            'VRAD PHI2':    -0.6244 ,
            'VRAD PHI3':    4.9265 ,
            'VRAD PHI4':    4.2866 ,
            'VRAD PHI5':    3.4535 ,
            'VRAD PHI6':    2.801 ,
            'VRAD PHI7':    1.726 ,
            'VRAD PHI8':    1.165 ,
            'VRAD R2':      0.41703 ,
            'VRAD R3':      0.21495 ,
            'VRAD R4':      0.12066 ,
            'VRAD R5':      0.05859 ,
            'VRAD R6':      0.0387 ,
            'VRAD R7':      0.01955 ,
            'VRAD R8':      0.00993 ,
            'd_kpc':        0.274 ,
        }

def fit(p=None):
    if p is None:
        p = p_splines
    # - list parameters which we do not wish to fit
    doNotFit= ['MJD0','METAL', 'd_kpc','EXCESS WL0', 'EXCESS EXP']
    fitOnly = None
    # - alternatively, we can list the only parameters we wich to fit
    # fitOnly = filter(lambda x: x.startswith('TEFF ') or x.startswith('VRAD '),
    #                 first_guess.keys())
    # obs = filter(lambda o: 'vrad' in o[1] or 'teff' in o[1], obs)

    fit = spips.fit(obs, p, doNotFit=doNotFit, fitOnly=fitOnly,
            normalizeErrors='techniques', # 'observables' is the alternative
            ftol=5e-4, epsfcn=1e-9, maxfev=500, # fit parameters
            starName='delta Cep', maxCores=4,
            follow=['P-FACTOR'] # list here parameters you want to see during fit
            )
    spips.dispCor(fit) # show the correlation matrix between parameters

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
