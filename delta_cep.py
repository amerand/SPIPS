import os, sys
if not any(['SPIPS' in x for x in sys.path]):
    _dirs = ['/Users/amerand/Codes/Python/SPIPS',]
    for d in _dirs:
        if os.path.isdir(d):
            sys.path.append(d)

import spips
import delta_cep_data # contains all the observations

runAny = False # do not change!

obs = delta_cep_data.data

# -- this parametrization minimize the correlations
aBW = {'DIAMAVG':      1.45573 , # +/- 0.00105
    'E(B-V)':       0.09109 , # +/- 0.00192
    'EXCESS EXP':   0.4 ,
    'EXCESS SLOPE': 0.06278 , # +/- 0.00149
    'EXCESS WL0':   1.2 ,
    'METAL':        0.06 ,
    'MJD0':         48304.7362421 ,
    'P-FACTOR':     1.2687 , # +/- 0.0173
    'PERIOD':       5.36626201 ,
    'PERIOD1':      -0.0086 ,
    'TEFF DVAL1':   -219.91 , # +/- 4.272
    'TEFF DVAL2':   163.7 , # +/- 17.87
    'TEFF DVAL3':   709.69 , # +/- 13.56
    'TEFF DVAL4':   571.61 , # +/- 12.33
    'TEFF PHI0':    0.8844 , # +/- 0.00157
    'TEFF POW':     1.5856 , # +/- 0.0369
    'TEFF VAL0':    5704.094 , # +/- 6.013
    'VRAD DVAL1':   14.002 , # +/- 0.282
    'VRAD DVAL2':   22.084 , # +/- 0.158
    'VRAD DVAL3':   18.935 , # +/- 0.413
    'VRAD DVAL4':   -1.221 , # +/- 0.526
    'VRAD DVAL5':   -13.712 , # +/- 0.321
    'VRAD PHI0':    0.84471 , # +/- 0.00171
    'VRAD POW':     1.9098 , # +/- 0.0492
    'VRAD VAL0':    -21.371 , # +/- 0.172
    'd_kpc':        0.274 ,
    }

# ========================================
perform_fit = True; N_monte_carlo=2;
perform_fit = True; N_monte_carlo=0;
perform_fit = False; N_monte_carlo=0;
# ========================================

try:
    # this try/except avoids to run bw2.model at import since it will fails:
    # it contains calls to 'multiprocessing.Pool'. When reloaded, the script
    # will run.
    if loaded:
        runAny = True
except:
    # will work only at reload, because now 'loaded' exist
    loaded = True

if runAny:
    if perform_fit:
        doNotFit=['MJD0','METAL', 'd_kpc',
                  'PERIOD','PERIOD1',
                  'EXCESS WL0', 'EXCESS EXP',
                  #'VPULS POW', 'TEFF POW',
                  #'P-FACTOR',
                  #'E(B-V)',
                  #'K EXCESS', 'H EXCESS',
                 ]
        fitOnly = None

        fit = spips.fit(obs, aBW, doNotFit=doNotFit, fitOnly=fitOnly,
                N_monte_carlo=N_monte_carlo, monte_carlo_method='randomize',
                normalizeErrors='techniques', ftol=5e-4, epsfcn=1e-9,
                starName='delta Cep', maxCores=4, maxfev=100,
                follow=['P-FACTOR'])
        spips.dispCor(fit)

    else:
        # -- display best fitted parameters (no fit)
        Y = spips.model(obs, aBW, plot=10, title='delta Cep', verbose=True, exportFits=False)
