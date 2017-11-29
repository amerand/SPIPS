"""
-- Parallax of pulsation method --

This is the second version, using dictionnary based model's parameters,
instead of a list (for pdfit.py). The function 'model' of this package is the
actual modeling function. 'modelM' is the parallelized version of 'model'.

"""
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)

import time
import os, sys
import getpass
import multiprocessing
from multiprocessing import Pool
import cPickle
import numpy as np
from scipy.interpolate import interp1d # spline function
from scipy.interpolate import RectBivariateSpline
from scipy import special # for gamma and bessel functions
import matplotlib
from matplotlib import pyplot as plt
import collections
import astropy.io.fits as pyfits
import photfilt2
#import slidop

#_ldCoef = 'PHOEBE'
#_ldCoef = 'ATLAS9'
#_ldCoef = 'NEILSON'
_ldCoef = 'SATLAS'

if _ldCoef=='PHOEBE':
    import ldphoebe
elif _ldCoef=='ATLAS9':
    import atlas9_cld
elif _ldCoef=='NEILSON':
    import neilson_cld
elif _ldCoef=='SATLAS':
    import ldsatlas
print ' > Limb Darkening models:', _ldCoef

import dpfit

__SEDmodel = 'atlas9' # -- check variable atlas9.useBOSZ !!!
#__SEDmodel = 'BOSZ' # -- check variable atlas9.useBOSZ !!!

#__SEDmodel = 'phoenix2' # using Goettingen grid
print ' > SED models models:', __SEDmodel

__monochromaticAlambda = False

_dir_data = '../SPIPS/DATA/'
for d in sys.path:
    if os.path.isdir(d) and 'SPIPS' in d:
        _dir_data = os.path.join(d, 'DATA')
        print '\033[43m', _dir_data, '\033[0m'

_dir_export = './'

plt.rc('font', family='monofur', size=9, style='normal')
#plt.rc('font', family='courier', size=8)

matplotlib.use('pgf') # ???

C_c    = 299792458.0 # m/s
C_G    = 6.6726e-11 # m3 kg-1 s-2
C_Msol = 1.988547e30  # kg
C_Rsol = 6.95508e8    # m
C_pc   = 3.0857e16  # m
C_Teffsol = 5779.57 # in K
C_mas = np.pi/(180*3600*1000.) # rad/mas


print "==== Spectro/Photo/Interferometry of Pulsating Stars ===="

def clean():
    """
    erase all hard coded files used to accelerate the calculations. Run if you added a new photometric filter, modified the photometric calculations or if you modified the CSE model for instance.
    """
    global __MAGgrid, __biasData, __SPE
    for f in ['BW2_maggrid_atlas9.dpy',
              'BW2_maggrid_phoenix2.dpy',
              'BW2_diambiask.dpy',
              'BW2_ATLAS9_aLambdaSPE.dpy',
              'BW2_satlas_models.dpy']:
        try:
            os.remove(os.path.join(_dir_data, f))
            print 'removing file:', os.path.join(_dir_data, f)
        except:
            pass
    # -- erasing variables containing those informations
    print 'removing internal variables'
    __MAGgrid = None
    __biasData = None
    photfilt2._data = None
    if _ldCoef=='SATLAS':
        ldsatlas._data = None
    __SPE = None
    print 'reloading photfilt2'
    reload(photfilt2)
    print ''
    print '*'*34
    print 'you should reload spips now!'
    print '*'*34
    return

# ==========================================================================
# ==========================================================================
# ==========================================================================

def fit(allobs, first_guess, doNotFit=None, guessDoNotFit=False, fitOnly=None, follow=None,
        N_monte_carlo=0, monte_carlo_method='randomize', ftol=1e-4, epsfcn=1e-7, plot=True,
        starName='', maxfev=0, normalizeErrors=False, maxCores=None, verbose=True,
        exportFits=False):
    """
    perform a parallax of pulsation fit to the data 'obs'.

    'obs' is a list of tuple containing the observations, each tuple should have
    the form accetpable by 'model' or 'modelM' function, with the measured
    values and errors added to it. For example:

    (mjd, 'vpuls') -> (mjd, 'vpuls', 12.24, 0.07)

    etc.
    """
    global phaseOffset
    phaseOffset = 1.0

    # -- keep only data with strictly positive errs
    obs = filter(lambda o: o[-1]>0, allobs)
    if len(obs)<len(allobs):
        print 'WARNING: Ignoring %d/%d data points'%(len(allobs)-len(obs), len(allobs))

    if guessDoNotFit:
        fitOnly=None
        #-- a good guess is that you rarely want to fit those:
        doNotFit = ['MJD0','METAL']
        doNotFit.extend(filter(lambda x: 'COMP 'in x, first_guess.keys()))
        if not any(['normalized spectrum' in o[1] for o in obs]):
            #-- no need to fit spectosctopic parameters if there are no
            #-- spectroscopic data in the dataset.
            doNotFit.extend(['Rspec', 'Vspec'])
        if not any(['mag' in o[1] or 'color' in o[1] for o in obs]):
            doNotFit.append('E(B-V)')
        if not any(['mag' in o[1] or
                    'color' in o[1] or
                    'teff' in o[1] for o in obs]):
            #-- no need to fit Teff params
            doNotFit.extend(filter(lambda x: 'TEFF ' in x, first_guess.keys()))
        if not any(['vrad' in o[1] or 'vpuls' in o[1]
                    or 'normalized spectrum' in o[1] for o in obs]):
            doNotFit.append('d_kpc')
            doNotFit.extend(filter(lambda x: 'VPULS ' in x, first_guess.keys()))
        if not( (any(['vrad' in o[1] for o in obs]) and
                 not any(['vpuls' in o[1] for o in obs])) or
                (any(['vuls' in o[1] for o in obs]) and
                 not any(['vrad' in o[1] for o in obs]))):
            # -- need Vrad and Vpuls to fit P-FACTOR
            doNotFit.append('P-FACTOR')
        if 'P-FACTOR' in first_guess.keys() and\
            'd_kpc' in first_guess.keys() and\
            not 'P-FACTOR' in doNotFit and\
            not 'd_kpc' in doNotFit :
            doNotFit.append('P-FACTOR')
        if verbose:
            print 'doNotFit=', doNotFit

    errs = [o[-1] for o in obs]
    if normalizeErrors is True:
        normalizeErrors='observables'
    if normalizeErrors=='observables':
        # -- list all famillies of observable
        types = []
        if verbose:
            print '='*3, 'normalizing error bars', '='*3
        for o in obs:
            if 'color' in o[1] or 'mag' in o[1] or 'flux' in o[1]:
                # -- ignore different sources:
                #types.append(o[1].split(';')[0]+o[2])
                # -- differentiate different sources:
                types.append(o[1]+' '+o[2])
            elif 'diam' in o[1]:
                if np.isscalar(o[2]):
                    # -- ignore different sources:
                    #types.append(o[1].split(';')[0]+str(o[2]))
                    # -- differentiate different sources:
                    types.append(o[1]+' '+str(o[2]))
                else:
                    # -- ignore different sources:
                    #types.append(o[1].split(';')[0]+str(o[2][0]))
                    # -- differentiate different sources:
                    types.append(o[1]+' '+str(o[2][0]))
            else:
                # -- ignore different sources:
                #types.append(o[1].split(';')[0])
                # -- differentiate different sources:
                types.append(o[1])

        for t in set(types):
            w = np.where(np.array(types)==t)
            norma = np.sqrt(float(len(w[0]))/len(types)*len(set(types)))
            if verbose:
                print '%-45s -> *= %5.3f'%(t, norma)
            for i in w[0]:
                errs[i] *= norma

    if normalizeErrors=='techniques':
        # -- 2nd stage of normalization, by techniques:
        techs = [('vrad', 'vpuls'),
                 ('diam', 'UDdiam'),
                 ('teff',),
                 ('magVIS','colorVIS'),
                 ('magNIR','colorNIR')]
        techs = [('vrad', 'vpuls'),
                  ('diam', 'UDdiam'),
                  ('teff',),
                  ('magVIS',),
                  ('magNIR',),
                  ('colorVIS','colorNIR')]

        n = np.zeros(len(techs))
        for o in obs:
            _tech = o[1].split(';')[0]
            if _tech=='mag' or _tech=='flux':
                _wl = photfilt2.effWavelength_um(o[2])
                if _wl>1.0:
                    _tech += 'NIR'
                else:
                    _tech += 'VIS'
            elif _tech=='color':
                _wl = np.mean([photfilt2.effWavelength_um(o[2].split('-')[0]),
                               photfilt2.effWavelength_um(o[2].split('-')[1])])
                if _wl>1.0:
                    _tech += 'NIR'
                else:
                    _tech += 'VIS'
            for j,t in enumerate(techs):
                    if any([_tech==x for x in t]):
                        n[j]+=1
        if verbose:
            print '-'*50
            print 'error bar normalization by technique:'
            for j,t in enumerate(techs):
                print t, n[j], np.sqrt(float(n[j])/len(obs)*len(n))
            print '-'*50

        for i,o in enumerate(obs):
            for j,t in enumerate(techs):
                if any([x in o[1] for x in t]):
                    errs[i] *= np.sqrt(float(n[j])/len(obs)*len(n))


    # -- avoid fitting string parameters
    tmp = filter(lambda x: isinstance(first_guess[x], str), first_guess.keys())
    if len(tmp)>0:
        if doNotFit is None:
            doNotFit=tmp
        else:
            doNotFit.extend(tmp)
        try:
            fitOnly = filter(lambda x: not isinstance(first_guess[x], str), fitOnly)
        except:
            pass

    if N_monte_carlo>0:
        mc_fits = []
        if verbose:
            print 'Bootstraping: (', monte_carlo_method, ')'

        while len(mc_fits)<N_monte_carlo:
            print '='*10, len(mc_fits)+1, '/', N_monte_carlo, time.asctime(), '='*20
            obsp=[]
            if monte_carlo_method=='shuffle':
                # -- shuffle data
                w = np.int_(len(obs)*np.random.rand(len(obs)))
                for i in w:
                    obsp.append(obs[i])
            elif monte_carlo_method=='scramble':
                # -- scramble with respect to the error bar
                for o in obs:
                    tmp = list(o)
                    tmp[-2] += np.random.randn()*tmp[-1]
                    obsp.append(tuple(tmp))
            elif monte_carlo_method=='scramffle':
                # -- shuffle AND scramble data
                w = np.int_(len(obs)*np.random.rand(len(obs)))
                for i in w:
                    tmp = list(obs[i])
                    tmp[-2] += np.random.randn()*tmp[-1]
                    obsp.append(tmp)
            else:
                print 'unknown Monte Carlo Method:', monte_carlo_method

            errs = [o[-1] for o in obsp]
            if normalizeErrors:
                # -- list all famillies of observable
                types = []
                for o in obsp:
                    if 'color' in o[1] or 'mag' in o[1] or 'flux' in o[1]:
                        # -- ignore different sources:
                        #types.append(o[1].split(';')[0]+o[2])
                        # -- differentiate different sources:
                        types.append(o[1]+' '+o[2])
                    elif 'diam' in o[1]:
                        if np.isscalar(o[2]):
                            # -- ignore different sources:
                            #types.append(o[1].split(';')[0]+str(o[2]))
                            # -- differentiate different sources:
                            types.append(o[1]+' '+str(o[2]))
                        else:
                            # -- ignore different sources:
                            #types.append(o[1].split(';')[0]+str(o[2][0]))
                            # -- differentiate different sources:
                            types.append(o[1]+' '+str(o[2][0]))
                    else:
                        # -- ignore different sources:
                        #types.append(o[1].split(';')[0])
                        # -- differentiate different sources:
                        types.append(o[1])

            if normalizeErrors:
                for t in set(types):
                    w = np.where(np.array(types)==t)
                    norma = np.sqrt(1.*len(w[0])/len(types)*len(set(types)))
                    for i in w[0]:
                        errs[i] *= norma
            errs = np.array(errs)

            # -- need to call once before fit
            # -> bug in Chi2 calculations
            # -> init the number of Cores
            tmp=modelM(obsp, first_guess, maxCores=maxCores)

            tmp = dpfit.leastsqFit( modelM, obsp, first_guess,
                        [np.array(o[-2]) for o in obsp],
                        err=errs, doNotFit=doNotFit,
                        fitOnly=fitOnly, verbose=False,
                        ftol=ftol, epsfcn=epsfcn, follow=follow, maxfev=maxfev)

            mc_fits.append(tmp)
            if verbose:
                print ' '*4, 'chi2= %5.3f '%(tmp['chi2']),
                print 'dist= %6.4f pc '%(tmp['best']['d_kpc']),
                print 'p-factor= %5.3f '%(tmp['best']['P-FACTOR'])

        if verbose:
            print '=== > done bootstraping'

        #-- save binary variables
        if starName!='':
            tmp = starName
            tmp = tmp.replace(' ', '_')
            tmp = tmp.replace('$', '')
            tmp = tmp.replace("\\", '')
            filename='mc_'+monte_carlo_method+'_'+tmp+'_N'+str(N_monte_carlo)+\
                      '_'+time.asctime().replace(' ','_').replace(':','')+'.dpy'
        else:
            filename='mc_'+monte_carlo_method+'_N'+str(N_monte_carlo)+\
                      '_'+time.asctime().replace(' ','_').replace(':','')+'.dpy'
        filename.replace(':', '.')
        f = open(os.path.join('MC_RUNS',filename), 'wb')
        mc_run={'first guess':first_guess, 'obs':obs, 'mc_fits':mc_fits}
        print 'saving \'mc_run\' dictionnary in', os.path.join('MC_RUNS',filename)
        cPickle.dump(mc_run, f, 2)
        f.close()
        return mc_fits
    else:
        #-- single Fit: needs to run it once before running the fit
        if maxCores==1:
            tmp = model(obs, first_guess)
            fit=dpfit.leastsqFit(model, obs, first_guess,
                                [o[-2] for o in obs],
                                err=errs, maxfev=maxfev,
                                doNotFit=doNotFit,fitOnly=fitOnly,follow=follow,
                                verbose=verbose, ftol=ftol, epsfcn=epsfcn)

        else:
            tmp = modelM(obs, first_guess, maxCores=maxCores)
            fit=dpfit.leastsqFit(modelM, obs, first_guess,
                                [o[-2] for o in obs],
                                err=errs, maxfev=maxfev,
                                doNotFit=doNotFit,fitOnly=fitOnly,follow=follow,
                                verbose=verbose, ftol=ftol, epsfcn=epsfcn)
            # fit = dpfit2.minimize( modelM, obs, first_guess,
            #             np.array([np.array(o[-2]) for o in obs]),
            #             err=np.array(errs), doNotFit=doNotFit,
            #             fitOnly=fitOnly, verbose=verbose)
        fit['options'] = {'maxfev':maxfev, 'ftol':ftol, 'epsfcn':epsfcn,
                        'normalizeErrors':normalizeErrors,
                        'first_guess':first_guess, 'doNotFit':doNotFit}
        fit['starName'] = starName

        if plot or exportFits:
            mod = model(allobs, fit['best'], plot=plot, starName=starName,
                  verbose=verbose, exportFits=exportFits)
        # -- if fits has been exported, edit file

        if exportFits:
            fit['export'] = mod
            f = pyfits.open(fit['export']['FITS'], mode='update')
            # -- add uncertainties
            for k in sorted(fit['uncer'].keys()):
                f[0].header['HIERARCH UNCER '+k] = fit['uncer'][k]
            # -- add correlation matrix
            cols = []
            n = str(max([len(s) for s in fit['fitOnly']]))
            cols.append(pyfits.Column(name='param', format='A'+n,
                                      array=fit['fitOnly']))
            for k in range(len(fit['fitOnly'])):
                cols.append(pyfits.Column(name=fit['fitOnly'][k],
                                          format='E',
                                          array=fit['cor'][k,:]))

            hdum = pyfits.BinTableHDU.from_columns(cols)
            hdum.header['EXTNAME'] = 'CORREL'
            f.append(hdum)
            f.flush()
            f.close()
            directory = fit['export']['FITS'].split('.fits')[0].upper()
            fit2html(fit,
                    fit['export']['FITS'].split('.fits')[0].lower(),
                    directory)
            os.rename(fit['export']['FITS'],
                        os.path.join(directory, os.path.basename(fit['export']['FITS'])))
            fit['export']['FITS'] = os.path.join(directory, os.path.basename(fit['export']['FITS']))
            if 'FIG' in fit['export']:
                for f in fit['export']['FIG']:
                    os.rename(f, os.path.join(directory, os.path.basename(f)))
        return fit

def fit2html(f, root='spips', directory='./', makePlots=True):
    """
    'f' is the result from the "fit" function, ran with "plot=True" and
    "exportFits=True"
    """
    if not os.path.isdir(directory):
        os.mkdir(directory)

    # -- text file with parameters and uncertainties
    keys = f['best'].keys()
    keys.sort()
    filename1 = root+'_param.html'
    fi = open(os.path.join(directory, filename1), 'w')
    fi.write('<!DOCTYPE html>\n')
    fi.write('<html>\n')
    fi.write('<body>\n')
    fi.write('<a href="https://github.com/amerand/SPIPS" target="_blank">SPIPS</a>')
    fi.write(' model for star <b>'+f['starName']+'</b> ')
    #fi.write('<a href="file://'+os.path.abspath(f['export']['FITS'])+'" target="_blank">'+
    #          '[FITS export]</a>'
    fi.write('<a href="'+os.path.basename(f['export']['FITS'])+'" target="_blank">'+
              '[FITS export]</a>')
    fi.write('</br>\n')
    fi.write('<hr>\n')
    for k in keys:
        if f['uncer'][k]>0:
            n = int(np.ceil(-np.log10(f['uncer'][k]))+1)
            fmt = "<b>'%s': %."+str(n)+'f, # +/- %.'+str(n)+'f </b>'
            fi.write(fmt%(k, f['best'][k], f['uncer'][k])+'</br>\n')
        else:
            fmt = "'%s': %s,"
            fi.write(fmt%(k, str(f['best'][k]))+'</br>\n')
    fi.write('</br>\n')
    fi.write('<hr>\n')
    for k in f['options']:
        if not k in ['first_guess', 'doNotFit']:
            fi.write('%s: %s </br>\n'%(k, str(f['options'][k])))

    fi.write('</body>\n')
    fi.write('</html>\n')
    fi.close()

    # -- correlation matrix
    filename2 = root+'_corr.html'
    fi = open(os.path.join(directory, filename2), 'w')

    # -- parameters names:
    nmax = np.max([len(x) for x in f['fitOnly']])
    fmt = '%%%ds'%nmax
    fmt = '%2d:'+fmt
    colors = {0.0:'#FFFFFF',
              0.5:'#CCCCCC',
              0.7:'#FFEE66',
              0.8:'#FF6666',
              0.9:'#FF66FF',}
    fi.write('<!DOCTYPE html>\n')
    fi.write('<html>\n')
    fi.write('<body>\n')
    fi.write('<table style="width:100%" border="1">\n')
    fi.write('<tr>\n')
    fi.write('<th> correlation thresholds </th>\n')
    for k in sorted(colors.keys()):
        fi.write('<th bgcolor="%s">%s</th>'%(colors[k], str(k)))
    fi.write('</tr>\n')
    fi.write('<tr>\n')
    fi.write('</table')
    fi.write('</br>\n')

    fi.write('<table style="width:100%" border="1">\n')
    # -- columns header
    fi.write('<tr>\n')
    fi.write('<th> parameters </th>\n')
    for i in range(len(f['fitOnly'])):
        fi.write('<th>%02d</th>'%i)
    fi.write('</tr>\n')
    fi.write('\n')
    # -- for each parameters
    for i,p in enumerate(f['fitOnly']):
        fi.write('<tr>\n ')
        fi.write('<td>'+str(i)+': '+p+'</td>')
        for j, x in enumerate(f['cor'][i,:]):
            # tmp = '%4.1f'%x
            # tmp = tmp.replace('0.', '.')
            # tmp = tmp.replace('1.0', '1.')
            tmp = ''
            u = False
            if (p.startswith('TEFF') and
                f['fitOnly'][j].startswith('TEFF')) or \
                 (p.startswith('VPULS') and
                f['fitOnly'][j].startswith('VPULS')) or \
                 (p.startswith('VRAD') and
                f['fitOnly'][j].startswith('VRAD')):
                u = True

            c = colors[0]
            if i!=j:
                for k in sorted(colors.keys()):
                    if abs(x)>=k:
                        c = colors[k]
            if u:
                tmp = '<b>'+tmp+'</b>'
                tmp = '-'
            if i==j:
                tmp = '#'
            fi.write('<td bgcolor="%s"> %s </td> '%(c,tmp))

        fi.write('\n</tr>\n')
    fi.write('</table>\n')


    fi.write('</body>\n')
    fi.write('</html>\n')
    fi.close()

    # -- param+corr page
    filename3 = root+'_main.html'
    fi = open(os.path.join(directory, filename3), 'w')

    page = """
        <!DOCTYPE html>
        <html>
        <FRAMESET cols="30%, 70%">
            <FRAMESET rows="30%, 70%">
                <FRAME src="__figure1__">
                <FRAME src="__filename1__">
            </FRAMESET>
            <FRAMESET rows="70%, 30%">
                <FRAME src="__figure0__">
                <FRAME src="__filename2__">
            </FRAMESET>
        </FRAMESET>
        </html>
        """
    page = """
    <!DOCTYPE html>
    <html>
    <FRAMESET rows="70%, 30%">
      <FRAMESET cols="20%, 80%">
        <FRAME src="__filename1__">
        <FRAME src="__figure0__">
      </FRAMESET>
      <FRAMESET cols="70%, 30%">
        <FRAME src="__filename2__">
        <FRAME src="__figure1__">
      </FRAMESET>
    </FRAMESET>
    </html>
    """
    page = page.replace('__filename1__', filename1).replace('__filename2__', filename2)
    page = page.replace('__figure0__',
                os.path.basename(filter(lambda x: 'Fig0' in x, f['export']['FIG'])[0]))
    page = page.replace('__figure1__',
                os.path.basename(filter(lambda x: 'Fig1' in x, f['export']['FIG'])[0]))

    fi.write(page)
    fi.close()
    return

def dispCor(fit):
    # -- parameters names:
    nmax = np.max([len(x) for x in fit['fitOnly']])
    fmt = '%%%ds'%nmax
    fmt = '%2d:'+fmt
    print '|Correlations| ',
    print '\033[45m>=.9\033[0m',
    print '\033[41m>=.8\033[0m',
    print '\033[43m>=.7\033[0m',
    print '\033[100m>=.5\033[0m',
    print '\033[0m>=.2\033[0m',
    print '\033[90m<.2\033[0m'

    print ' '*(2+nmax),
    for i in range(len(fit['fitOnly'])):
        print '%3d'%i,
    print ''
    for i,p in enumerate(fit['fitOnly']):
        print fmt%(i,p),
        for j, x in enumerate(fit['cor'][i,:]):
            if i==j:
                c = '\033[2m'
            elif (p.startswith('TEFF') and
                fit['fitOnly'][j].startswith('TEFF')) or \
                 (p.startswith('VPULS') and
                fit['fitOnly'][j].startswith('VPULS')) or \
                 (p.startswith('VRAD') and
                fit['fitOnly'][j].startswith('VRAD')):
                c = '\033[4m'
            else:
                c = '\033[0m'
            if i!=j:
                if abs(x)>=0.9:
                    col = '\033[45m'
                elif abs(x)>=0.8:
                    col = '\033[41m'
                elif abs(x)>=0.7:
                    col = '\033[43m'
                elif abs(x)>=0.5:
                    col = '\033[100m'
                elif abs(x)<0.2:
                    col = '\033[90m'
                else:
                    col = ''
            else:
                col = ''
            tmp = '%4.1f'%x
            tmp = tmp.replace('0.', '.')
            tmp = tmp.replace('1.0', '1.')
            if i==j:
                tmp = '###'
            print c+col+tmp+'\033[0m',
        print ''

def smartFit(obs, firstGuess, phaseDependent={'VPULS':('spline', 4),'TEFF':('fourier', 4)}, fitPeriod=False):
    """
    firstGuess should at least contain 'PERIOD', 'MJD0' and 'd_kpc'
    """
    # -- first fit the radial velocities: faster to keep only those
    obsp = filter(lambda x: x[1]=='vrad' or x[1]=='vpuls', obs)
    # -- P-R from Molinaro et al. 2012
    R_expected = 10**(0.75*np.log10(firstGuess['PERIOD'])+1.10)
    fg = {'DIAM0':2*R_expected/firstGuess['d_kpc']/107.5}
    for k in firstGuess:
        fg[k] = firstGuess[k]
    if not(any([o[1]=='vpuls' for o in obsp])):
        fg['P-FACTOR']=1.27
    if phaseDependent['VPULS'][0]=='fourier':
        if not(any([o[1]=='vpuls' for o in obsp])):
            fg['VPULS A0'] = 0.0 # gamma velocity for radial velocity
        for k in range(1,phaseDependent['VPULS'][1]+1):
            fg['VPULS A'+str(k)] = 1.0
            fg['VPULS PHI'+str(k)] = 0.0
    elif phaseDependent['VPULS'][0]=='spline':
        x_ = np.linspace(0,1,phaseDependent['VPULS'][1]+1)[:-1]
        for k in range(phaseDependent['VPULS'][1]):
            fg['VPULS PHI'+str(k)] = x_[k]
            fg['VPULS VAL'+str(k)] = 0.0
    fitOnly = filter(lambda x: 'VPULS ' in x, fg.keys())
    if fitPeriod:
        fitOnly.extend(filter(lambda x: 'PERIOD' in x, firstGuess.keys()))
    fit1 = fit(obsp, fg, fitOnly=fitOnly, plot=True, ftol=1e-3)
    print fit1['best']

    # -- add a single averaged temperature and fit Diam0 and d_kpc
    fg = fit1['best']
    fitOnly = ['d_kpc', 'DIAM0']
    if phaseDependent['TEFF'][0]=='fourier':
        fg['TEFF A0'] = 5000.0 #
        for k in range(1,phaseDependent['TEFF'][1]+1):
            fg['TEFF A'+str(k)] = 0.0
            fg['TEFF PHI'+str(k)] = 0.0
        if any([o[1]=='teff' or o[1]=='mag' or o[1]=='color' for o in obs]):
            fitOnly.append('TEFF A0')
    elif phaseDependent['TEFF'][0]=='spline':
        x_ = np.linspace(0,1,phaseDependent['TEFF'][1]+1)[:-1]
        for k in range(phaseDependent['VPULS'][1]):
            fg['TEFF PHI'+str(k)] = x_[k]
            fg['TEFF VAL'+str(k)] = 5000.0
            if any([o[1]=='teff' or o[1]=='mag' or o[1]=='color' for o in obs]):
                fitOnly.append('TEFF VAL'+str(k))
    if any([o[1]=='mag' or o[1]=='color' for o in obs]):
        fg['E(B-V)']=0.0
    if fitPeriod:
        fitOnly.extend(filter(lambda x: 'PERIOD' in x, firstGuess.keys()))
    fit1 = fit(obs, fg, fitOnly=fitOnly, plot=True)

    #-- fit everything altogether
    fg = fit1['best']
    fit1 = fit(obs, fg, guessDoNotFit=True, plot=True, epsfcn=1e-8)
    return fit1

def sigmoid(x,x0=0,dx=0.01):
    """
    map 0 -> 1 to -dx -> dx
    """
    if dx<=0:
        return x
    _s = lambda x: 1/(1+np.exp(-x))
    _x = -dx + 2*((x-x0)%1)*dx
    return ((_s(_x)-_s(-dx))/(_s(dx)-_s(-dx)) + x0)%1.

def fitsName(starName):
    if starName is None:
        return 'spips'
    else:
        filename = starName.lower().replace(' ', '_')
        filename = filename.replace('$','').replace("\\",'')
        filename = filename.replace('{','').replace('}','')
        filename = filename.replace(';','_')
        return filename

def model(x, a, plot=False, starName=None, verbose=False, uncer=None, showOutliers=False,
          exportFits=False, maxCores=None, splitDTeffects=False):
    """
    returns observable 'x' (Vpuls, photometry, etc) for a given model
    based on parameters 'a':

    x = list of tuples: (mjd, type, data...)
        if MJD <1, then the code assumes it is a pulsation phase and will not
        recompute it with the ephemeris given in 'a' ('PERIOD', 'MJD0')

        for example *x* means optional:
        (mjd, 'vpuls') -> Vpuls in km/s

        (mjd, 'vrad') -> radial velocity in km/s

        (mjd, 'vgamma') -> average Vpuls in km/s (mjd is ignored)

        (mjd, 'diam', *[wavelength_um, baseline_m]*) -> diameters in mas

        (mjd, 'UDdiam', [wl_um, *baseline_m*]) -> UD diameter in mas, at wavelength wl (um)

        (mjd, 'v2fluor', baseline_m) -> squared visibility for fluor

        (mjd, 'teff') -> effective temperature

        (mjd, 'Mbol') -> Absolute bolometric magnitude

        (mjd, 'mag', 'filter') -> photometric magnitude, for a given filter
        (mjd, 'flux', 'filter') -> apparent flux in Jy
        (None, 'mag', 'filter') -> photometric magnitude, for a given filter,
                        at unknown phase


        (mjd, 'color', 'filter1-filter2') -> filter1-filter2

        (mjd, 'normalized spectrum', wl_um) -> normalized spectrum;
                         assumes only one instrument is used,
                         i.e. single Rspec fo all spectra)

    Note that only the part before ';' will be considered for an observation's
    name, leaving room to keep track of comments (such as origin of the
    observation). For instance:

    (mjd, 'vrad; Bersier et al. 1994')

    a is a dictionnary: e.g,

    a={'PERIOD':35.551341000000001,
       'MJD0':52290.508124788146,
       'DIAM0':2.7653404798701073, # in mas
       'd_kpc':0.60066719626599596,  # in kpc
       'VPULS PHI0':-0.12539413431736629,
       'VPULS PHI1': 0.033484967745679037,
       'VPULS PHI2': 0.71483220499480626,
       'VPULS PHI3': 0.75988519192623627,
       'VPULS PHI4': 0.94896962017838637,
       'VPULS VAL0': 18.631596918810043, # in km/s
       'VPULS VAL1': -21.016020628354106,
       'VPULS VAL2': 27.495143188934478,
       'VPULS VAL3': 28.402426499671989,
       'VPULS VAL4': -12.592838804125455,
       'TEFF PHI0': -0.14499127119607119,
       'TEFF PHI1': 0.022753518466214342,
       'TEFF PHI2': 0.71599188489829835,
       'TEFF VAL0': 4963.4638463137926, # in K
       'TEFF VAL1': 5156.5715783762689,
       'TEFF VAL2': 4556.2069715595271,
       'E(B-V)':0.077356456362560944,
       }

    - 'd_kpc' is the distance in kilo-parsecs
    - 'P-FACTOR' is the spectroscopi projection factor
    - 'METAL:'' optional keyword to give the metalicity (0 for Solar)
    - 'MJD0', 'PERIOD': are the pulsation ephemeris (period in days)
    - optional: PERIOD1, PERIOD2, the 1rst and second order period variations
    - DIAM0: is the average angular diameter in mas
    - E(B-V) is the color excess (for redenning correction).
    - Rspec is the spectral resolution /1000. (will convolve outpu spectra)
    - Vspec,  additional velocity for spectra in km/s
    - 'K EXCESS' K band magnitude excess (applies also to H band, but with a
      factor 0.5). >0 means the model will be brighter in K
    - 'Rv': reddening function (between 3.1 and 5, default is 3.1)

    additional parameters are: 'COMP DIAM', 'COMP TEFF', 'COMP LOGG' for a
    non-pulsating companion in the photometric field of view.

    VPULS and TEFF phase-dependent profiles can be parametrized using periodic
    splines or Fourier terms. In case of Spline, the parameters are {'VPULS
    PHI0', 'VPULS VAL0', ..., 'VPULS PHIi', 'VPULS VALi', ...} for the nodes
    positions (phi_i, val_i). In case of Fourier parameters, one should give
    {'VPULS A0', 'VPULS A1', 'VPULS PHI1',..., 'VPULS Ai','VPULS PHIi', ...} for
    the amplitudes and phases of the Fourier terms. Note that 'PHI0' is not a
    useful parameters and that 'A0' will be the gamma velocity for VPULS and the
    average temperature for TEFF.

    Splines can be also defined using a 'comb' for the phase axis. The comb is
    defined usinf 'POW' and 'PHI0': 'POW' defines the over-density of nodes
    around phase 'PHI0'. 'POW' of 1 is uniform phase coverage, and larger values
    of POW will lead to higher densities. Typical value of POW is 2.5. Nodes
    values can be defined using 'VAL0' for the first one and then using 'DVALi'
    for i>0. 'VAL0' defines the value of the node for the lowest phase density,
    which ensures that VAL0 is close to the average value of the profile. This is
    unfortunately at the expense of naming logic, because one might have expected
    VAL0 == value at PHI0. Example:
    {
    'VRAD DVAL1':   17.301,
    'VRAD DVAL2':   8.933,
    'VRAD DVAL3':   -3.685,
    'VRAD DVAL4':   -11.586,
    'VRAD DVAL5':   -16.593,
    'VRAD PHI0':    0.91675,
    'VRAD POW':     2.388,
    'VRAD VAL0':    21.612,
    }

    Note about redenning: the redenning is corrected using a law for ISM



    """
    global phaseOffset

    if verbose:
        print '-'*20, 'MODEL', '-'*28
    maxTeff = 10000
    minTeff = 4000.
    maxLogg = 5.0
    minLogg = 0.0


    # compute Vpuls and angular diam for regular padded points
    # get control points and duplicate them to get from phi=-1 to phi=+2
    Nintern = 3000
    phi_intern = np.linspace(0.,1.,Nintern)[:-1]
    phi_intern = np.linspace(-1.,2.,Nintern)[:-1]

    fits_model = collections.OrderedDict()

    if a.has_key('TEFF PHASE OFFSET'):
        teff_phase_offset = a['TEFF PHASE OFFSET']
    else:
        teff_phase_offset = 0.0

    if a.has_key('PHOT CORR'):
        phot_corr = a['PHOT CORR']
    else:
        phot_corr = 1.0

    # -- K band excess
    if a.has_key('K EXCESS'):
        k_excess = a['K EXCESS']
    else:
        k_excess = 0.

    # -- excess exponent:
    if a.has_key('EXCESS SLOPE') and a.has_key('K EXCESS'):
        f_excess = lambda x: max(0, a['K EXCESS'] + (x-2.12)*a['EXCESS SLOPE'])*(x>=1.1)
        title_excess = 'IR$_{ex}$ = %5.3f + %5.3f($\lambda$-2.12) mag'%(a['K EXCESS'],a['EXCESS SLOPE'])
        k_excess =  f_excess(2.12)
    elif a.has_key('EXCESS EXP') and a.has_key('K EXCESS'):
        f_excess = lambda x: max(k_excess*(x/2.2)**a['EXCESS EXP'],0.)*(x>=1.1)
        title_excess = 'IR$_{ex}$ = %5.3f($\lambda$/2.2)$^{%5.3f}$ mag'%(k_excess, a['EXCESS EXP'])
        k_excess = f_excess(2.12)
    elif a.has_key('EXCESS SLOPE') and a.has_key('EXCESS WL0'):
        if a.has_key('EXCESS EXP'):
            f_excess = lambda x: a['EXCESS SLOPE']*max(x-a['EXCESS WL0'],0)**a['EXCESS EXP']
            title_excess = 'IR$_{ex}$ = %5.3f($\lambda$ - %5.3f)$^{%5.3f}$ mag'%(
                                        a['EXCESS SLOPE'], a['EXCESS WL0'], a['EXCESS EXP'])
        else:
            f_excess = lambda x: a['EXCESS SLOPE']*max(x-a['EXCESS WL0'],0)
            title_excess = 'IR$_{ex}$ = %5.3f($\lambda$-%5.3f) mag'%(
                                        a['EXCESS SLOPE'], a['EXCESS WL0'] )
        k_excess = f_excess(2.12)
    else:
        f_excess = None

    if a.has_key('Rshell/Rstar'):
        RshellRstar = np.abs(a['Rshell/Rstar'])
    else:
        RshellRstar = 2.5

    h_excess = 0.0
    # if a.has_key('H EXCESS'):
    #     h_excess = np.abs(a['H EXCESS'])
    # else:
    #     # -- 1/2 the K excess:
    #     h_excess = -2.5*np.log10((10**(-k_excess/2.5)-1)/2+1)
    #     if h_excess!=0 :
    #         fits_model['H EXCESS'] = h_excess

    j_excess = 0.0;
    # if a.has_key('J EXCESS'):
    #     j_excess = np.abs(a['J EXCESS'])
    # else:
    #     # -- 1/4 the K excess:
    #     j_excess = -2.5*np.log10((10**(-k_excess/2.5)-1)/4+1)
    #     if j_excess!=0 :
    #         fits_model['J EXCESS'] = j_excess

    l_excess = 0.0
    # if a.has_key('L EXCESS'):
    #     l_excess = np.abs(a['L EXCESS'])
    # else:
    #     # -- 1/4 the K excess:
    #     l_excess = -2.5*np.log10((10**(-k_excess/2.5)-1)*2+1)
    #     if l_excess!=0 :
    #         fits_model['L EXCESS'] = l_excess

    if a.has_key('Rv'):
        Rv = a['Rv']
    else:
        Rv = 3.1 # default redenning
        fits_model['Rv'] = Rv

    if a.has_key('P-FACTOR'):
        pfactor = a['P-FACTOR']
    else:
        pfactor = 1.27
        fits_model['P-FACTOR'] = pfactor

    # -- replicate VPULS points to get phase -4 -> +4
    if any(['VPULS VAL' in k for k in a.keys()]):
        # -- using splines:
        useFourierVpuls=False
        nvpuls = len(filter(lambda x: 'VPULS VAL' in x, a.keys()))
        nx = nvpuls
        if nvpuls == 1:
            # -- checks if values are given with respect to first one:
            ndvpuls = len(filter(lambda x: 'VPULS DVAL' in x, a.keys()))
            nx += ndvpuls
        if 'VPULS POW' in a.keys():
            # -- nodes
            #x0 = np.linspace(-0.5, 0.5, nvpuls)
            #x0 = np.sign(x0)*(np.abs(x0)**a['VPULS POW'])+a['VPULS PHI0']
            # -- nodes, alternate formula
            x0 = np.linspace(-1., 1., nx+1)
            x0 = 0.5*np.sign(x0)*(np.abs(x0)**a['VPULS POW']) + a['VPULS PHI0']
            x0 = x0[:-1]
            xv_pow = x0
            x0_i = range(nx)
        else:
            # -- create list of nodes:
            x0 = np.array([a['VPULS PHI'+str(k)] for k in range(nx)])
            xv_pow = None
            x0_i = None

        y0 = [a['VPULS VAL0']]
        if nvpuls > 1:
            y0.extend([a['VPULS VAL'+str(k+1)] for k in range(nvpuls-1)])
        else:
            y0.extend([a['VPULS DVAL'+str(k+1)] + a['VPULS VAL0']
                        for k in range(ndvpuls)])

        y0 = np.array(y0)
        #print y0, x0

        y0 = y0[x0.argsort()]
        x0 = x0[x0.argsort()]

        xp = list(x0-2)
        xp.extend(list(x0-1))
        xp.extend(list(x0))
        xp.extend(list(x0+1))
        xp.extend(list(x0+2))

        yp = list(y0)*5
        if not x0_i is None:
            x0_i = x0_i*5

        xp = np.array(xp)
        yp = np.array(yp)

        if not uncer is None:
            ex0 = np.array([uncer['VPULS PHI'+str(k)] for k in range(nvpuls)])%1
            ey0 = np.array([uncer['VPULS VAL'+str(k)] for k in range(nvpuls)])
            ey0 = ey0[ex0.argsort()]
            ex0 = ex0[ex0.argsort()]

            exp = list(ex0)
            exp.extend(list(ex0))
            exp.extend(list(ex0))
            exp.extend(list(ex0))
            exp.extend(list(ex0))

            eyp = list(ey0)
            eyp.extend(list(ey0))
            eyp.extend(list(ey0))
            eyp.extend(list(ey0))
            eyp.extend(list(ey0))

            exp = np.array(exp)
            eyp = np.array(eyp)

        # -- Vpuls in km/s, including Vgamma
        if 'VPULS KIND' in a:
            _kind = a['VPULS KIND']
        else:
            _kind='cubic'
            fits_model['VPULS SPLINE KIND'] = 'cubic'

        Vpuls = interp1d(xp, yp, kind=_kind, fill_value=0.0,
                          assume_sorted=True)(phi_intern)
        Vgamma = interp1d(xp, yp, kind=_kind, fill_value=0.0,
                          assume_sorted=True)(np.linspace(0,1,1000)[:-1]).mean()

        if verbose:
            print 'VGAMMA:', round(Vgamma,3), 'km/s'
    elif 'VPULS A0' in a.keys():
        #-- assume Fourier coefficients:
        useFourierVpuls=True

        xvpuls = phi_intern

        Ns = [int(k.split('I')[1]) for k in filter(lambda x: 'VPULS PHI' in x, a.keys())]
        Vpuls = np.ones(Nintern-1)*a['VPULS A0']

        for k in Ns:
            if a.has_key('VPULS A'+str(k)) and \
               a.has_key('VPULS PHI'+str(k)):
               # -- default behaviorr
                Vpuls += a['VPULS A'+str(k)]*np.cos(2*np.pi*k*xvpuls +
                                                    a['VPULS PHI'+str(k)])
            elif a.has_key('VPULS A1') and \
                 a.has_key('VPULS PHI1') and \
                 a.has_key('VPULS R'+str(k)) and \
                 a.has_key('VPULS PHI'+str(k)):
                # -- ratio of amplitudes
                Vpuls += a['VPULS A1']*a['VPULS R'+str(k)]*\
                        np.cos(2*np.pi*k*xvpuls + a['VPULS PHI1'] -
                               a['VPULS PHI'+str(k)])
        Vgamma = Vpuls[:-1].mean()

    xv_pow = None
    # -- replicate VRAD points to get phase -4 -> +4
    if any(['VRAD VAL' in k for k in a.keys()]):
        # -- using splines:
        useFourierVpuls=False
        nvpuls = len(filter(lambda x: 'VRAD VAL' in x, a.keys()))
        nx = nvpuls
        if nvpuls == 1:
            # -- checks if values are given with respect to first one:
            ndvpuls = len(filter(lambda x: 'VRAD DVAL' in x, a.keys()))
            nx += ndvpuls
        else:
            ndvpuls = 0
        if 'VRAD POW' in a.keys():
            # -- nodes
            #x0 = np.linspace(-0.5, 0.5, nvpuls)
            #x0 = np.sign(x0)*(np.abs(x0)**a['VRAD POW'])+a['VRAD PHI0']
            # -- nodes, alternate formula
            x0 = np.linspace(-1., 1., nx+1)
            x0 = 0.5*np.sign(x0)*(np.abs(x0)**a['VRAD POW']) + a['VRAD PHI0']
            x0 = x0[:-1]
            xv_pow = x0
            x0_i = range(nx)
        else:
            # -- create list of nodes:
            x0 = np.array([a['VRAD PHI'+str(k)] for k in range(nx)])
            xv_pow = None
            x0_i = None

        y0 = [a['VRAD VAL0']]
        if nvpuls > 1:
            y0.extend([a['VRAD VAL'+str(k+1)] for k in range(nvpuls-1)])
        else:
            y0.extend([a['VRAD DVAL'+str(k+1)] + a['VRAD VAL0']
                        for k in range(ndvpuls)])

        y0 = np.array(y0)
        #print y0, x0

        y0 = y0[x0.argsort()]
        x0 = x0[x0.argsort()]

        xp = list(x0-2)
        xp.extend(list(x0-1))
        xp.extend(list(x0))
        xp.extend(list(x0+1))
        xp.extend(list(x0+2))

        yp = list(y0)*5
        if not x0_i is None:
            x0_i = x0_i*5

        xpV = np.array(xp)
        ypV = np.array(yp)

        if not uncer is None:
            ex0 = np.array([uncer['VRAD PHI'+str(k)] for k in range(nvpuls)])%1
            ey0 = np.array([uncer['VRAD VAL'+str(k)] for k in range(nvpuls)])
            ey0 = ey0[ex0.argsort()]
            ex0 = ex0[ex0.argsort()]

            exp = list(ex0)
            exp.extend(list(ex0))
            exp.extend(list(ex0))
            exp.extend(list(ex0))
            exp.extend(list(ex0))

            eyp = list(ey0)
            eyp.extend(list(ey0))
            eyp.extend(list(ey0))
            eyp.extend(list(ey0))
            eyp.extend(list(ey0))

            exp = np.array(exp)
            eyp = np.array(eyp)

        # -- Vpuls in km/s, including Vgamma
        if 'VRAD KIND' in a:
            _kind = a['VRAD KIND']
        else:
            _kind='cubic'
            fits_model['VRAD SPLINE KIND'] = 'cubic'

        Vrad = interp1d(xpV, ypV, kind=_kind, fill_value=0.0,
                          assume_sorted=True)(phi_intern)
        Vgamma = interp1d(xpV, ypV, kind=_kind, fill_value=0.0,
                          assume_sorted=True)(np.linspace(0,1,1000)[:-1]).mean()

        Vpuls = (Vrad-Vgamma)*a['P-FACTOR'] + Vgamma

        if verbose:
            print 'VGAMMA:', round(Vgamma,3), 'km/s'
    elif 'VRAD A0' in a.keys():
        #-- assume Fourier coefficients:
        useFourierVpuls=True

        xvpuls = phi_intern

        Ns = [int(k.split('I')[1]) for k in filter(lambda x: 'VRAD PHI' in x, a.keys())]
        Vrad = np.ones(Nintern-1)*a['VRAD A0']

        for k in Ns:
            if a.has_key('VRAD A'+str(k)) and \
               a.has_key('VRAD PHI'+str(k)):
                # -- default behavior
                _phi = xvpuls + a['VRAD PHI'+str(k)]/(2*np.pi)/k
                Vrad += a['VRAD A'+str(k)]*np.cos(2*np.pi*k*_phi)
            elif a.has_key('VRAD A1') and \
                 a.has_key('VRAD PHI1') and \
                 a.has_key('VRAD R'+str(k)) and \
                 a.has_key('VRAD PHI'+str(k)):
                # -- ratio of amplitudes
                Vrad += a['VRAD A1']*a['VRAD R'+str(k)]*\
                        np.cos(2*np.pi*k*xvpuls + a['VRAD PHI1'] -
                               a['VRAD PHI'+str(k)])
        Vgamma = Vrad[:-1].mean()
        Vpuls = (Vrad-Vgamma)*a['P-FACTOR'] + Vgamma

    # ================================

    fits_model['VGAMMA_KM/S'] = Vgamma

    # -- extract julian dates as the first value in the tuple
    mjd = np.array([obs[0] if not obs[0] is None else np.nan for obs in x])

    phi, Period = phaseFunc(mjd, a, vgamma=Vgamma)

    Period[np.isnan(Period)] = np.nanmean(Period)
    if all(np.isnan(Period)):
        Period = a['PERIOD']*np.ones(len(mjd))
    # for k in range(len(phi)):
    #     if np.isnan(phi[k]):
    #         print '@'*6, x[k]

    if not isinstance(Period, np.ndarray):
        Period = np.array(Period)

    #np.mod((mjd-a['MJD0']), Period)/Period

    # -- assumes that values <1 are phases, not MJD:
    phi[np.where(mjd<1)] = mjd[np.where(mjd<1)]

    # -- acceleration in cm/s2 (comparable with logg)
    # -- numerical derivation
    acc = 100*1000*np.gradient(Vpuls)/(np.gradient(phi_intern)*3600*24.*Period.mean())

    # if not useFourierVpuls:
    #     # -- numerical derivation
    #     acc = 100*1000*np.gradient(Vpuls)/(np.gradient(phi_intern)*3600*24.*Period.mean())
    # else:
    #     # -- Fourier derivation
    #     acc = 0.0*phi_intern
    #     for k in Ns:
    #         acc += -a['VPULS A'+str(k)]*2*np.pi*k*\
    #             np.sin(2*np.pi*k*phi_intern + a['VPULS PHI'+str(k)])
    #     acc *= 100*1000/(3600*24.*Period.mean())

    # -- apparent diam, in mas
    P_s = Period.mean()*24*3600. # period in seconds
    d_m = a['d_kpc']*1000*C_pc # distance in meters
    c_v = 1000. # integration cste: m/s

    # -- numerical integration
    Diam = -phi_intern.ptp()*2*P_s/d_m*np.cumsum(Vpuls-Vgamma)*c_v/C_mas/float(len(phi_intern))
    if 'DIAM0' in a.keys():
        Diam += a['DIAM0']
    elif 'DIAMAVG' in a.keys():
        Diam -= Diam[:-1].mean() - a['DIAMAVG']
    # else:
    #     # -- Fourier Integral
    #     Diam = a['DIAM0']+0*phi_intern
    #     for k in Ns:
    #         Diam -= c_v*2*P_s/d_m/C_mas*a['VPULS A'+str(k)]/(2*np.pi*k)*\
    #             (np.sin(2*np.pi*k*phi_intern + a['VPULS PHI'+str(k)])-
    #              np.sin(a['VPULS PHI'+str(k)]))

    # -- radius in km
    linRadius = 0.5*Diam*C_mas*d_m
    expectedRadius = 10**(0.75*np.log10(Period.mean())+1.10)
    if verbose:
        print 'AVG ANGULAR DIAM:', round(Diam.mean(),4), 'mas'
        print 'AVG RADIUS:', round(linRadius.mean()/C_Rsol,3), 'Rsol',
        print 'log(R)=', round(np.log10(linRadius.mean()/C_Rsol),2)
        print 'exp. RADIUS:', round(expectedRadius,2),
        print 'Rsol (P-R from Molinaro et al. 2012)'
    fits_model['AVG_ANGDIAM'] = (round(Diam.mean(),4), 'mas')
    fits_model['AVG_RADIUS'] = (round(linRadius.mean()/C_Rsol,3), 'Rsol')

    # M(R) Bono et al. ApJ 563-319 (2001)
    mass_pr = 10**(-(np.log10(Period.mean()) + 1.70 -
                   1.86*np.log10(linRadius.mean()/C_Rsol))/.9)
    mass_r = 10**(-0.03+0.48*np.log10(linRadius.mean()/C_Rsol))
    mass_r2 = 10**(-0.09+0.48*np.log10(linRadius.mean()/C_Rsol))

    if verbose:
        print 'exp. MASS: %5.2f'%round(mass_pr,2),\
            'Msol (PMR from Bono et al. 2001)'
        print '           %5.2f'%round(mass_r,2),\
            'Msol (evol MR from Bono et al. 2001)'
        print '           %5.2f'%round(mass_r2,2),\
            'Msol (puls MR from Bono et al. 2001)'

    fits_model['ASSUMED_MASS'] = (round(mass_pr,3), 'Msol, PMR Bono+ ApJ 563-319 (2001)')

    # -- logg in cm/s2 (hence the 100*)
    # -- only takes into account gravity
    logg = np.log10(100*C_G*mass_pr*C_Msol/linRadius**2)

    # -- effective logg, taking into account the pulsation
    eff_logg = np.log10(np.maximum(10**logg-acc, 10**0))
    # -- mass leading to a free fall at maximum contraction:

    if verbose:
        print 'acc (cm/s2)   from %6.1f (phi=%6.4f) to %6.1f (phi=%6.4f)'%(acc.min(),
                                                                           phi_intern[acc.argmin()]%1,
                                                                           acc.max(),
                                                                           phi_intern[acc.argmax()]%1)
        print 'grav (cm/s2)  from %6.1f (phi=%6.4f) to %6.1f (phi=%6.4f), AVG: %6.1f'%(
            10**logg.min(), phi_intern[logg.argmin()]%1,
            10**logg.max(), phi_intern[logg.argmax()]%1,
            10**logg.mean())
        tmp = 10**logg+acc
        print 'tot. (cm/s2)  from %6.1f (phi=%6.4f) to %6.1f (phi=%6.4f), AVG: %6.1f'%(
            tmp.min(), phi_intern[tmp.argmin()]%1,
            tmp.max(), phi_intern[tmp.argmax()]%1,
            tmp.mean())
        print 'logg          from %6.3f              to %6.3f             , AVG: %6.3f'%(
            logg.min(), logg.max(), logg.mean())

    # -- interpolation
    iVpuls = lambda x: interp1d(phi_intern, Vpuls-Vgamma, kind='linear', assume_sorted=1)(x%1)
    iVrad = lambda x: interp1d(phi_intern, (Vpuls-Vgamma)/pfactor, kind='linear', assume_sorted=1)(x%1)

    # -- p is a funtion of the pulsating velocity
    if a.has_key('P-FACTOR LIN'):
        pfactor = a['P-FACTOR']*(1+np.abs(Vpuls-Vgamma)/
                                        np.abs(Vpuls-Vgamma).max()*a['P-FACTOR LIN']-
                                                                    a['P-FACTOR LIN'])
        if verbose:
            print 'p-factor: %5.3f (Vpuls=%3.1fkm/s) -> %5.3f (Vpuls=%3.1fkm/s)'%(
                    pfactor.min(), Vpuls[pfactor.argmin()]-Vgamma,
                    pfactor.max(), Vpuls[pfactor.argmax()]-Vgamma)
        iVrad = lambda x: interp1d(phi_intern, (Vpuls-Vgamma)/pfactor,
                               kind='linear', assume_sorted=1)(x%1)
    iRadius = lambda x: interp1d(phi_intern, linRadius, kind='linear', assume_sorted=1)(x%1)
    iDiam = lambda x: interp1d(phi_intern, Diam, kind='linear', assume_sorted=1)(x%1)
    # -- this *DOES NOT* take into account the pulsation
    ilogg_m = lambda x: interp1d(phi_intern, logg, kind='linear', assume_sorted=1)(x%1)
    # -- this *does* take into account the pulsation
    #ilogg_m = lambda x: interp1d(phi_intern, eff_logg, kind='linear')(x%1)

    useSplineTeff, _troll = False, 0.0
    useSplineLum, _vroll = False, 0.0
    xt_pow = None
    if any([k.startswith('TEFF') for k in a.keys()]):
        if any(['TEFF VAL' in k for k in a.keys()]):
            # -- using splines:
            useSplineTeff=True
            # -- get the nodes
            nteff = len(filter(lambda x: 'TEFF VAL' in x, a.keys()))
            nx = nteff
            if nteff == 1: # only val0 is defined
                ndteff = len(filter(lambda x: 'TEFF DVAL' in x, a.keys()))
                nx += ndteff
            else:
                ndteff = 0
            nphi = len(filter(lambda x: 'TEFF PHI' in x, a.keys()))
            if nphi < nteff or nphi < ndteff-1:
                # -- if comb is not defined, use the same one as in Vpuls
                if not 'TEFF PHI0' in a.keys() and 'VPULS PHI0' in a.keys():
                    a['TEFF PHI0'] = a['VPULS PHI0']
                if not 'TEFF POW' in a.keys() and 'VPULS POW' in a.keys():
                    a['TEFF POW'] = a['VPULS POW']
                if not 'TEFF PHI0' in a.keys() and 'VRAD PHI0' in a.keys():
                    a['TEFF PHI0'] = a['VRAD PHI0']
                if not 'TEFF POW' in a.keys() and 'VRAD POW' in a.keys():
                    a['TEFF POW'] = a['VRAD POW']

                # -- nodes
                #x0 = np.linspace(-.5, .5, nteff)
                #x0 = np.sign(x0)*(np.abs(x0)**a['TEFF POW'])+a['TEFF PHI0']
                # -- nodes
                x0 = np.linspace(-1., 1., nx+1)
                x0 = 0.5*np.sign(x0)*(np.abs(x0)**a['TEFF POW']) + a['TEFF PHI0']
                x0 = x0[:-1]
                xt_pow = x0
                xt_i = range(nx)
            else:
                # -- create list of nodes:
                x0 = np.array([a['TEFF PHI'+str(k)] for k in range(nx)])%1

            y0 = [a['TEFF VAL0']]
            if nteff > 1:
                # -- original
                y0.extend([a['TEFF VAL'+str(k+1)] for k in range(nteff-1)])
            else:
                y0.extend([a['TEFF DVAL'+str(k+1)] + a['TEFF VAL0']
                        for k in range(ndteff)])

            xpT = []
            xpT.extend(list(x0-2))
            xpT.extend(list(x0-1))
            xpT.extend(list(x0))
            xpT.extend(list(x0+1))
            xpT.extend(list(x0+2))

            ypT = list(y0)*5

            xpT = np.array(xpT)
            ypT = np.array(ypT)[xpT.argsort()]
            xpT = xpT[xpT.argsort()]

            if not uncer is None:
                ex0 = [uncer['TEFF PHI'+str(k)] for k in range(nteff)]
                ey0 = [uncer['TEFF VAL'+str(k)] for k in range(nteff)]
                expT = list(ex0)
                expT.extend(list(ex0))
                expT.extend(list(ex0))

                eypT = list(ey0)
                eypT.extend(list(ey0))
                eypT.extend(list(ey0))

                expT = np.array(expT)
                eypT = np.array(eypT)[expT.argsort()]
                expT = expT[expT.argsort()]

            # -- spline interpolation
            if 'TEFF KIND' in a:
                _kind = a['TEFF KIND']
            else:
                _kind='cubic'
                fits_model['TEFF SPLINE KIND'] = _kind
            if 'TEFF PHIR' in a.keys():
                _troll = a['TEFF PHIR']

            Teff = interp1d(xpT, ypT,
                            kind=_kind,
                            assume_sorted=1,
                            bounds_error=True,
                            fill_value=0.0)(phi_intern)
            Teff = np.minimum(Teff, maxTeff)
            Teff = np.maximum(Teff, minTeff)
        elif 'TEFF A0' in a.keys():
            #-- Fourier coefficients:
            useSplineTeff=False

            Teff = np.zeros(Nintern-1)
            xteff = phi_intern

            Ns = [int(k.split('I')[1]) for k in filter(lambda x: 'TEFF PHI' in x, a.keys())]
            if 'TEFF A0' in a:
                Teff += a['TEFF A0']
            if any('TEFF R' in k for k in a.keys()):
                Teff +=  a['TEFF A1']*np.cos(2*np.pi*xteff + a['TEFF PHI1'])
                Ns.remove(1)
                for k in Ns:
                    Teff += a['TEFF A1']*a['TEFF R'+str(k)]*\
                            np.cos(2*np.pi*k*xteff + a['TEFF PHI'+str(k)]-a['TEFF PHI1'])
            else:
                for k in Ns:
                    Teff += a['TEFF A'+str(k)]*\
                            np.cos(2*np.pi*k*xteff + a['TEFF PHI'+str(k)])

        else:
            useSplineTeff=False
            # -- Luminosity as linear function of Vpuls (seems to work!)
            Ns = [int(k.split('S')[1]) for k in filter(lambda x: 'TEFF S' in x, a.keys())]
            _L = 0.0
            for k in Ns:
                _L += (Vpuls-Vgamma)**k*a['TEFF S'+str(k)]
            Teff = C_Teffsol*(_L/(linRadius/C_Rsol)**2)**0.25
        # -- luminosity, in Lsol
        Lum = (linRadius/C_Rsol)**2*(Teff/C_Teffsol)**4
    else:
        # -- no Teff is defined, assume model parametrized in LUMINOSITY instead?
        Lum = None
        if any(['LUM VAL' in k for k in a.keys()]):
            # -- using splines:
            useSplineLum=True
            # -- get the nodes
            nteff = len(filter(lambda x: 'LUM PHI' in x, a.keys()))
            x0 = np.array([a['LUM PHI'+str(k)] for k in range(nteff)])%1
            y0 = [a['LUM VAL'+str(k)] for k in range(nteff)]

            xpT = list(x0-3)
            xpT.extend(list(x0-2))
            xpT.extend(list(x0-1))
            xpT.extend(list(x0))
            xpT.extend(list(x0+1))
            xpT.extend(list(x0+2))
            xpT.extend(list(x0+3))

            ypT = list(y0)*7

            xpT = np.array(xpT)
            ypT = np.array(ypT)[xpT.argsort()]
            xpT = xpT[xpT.argsort()]

            if not uncer is None:
                ex0 = [uncer['LUM PHI'+str(k)] for k in range(nteff)]
                ey0 = [uncer['LUM VAL'+str(k)] for k in range(nteff)]
                expT = list(ex0)
                expT.extend(list(ex0))
                expT.extend(list(ex0))

                eypT = list(ey0)
                eypT.extend(list(ey0))
                eypT.extend(list(ey0))

                expT = np.array(expT)
                eypT = np.array(eypT)[expT.argsort()]
                expT = expT[expT.argsort()]

            # -- spline interpolation
            if 'LUM KIND' in a:
                _kind = a['LUM KIND']
            else:
                _kind='quadratic'
                fits_model['LUM SPLINE KIND'] = 'quadratic'
            Lum = interp1d(xpT, ypT,
                            kind=_kind,
                            assume_sorted=1,
                            bounds_error=False,
                            fill_value=0.0)(np.mod(phi_intern, 1.))

        elif 'LUM A0' in a.keys() or 'LUM A1' in a.keys():
            #-- Fourier coefficients:
            useSplineLum=False
            Ns = [int(k.split('I')[1]) for k in filter(lambda x: 'LUM PHI' in x, a.keys())]
            Lum = np.zeros(Nintern-1)
            if 'LUM A0' in a:
                Lum += a['LUM A0']
            for k in Ns:
                Lum += np.array(a['LUM A'+str(k)])*\
                        np.cos(2*np.pi*k*phi_intern + a['LUM PHI'+str(k)])
            Teff = C_Teffsol*(Lum/(linRadius/C_Rsol)**2)**(0.25)


    if 'LUM SLOPE' in a.keys(): # assumes at minimum LUM A0 was set
        _v = (Vpuls-Vgamma)
        origTeff = Teff.copy()
        Lum += a['LUM SLOPE']*np.sign(_v)*np.abs(_v)
        Teff = C_Teffsol*(Lum/(linRadius/C_Rsol)**2)**(0.25)

    # -- interpolated effective temperature
    iTeff = lambda x: interp1d(phi_intern, Teff, kind='linear', assume_sorted=1)((x-_troll+1)%1+_troll)
    iLum = lambda x: interp1d(phi_intern, Lum, kind='linear', assume_sorted=1)(x%1)
    # -- compute the phase offset so the model has max luminosity
    # -- for phi==0
    w = (phi_intern>=0)*(phi_intern<1)
    phaseOffset = (1-phi_intern[w][Lum[w].argmax()]+0.5)%1-0.5

    if verbose:
        print 'Radius (Rsol) from %6.2f (phi=%6.4f) to %6.2f (phi=%6.4f), AVG: %6.2f'%(
                                        (linRadius/C_Rsol).min(), phi_intern[linRadius.argmin()]%1,
                                        (linRadius/C_Rsol).max(), phi_intern[linRadius.argmax()]%1,
                                        (linRadius/C_Rsol).mean())
        print 'Teff (K)      from %6.0f (phi=%6.4f) to %6.0f (phi=%6.4f), AVG: %5.0f'%(
                                        Teff[w].min(), phi_intern[Teff[w].argmin()]%1,
                                        Teff[w].max(), phi_intern[Teff[w].argmax()]%1,
                                        Teff[w].mean())
        print 'Lum (Lsol)    from %6.0f (phi=%6.4f) to %6.0f (phi=%6.4f), AVG: %5.0f'%(
        Lum[w].min(), (phi_intern[w][Lum[w].argmin()]+1)%1,
        Lum[w].max(), (phi_intern[w][Lum[w].argmax()]+1)%1,
        Lum[w].mean())
        print 'PHASE OFFSET ==', phaseOffset

    fits_model['AVG_TEFF'] = (round(Teff[w].mean(), 2), 'K')
    fits_model['AVG_LUM'] = (round(Lum[w].mean(), 2), 'Lsol')
    fits_model['AVG_LOGG'] = (round(logg[w].mean(), 3), 'cgs')

    # -- absolute bolometric magnitude
    Mbol = -2.5*np.log10(Lum)+4.74
    # -- apparent bolometric magnitude
    apparentMbol = Mbol + 5*np.log10(a['d_kpc']*1000)

    if verbose:
        print 'Mbol from %5.3f to %5.3f, AVG: %5.3f'%(
        Mbol[w].min(), Mbol[w].max(), Mbol[w].mean())

        if verbose and not xv_pow is None:
            print 'VRAD/VPULS Spline Comb  :', np.round(xv_pow, 4)
            print 'VRAD/VPULS Spline values:', np.round(ypV[:len(xv_pow)], 2)

        if verbose and not xt_pow is None:
            print 'TEFF       Spline Comb  :', np.round(xt_pow, 4)
            print 'TEFF       Spline values:', np.round(ypT[:len(xt_pow)], 1)

    fits_model['AVG_MBOL'] = round(Mbol[w].mean(), 3)

    # compute results table
    res = np.zeros(len(mjd))
    res = []
    list_filt = []
    list_color = []
    figures = []

    # spectroscopy additional parameters:
    if a.has_key('Vspec'):
        vspec=a['Vspec']
    else:
        vspec = 0.0
    if a.has_key('Aspec'):
        aspec=a['Aspec']
    else:
        aspec = 1.0

    if a.has_key('Rspec'):
        Rspec = lambda x: a['Rspec']
    elif 'Rspec PHI0' in a.keys():
        nr = len(filter(lambda x: 'Rspec PHI' in x, a.keys()))
        x0 = np.array([a['Rspec PHI'+str(k)] for k in range(nr)])%1
        y0 = [a['Rspec VAL'+str(k)] for k in range(nr)]
        xpR = list(x0-1)
        xpR.extend(list(x0))
        xpR.extend(list(x0+1))

        ypR = list(y0)
        ypR.extend(list(y0))
        ypR.extend(list(y0))

        xpR = np.array(xpR)
        ypR = np.array(ypR)[xpR.argsort()]
        xpR = xpR[xpR.argsort()]
        # spline interpolation
        Rspec = lambda x: interp1d(xpR, ypR, kind='quadratic',
                                   bounds_error=False, assume_sorted=1,
                                   fill_value=ypR.mean())(x%1)
    elif a.has_key('Rspec0'):
        Rspec = lambda x: a['Rspec0'] - iVpuls(x)/a['Rv0']
    else:
        Rspec = lambda x: 50

    # -- set default companion if absent:
    if not a.has_key('COMP DIAM'):
        if a.has_key('COMP TEFF'):
            # -- assumes a main sequence (Allen)
            # -- Radius in Rsol:
            __r = [0.85, 0.92, 1.1,  1.3,  1.5,  1.7,  2.4,
                    3.0,   3.9,   4.8,   7.4,   8.5]
            # -- Teff in K
            __t = [5150, 5560, 5940, 6650, 7300, 8180, 9790,
                    11100, 15200, 17600, 30000, 35000]
            r0 = np.interp(np.abs(a['COMP TEFF']), __t, __r)
            COMP_DIAM = r0*iDiam(np.linspace(0,1,100)).mean()/\
                    (linRadius.mean()/C_Rsol)
            if verbose:
                print 'companion radius (main sequence):', round(r0, 3), 'Rsol'
                print 'companion angular diameter:', round(COMP_DIAM,4), 'mas'
        else:
            COMP_DIAM = 1e-6*iDiam(np.linspace(0,1,100)).mean()
            if np.isnan(COMP_DIAM):
                COMP_DIAM=1e-6
    else:
        COMP_DIAM = np.abs(a['COMP DIAM'])*iDiam(np.linspace(0,1,100)).mean()
        if verbose:
            print 'companion diameter:', round(COMP_DIAM,4), 'mas'
            print 'companion radius:',
            print round(np.abs(a['COMP DIAM'])*linRadius.mean()/C_Rsol,3), 'Rsol'

    if not a.has_key('COMP TEFF'):
        COMP_TEFF = 10000
    else:
        COMP_TEFF = a['COMP TEFF']
    if not a.has_key('COMP LOGG'):
        COMP_LOGG = 4.5 # assumes main sequence
    else:
        COMP_LOGG = a['COMP LOGG']

    if not plot:
        # NOTE: plot=False will compute all observables one after the other,
        # This is used by modelM to compute in parallel, calling the function
        # 'model' with plot=False.
        for k, obs in enumerate(x):
            if obs[1].split(';')[0]=='vpuls': # includes Vgamma
                res.append(iVpuls(phi[k]))
            elif obs[1].split(';')[0]=='vrad':
                res.append(iVrad(phi[k]) + Vgamma)
            elif obs[1].split(';')[0]=='vgamma':
                res.append(Vgamma)
            elif obs[1].split(';')[0]=='diam':
                res.append(float(iDiam(phi[k])))
                if  len(obs)>2 and ((isinstance(obs[2], list) or
                                     isinstance(obs[2], tuple)) and
                                    obs[2][0]>=1.9 and obs[2][0]<=2.5):
                    # -- correct for K excess!
                    res[-1] *= diamBiasK(res[-1], obs[2][1], 10**(k_excess/2.5)-1,
                                         RshellRstar)
                if  len(obs)>2 and ((isinstance(obs[2], list) or
                                     isinstance(obs[2], tuple)) and
                                    obs[2][0]>=1.4 and obs[2][0]<1.9):
                    # -- correct for H excess!
                    #res[-1] *= diamBiasK(res[-1], obs[2][1], 10**(h_excess/2.5)-1,
                    #                     RshellRstar)
                    pass

            elif obs[1].split(';')[0]=='UDdiam':
                try:
                    if _ldCoef=='PHOEBE':
                        res.append(float(iDiam(phi[k])*
                                   ldphoebe.UD_LD_c1c2c3c4(obs[2],
                                                           iTeff(phi[k]),
                                                           ilogg_m(phi[k]))))
                    elif _ldCoef=='ATLAS9':
                        res.append(float(iDiam(phi[k])*
                                atlas9_cld.UDLD(obs, iDiam(phi[k]),
                                                iTeff(phi[k]))))
                    elif _ldCoef=='NEILSON':
                        res.append(float(iDiam(phi[k])*
                                neilson_cld.UDLD(obs, iDiam(phi[k]),
                                                iTeff(phi[k]))))
                    elif _ldCoef=='SATLAS':
                        res.append(float(iDiam(phi[k])*
                                ldsatlas.UDLD(obs, iDiam(phi[k]),
                                                iTeff(phi[k]))))
                    else:
                        res.append(0.0)
                except:
                    print obs
                    res.append(0)
                    print 'LD ERROR!'

                if  (isinstance(obs[2], list) or isinstance(obs[2], tuple)) and \
                        obs[2][0]>=1.9 and obs[2][0]<=2.5:
                    res[-1] *= diamBiasK(res[-1], obs[2][1], 10**(k_excess/2.5)-1,
                                         RshellRstar)
                if  (isinstance(obs[2], list) or isinstance(obs[2], tuple)) and \
                        obs[2][0]>=1.3 and obs[2][0]<1.9:
                    #print 'correct UDdiam for H EXCESS',
                    #res[-1] *= diamBiasK(res[-1], obs[2][1], 10**(h_excess/2.5)-1)
                    pass
            elif obs[1].split(';')[0]=='Ravg':
                res.append(expectedRadius)
            elif obs[1].split(';')[0]=='logg':
                res.append(ilogg_m(phi[k]))
            elif obs[1].split(';')[0]=='teff':
                if phi[k] is None:
                    _x = np.linspace(0,1,50)[::-1]
                    _y = iTeff(_x)
                    if obs[-2]<=min(_y):
                        res.append(min(_y))
                    elif obs[-2]>max(_y):
                        res.append(max(_y))
                    else:
                        res.append(obs[-2])
                else:
                    res.append(float(iTeff((phi[k]-teff_phase_offset)%1.0)))

            elif obs[1].split(';')[0]=='normalized spectrum':
                print '--- normalized spectrum...',
                nAir = n_air_P_T(obs[2])
                res.append(vpuls_phoenix.spectrum(obs[2]*nAir,
                    {'VPULS':iVpuls(phi[k]), 'TEFF':iTeff(phi[k]),
                    'VGAMMA':Vgamma+vspec,'LOGG':ilogg_m(phi[k]),
                    'RSPEC':Rspec(phi[k])*1000, 'NORMALIZED':True})**aspec)
                print 'OK'
            elif obs[1].split(';')[0]=='Mbol':
                res.append(np.interp(phi[k], phi_intern,Mbol))
            elif obs[1].split(';')[0]=='mag' or obs[1].split(';')[0]=='flux':
                _jy = obs[1].split(';')[0]=='flux'
                def _tmp(x):
                    _res = photometrySED([phot_corr*iDiam(x), COMP_DIAM],
                                          [iTeff(x), COMP_TEFF],
                                          obs[2], metal=a['METAL'],
                                          logg=[ilogg_m(x), COMP_LOGG], jy=_jy)[0][0]
                    wl = photfilt2.effWavelength_um(obs[2])
                    if not f_excess is None:
                        if not _jy:
                            _res -= f_excess(wl)
                        else:
                            _res *= 10**(f_excess(wl)/2.5)

                    if __monochromaticAlambda:
                        Albda = Alambda_Exctinction(wl, EB_V=a['E(B-V)'], Rv=Rv)
                    else:
                        Albda = Alambda_Exctinction(obs[2], EB_V=a['E(B-V)'],
                                                       Rv=Rv, Teff=iTeff(x))
                    if not _jy:
                        _res += Albda
                    else:
                        _res /= 10**(Albda/2.5)
                    return _res

                if np.isnan(phi[k]) :
                    wei = 1.
                    # -- unknown phase: can be any phase
                    _y = [_tmp(_x) for _x in np.linspace(0,1,40)[:-1]]
                    if obs[-2] >= max(_y): # compare range to measurement
                        res.append(wei*max(_y)+(1-wei)*np.mean(_y))
                    elif obs[-2] <= min(_y):
                        res.append(wei*min(_y)+(1-wei)*np.mean(_y))
                    else:
                        # average between actual value and mean
                        res.append(wei*obs[-2]+(1-wei)*np.mean(_y))
                else:
                    res.append(_tmp(phi[k]))

                #-- keep a list of all filters encoutered
                if not ('_' if _jy else '')+obs[2] in list_filt:
                    list_filt.append(('_' if _jy else '')+obs[2])

            elif obs[1].split(';')[0]=='color':
                res.append(photometrySED([iDiam(phi[k]), COMP_DIAM],
                                            [iTeff(phi[k]), COMP_TEFF],
                                            obs[2].split('-')[0],metal=a['METAL'],
                                            logg=[ilogg_m(phi[k]),COMP_LOGG])[0][0]-
                           photometrySED([iDiam(phi[k]), COMP_DIAM],
                                            [iTeff(phi[k]), COMP_TEFF],
                                            obs[2].split('-')[1],metal=a['METAL'],
                                            logg=[ilogg_m(phi[k]), COMP_LOGG])[0][0])
                wl0 = photfilt2.effWavelength_um(obs[2].split('-')[0])
                wl1 = photfilt2.effWavelength_um(obs[2].split('-')[1])
                if __monochromaticAlambda:
                    res[-1] += Alambda_Exctinction(wl0, EB_V=a['E(B-V)'], Rv=Rv) -\
                               Alambda_Exctinction(wl1, EB_V=a['E(B-V)'], Rv=Rv)
                else:
                    res[-1] += Alambda_Exctinction(obs[2].split('-')[0], EB_V=a['E(B-V)'],
                                                   Rv=Rv, Teff=iTeff(phi[k])) -\
                                Alambda_Exctinction(obs[2].split('-')[1], EB_V=a['E(B-V)'],
                                                    Rv=Rv, Teff=iTeff(phi[k]))

                if not f_excess is None:
                    res[-1] -= f_excess(wl0) - f_excess(wl1)

                # keep a list of all filters encoutered
                if not obs[2] in list_color:
                    list_color.append(obs[2])
            elif obs[1].split(';')[0]=='E(B-V)':
                res.append(a['E(B-V)'])
            else:
                print '\033[41mWARNING: unknown request:', obs[1].split(';')[0], '\033[0m'
        #if len(list_filt)>0:
        #  print 'LIST_FILT:', list_filt

        # -- check for mag offsets:
        if any(['dMAG ' in k for k in a.keys()]):
            for k in a.keys():
                if 'dMAG ' in k:
                    #print k
                    for i,o in enumerate(x):
                       if 'mag' in o[1] and o[2] in k:
                           res[i] += a[k]
                       elif 'color' in o[1] and o[2].split('-')[0] in k:
                           res[i] += a[k]
                       elif 'color' in o[1] and o[2].split('-')[1] in k:
                           res[i] -= a[k]
        if not all(np.isfinite(res)):
            print "WARNING! nan or infinites detected in model's result",
            #print set([x[k][1] for k in range(len(x)) if not np.isfinite(res[k])])
        if not exportFits:
            return res
    #-- compute observables to plot later. use parallel version
    list_filt = []
    list_flux = []
    list_color = []
    for k, obs in enumerate(x):
        if obs[1].split(';')[0]=='mag':
            if not obs[2] in list_filt:
                list_filt.append(obs[2])
        if obs[1].split(';')[0]=='flux':
            if not obs[2] in list_filt:
                list_flux.append(obs[2])
        if obs[1].split(';')[0]=='color':
            if not obs[2] in list_color:
                list_color.append(obs[2])

    list_all_filt = list(list_filt)
    list_all_filt.extend([l.split('-')[0].strip() for l in list_color])
    list_all_filt.extend([l.split('-')[1].strip() for l in list_color])
    list_all_filt=list(set(list_all_filt))

    ### sort by effective wavelength:
    wl_filt = np.array([photfilt2.effWavelength_um(l) for l in list_all_filt])
    list_all_filt = np.array(list_all_filt)[wl_filt.argsort()]
    wl_filt = np.array([photfilt2.effWavelength_um(l) for l in list_filt])
    list_filt = np.array(list_filt)[wl_filt.argsort()]

    wl_color = [0.5*(photfilt2.effWavelength_um(f.split('-')[0])+
                     photfilt2.effWavelength_um(f.split('-')[1])) for f in list_color]
    list_color = np.array(list_color)[np.argsort(wl_color)]

    if verbose:
        print '-'*82
        print 'Photometric Zero Points and redenning:'
        print '  - A_lambda for Rv=%4.2f, Teff=4500, 5500, 6500K'%(Rv)
        print '  - unred Mag for 1mas diameter, Teff=4500K, 5500K, 6500K, logg=1.5:'
    for l in list_all_filt:
        if __monochromaticAlambda:
            al = (Alambda_Exctinction(photfilt2.effWavelength_um(l),EB_V=1.0, Rv=Rv),
                  Alambda_Exctinction(photfilt2.effWavelength_um(l),EB_V=1.0, Rv=Rv),
                  Alambda_Exctinction(photfilt2.effWavelength_um(l),EB_V=1.0, Rv=Rv))
        else:
            al = (Alambda_Exctinction(l, EB_V=1.0, Rv=Rv, Teff=4500),
                  Alambda_Exctinction(l, EB_V=1.0, Rv=Rv, Teff=5500),
                  Alambda_Exctinction(l, EB_V=1.0, Rv=Rv, Teff=6500))
        if verbose:
            print ' %-18s  %6.3f[um] %5.4e[W/m2/um] Al=%5.3f, %5.3f, %5.3f Mag=%6.3f, %6.3f, %6.3f'%(l,
                photfilt2.effWavelength_um(l), photfilt2.zeroPoint_Wm2um(l),
                al[0], al[1], al[2],
                photometrySED(1.0, 4500.0, l, logg=1.5, metal=a['METAL']),
                photometrySED(1.0, 5500.0, l, logg=1.5, metal=a['METAL']),
                photometrySED(1.0, 6500.0, l, logg=1.5, metal=a['METAL']),)

        fits_model['ZP_WM2UM '+l] = photfilt2.zeroPoint_Wm2um(l)
        fits_model['MAG_1MAS 4500K_LOGG1.5 '+l] = round(photometrySED(1.0, 4500.0, l, logg=1.5, metal=a['METAL'])[0][0], 3)
        fits_model['MAG_1MAS 5500K_LOGG1.5 '+l] = round(photometrySED(1.0, 5500.0, l, logg=1.5, metal=a['METAL'])[0][0], 3)
        fits_model['MAG_1MAS 6500K_LOGG1.5 '+l] = round(photometrySED(1.0, 6500.0, l, logg=1.5, metal=a['METAL'])[0][0], 3)
        fits_model['ALAMBDA 4500K '+l] = round(al[0], 5)
        fits_model['ALAMBDA 5500K '+l] = round(al[1], 5)
        fits_model['ALAMBDA 6500K '+l] = round(al[2], 5)

    # if verbose:
    #    print '-'*20, 'Computing observables:', '-'*20
    # res = model(x,a, plot=False) # much slower (single processor) but useful for debug

    if verbose:
         print '-'*20, 'Computing observables on multi-cores:', '-'*20
    res = modelM(x, a, maxCores=maxCores) # much faster (multi processor)

    chi2 = 0.0
    for k in range(len(res)):
        tmp = (res[k] - x[k][-2])**2/x[k][-1]**2
        #if res[k]==0:
        #    print x[k]
        if isinstance(tmp, np.ndarray):
            chi2 += np.mean(tmp)
        else:
            chi2 += tmp

    allChi2= [('TOTAL', chi2/len(res))]

    # #############################
    # ###### plot results #########
    # #############################

    # -- data density as a function of MJD:
    if plot and False:
        plt.figure(9)
        plt.clf()
        plt.title('data density')
        _mjd = np.array([o[0] for o in x])
        _d = np.ones(len(x))
        _d, _mjd = _d[_mjd>1], _mjd[_mjd>1]
        _mjd.sort()

        _n = 100
        _d_100 = []
        for m in _mjd:
            _d_100.append( _d[np.abs(_mjd-m)<_n/2.].sum())
        plt.plot(_mjd, _d_100, '.k', label='(+-%d days)'%(_n/2.),
                 linewidth=2)
        _n = 500
        _d_500 = []
        for m in _mjd:
            _d_500.append( _d[np.abs(_mjd-m)<_n/2.].sum())
        plt.plot(_mjd, _d_500, '.r', label='(+-%d days)'%(_n/2.),
                 linewidth=2)

        plt.xlabel('MJD')
        plt.ylabel('number of data points')
        plt.legend(loc='upper left')
    # -----------------------------------------------

    data  = [obs[-2] for obs in x]
    edata = [obs[-1] for obs in x]
    types = np.array([obs[1].split(';')[0] for obs in x])
    orig  = np.array([obs[1].split(';')[1].strip() if ';' in obs[1] else '' for obs in x])

    filtname = []
    for o in x:
        if isinstance(o[2], list) or isinstance(o[2], tuple):
            filtname.append(str(o[2][0]))
        else:
            filtname.append(str(o[2]))
    filtname = np.array(filtname)

    X = np.linspace(-0.5, 1.5, 300)[:-1]
    colors = [(0.6,0.3,0), (0,0.6,0.3), (0.3,0,0.6),
              (0.6,0,0.3), (0.3,0.6,0), (0,0.3,0.6)][::-1]
    #styles = ['.', '+']
    #Bstyles = ['s', 'd', 'p']

    teffColorPalette = lambda t: (np.array(plt.cm.RdYlBu((t-Teff.min())/
        Teff.ptp()))/2.+0.2)[:3]
    teffColorPalette = lambda t: (np.array(plt.cm.get_cmap('rainbow_r')((t-Teff.min())/
        Teff.ptp()))/2+0.2)[:3]

    # -- start FITS data
    fits_data = collections.OrderedDict()
    fits_data['PHASE'] = np.linspace(0,1,1001)[:-1]
    fits_data['Vpuls'] = iVpuls(fits_data['PHASE'])
    fits_data['Vrad'] = iVrad(fits_data['PHASE'])+Vgamma
    fits_data['diam'] = iDiam(fits_data['PHASE'])
    fits_data['R'] = np.interp(fits_data['PHASE'], phi_intern, linRadius/C_Rsol)
    fits_data['Teff'] = iTeff(fits_data['PHASE'])
    fits_data['Lum'] = iLum(fits_data['PHASE'])
    fits_data['logg'] = ilogg_m(fits_data['PHASE'])

    ####### FIRST plot: Vpuls, Diam and Teff ###########
    colorMap = 'jet'
    colorModel = '0.5'

    test1plot = True

    uni = 9312 # circled number
    uni = 9424 # circled small letters

    uni = 97
    def labelPanel(uni, x0=-0.08):
        if uni>255:
            # -- unicode
            plt.text(x0, plt.ylim()[0]+0.05*(plt.ylim()[1]-plt.ylim()[0]),
                 unichr(uni), va='bottom', ha='left', fontweight='bold',
                 fontsize=16, color='0.2')
        else:
            # -- ascii character
            plt.text(x0, plt.ylim()[0]+0.02*(plt.ylim()[1]-plt.ylim()[0]),
                  r'$\circled{'+chr(uni)+'}$', va='bottom', ha='left', fontweight='bold',
                  fontsize=25, color='0.2')

        return uni+1

    if plot:
        fignum  = 10
    elif isinstance(plot, int):
        fignum = plot
    else:
        print '!!!ERROR: plot=True or integer value (figure number)'
        return

    if plot:
        if test1plot:
            figures.append(plt.figure(fignum, figsize=(12,7)))

            plt.clf()
            plt.subplots_adjust(left=0.06, top=0.95, bottom=0.06,
                                right=0.98, hspace=0.02, wspace=0.17)
        else:
            figures.append(plt.figure(fignum, figsize=(6,9.5)))
            plt.clf()
            plt.subplots_adjust(left=0.15, top=0.95, bottom=0.07,
                               right=0.95, hspace=0.01)
    nplot = 3
    if starName is None:
        title = ''
    else:
        title = starName
    if a['d_kpc']>=2:
        title += ' (P~%6.3fd) p=%5.3f d=%5.1fkpc'%(a['PERIOD'], a['P-FACTOR'], a['d_kpc'])
    else:
        title += ' (P~%6.3fd) p=%5.3f d=%5.1fpc'%(a['PERIOD'], a['P-FACTOR'], 1000*a['d_kpc'])
    if 'E(B-V)' in a.keys():
        title += ' E(B-V)=%5.3f'%(a['E(B-V)'])
    if f_excess is None:
        if l_excess>0 :
            title += r' L$_{ex}$=%5.3f'%(l_excess)
        if k_excess>0 :
            title += r' K$_{ex}$=%5.3f'%(k_excess)
        if h_excess:
            title += r' H$_{ex}$=%5.3f'%(h_excess)
        if j_excess:
            title += r' J$_{ex}$=%5.3f'%(j_excess)
    else:
        title += '; '+title_excess

    if plot and title[0] != ' ':
        plt.suptitle(title, fontsize=12, fontweight='bold')

    # ------ radial velocity -------
    if plot:
        if test1plot:
            # ax_e = plt.axes([0.06, 0.50, 0.43, 0.08]) # residuals
            # ax = plt.axes([0.06, 0.58, 0.43, 0.35]) # model and data
            win_ym, win_ys, win_yo, win_e = 0.95, 0.35, 0.01, 0.
            #ax_e = plt.axes([0.06, win_ym-(1.+win_e)*win_ys, 0.43, win_ys*win_e]) # residuals
            ax = plt.axes([0.06, win_ym-win_ys, 0.43, win_ys]) # model and data
        else:
            ax = plt.subplot(nplot, 1, 1)

        plt.hlines(Vgamma, -1,2,color=colors[-1], linestyle='dotted',
                      linewidth=1, label='V$_\gamma$=%4.2f km/s'%Vgamma)

    y_min, y_max = Vgamma, Vgamma
    MJDoutliers = []

    if any(['VRAD 'in k for k in a.keys()]) and not useFourierVpuls:
        j = 0
        for i in range(len(xpV)):
            if xpV[i]>=0 and xpV[i]<1:
                fits_model['VRAD SPLINE NODE PHI'+str(j)] = (round(xpV[i],4), 'phase')
                fits_model['VRAD SPLINE NODE VAL'+str(j)] = (ypV[i], 'km/s')
                j+=1

    if any(['VPULS 'in k for k in a.keys()]) and not useFourierVpuls:
        j = 0
        for i in range(len(xp)):
            if xp[i]>=0 and xp[i]<1:
                fits_model['VPULS SPLINE NODE PHI'+str(j)] = (round(xp[i],4), 'phase')
                fits_model['VPULS SPLINE NODE VAL'+str(j)] = (yp[i], 'km/s')
                j+=1

    w = np.where(types=='vpuls')
    if len(w[0])>0:
        y_min = min(y_min, np.min([data[k]-edata[k] for k in w[0]]))
        y_max = max(y_max, np.max([data[k]+edata[k] for k in w[0]]))
        chi2 = np.mean([((data[k]-res[k])/edata[k])**2 for k in w[0]])
        # -- Vpuls model
        if plot:
            plt.plot(X, iVpuls(X)+Vgamma, color=(0.1,0.5,0.25), linewidth=2,
                        label='Vpuls + V$_\gamma$', alpha=0.5, linestyle='dashed')
        y_min = min(y_min, iVpuls(X).min()+Vgamma)
        y_max = max(y_max, iVpuls(X).max()+Vgamma)

        # -- plot each familly of point with a different color
        tmp = list(set(orig[w]))
        for i,s in enumerate(np.sort(tmp)):
            color = plt.cm.get_cmap(colorMap)(i/float(max(len(tmp)-1,1)))
            color = [0.8*np.sqrt(c) for c in color[:-1]]
            color.append(1)
            _w = np.where(orig[w]==s)
            ### data
            for df in [-1,0,1]:
                if plot:
                    plt.errorbar(phi[w][_w]+df, [data[k] for k in w[0][_w]],
                        yerr=[edata[k] for k in w[0][_w]], linestyle='none',
                        marker='.', alpha=0.5, label=s if df==0 else '',
                        markersize=3, color=color)
        allChi2.append(('VPULS', chi2))

        # -- find outliers:
        w__ = np.where(np.array([np.abs(data[k]-res[k])
                       for k in w[0]])>3.*np.array([edata[k] for k in w[0]]))
        if len(w__[0])>0 and showOutliers:
            print 'outliers in vpuls'
            print [x[w[0][i]][0] for i in w__[0]]
            MJDoutliers.extend([x[w[0][i]][0] for i in w__[0]])
        if plot:
            plt.text(1.05, y_max,
                         r'Vpuls $\chi^2=$%4.2f'%chi2, va='top', ha='right',
                         size=10)
            plt.plot(X, iVpuls(X), '-', label='Vpuls model', linewidth=3, color='0.5')
            plt.plot(X, 0*iVpuls(X), '-', linewidth=2, color='0.5', linestyle='dotted')
        y_min = min(y_min, iVpuls(X).min())
        y_max = max(y_max, iVpuls(X).max())
        # -- plot nodes
        if plot and not uncer is None:
            plt.errorbar(xpV,ypV-Vgamma, fmt='.k', markersize=9,
                            xerr=exp, yerr=eyp)
        if not useFourierVpuls and plot and False:
            plt.plot(xp, yp-Vgamma, 'pw', markersize=9,
                        label='Spline Nodes - V$\gamma$', alpha=0.8)
            if not xv_pow is None:
                color=(0.8,0.4,0.0)
                plt.plot(xv_pow, Vgamma+0*xv_pow, '|', color=color,
                         linewidth=2, markersize=12, label='Spline comb')
                plt.plot(xv_pow-1, Vgamma+0*xv_pow, '|', color=color,
                          linewidth=2, markersize=12)

    w = np.where(types=='vrad')
    if len(w[0])>0:
        y_min = min(y_min, np.min([data[k]-edata[k] for k in w[0]]))
        y_max = max(y_max, np.max([data[k]+edata[k] for k in w[0]]))

        chi2 = np.mean([((data[k]-res[k])/edata[k])**2 for k in w[0]])
        allChi2.append(('VRAD', chi2))
        # -- Vrad model
        if plot:
            plt.plot(X, iVrad(X)+Vgamma, color=(0.1,0.5,0.25), linewidth=2,
                        label='model, ptp=%5.2fkm/s'%np.ptp(iVrad(X)), alpha=0.5)
        if not useFourierVpuls and plot:
            if any(['VRAD 'in k for k in a.keys()]):
                plt.plot(xpV, ypV, 'p', markersize=5, color=(0.5,0.25,0.1),
                        label='Spline Nodes', alpha=0.8)
            else:
                plt.plot(xp, (yp-Vgamma)/pfactor+Vgamma, 'p', markersize=5, color=(0.5,0.25,0.1),
                        label='Spline Nodes', alpha=0.8)

            if not xv_pow is None:
                color=(0.8,0.4,0.)
                plt.plot(xv_pow, Vgamma+0*xv_pow, '|', color=color,
                         linewidth=2, markersize=16, label='Spline comb')

                plt.plot(xv_pow-1, Vgamma+0*xv_pow, '|', color=color,
                          linewidth=2, markersize=16)
                for i in x0_i:
                    if xv_pow[i]<1.1:
                        plt.text(xv_pow[i], Vgamma, '%d'%i, color=color,
                                alpha=0.5, size=8)
                    if xv_pow[i]>0.9:
                        plt.text(xv_pow[i]-1, Vgamma, '%d'%i, color=color,
                                alpha=0.5, size=8)

        tmp = list(set(orig[w]))
        for i,s in enumerate(np.sort(tmp)):
            color = plt.cm.get_cmap(colorMap)(i/float(max(len(tmp)-1,1)))
            color = [0.8*np.sqrt(c) for c in color[:-1]]
            color.append(1)
            _w = np.where(orig[w]==s)
            for df in [-1,0,1]:
                ### data
                if plot:
                    plt.errorbar(phi[w][_w]+df, [data[k] for k in w[0][_w]],
                        yerr=[edata[k] for k in w[0][_w]], linestyle='none',
                        marker='.', label=s if df==0 else '',
                        markersize=3, color=color,
                        alpha=0.2 if any([edata[k]<0 for k in w[0][_w]]) else 1)
                ### residuals:
                # ax_e.plot(phi[w][_w]+df, [(data[k]-res[k])/edata[k] for k in w[0][_w]],
                #     linestyle='none', marker='o', alpha=1, markersize=3,
                #     color=color)

        # -- find outliers:
        w__ = np.where(np.array([np.abs(data[k]-res[k])
                       for k in w[0]])>3.*np.array([edata[k] for k in w[0]]))
        if len(w__[0])>0 and showOutliers:# and showOutliers:
            print 'outliers in vrad'
            print [x[w[0][i]][0] for i in w__[0]]
            MJDoutliers.extend([x[w[0][i]][0] for i in w__[0]])
        if plot:
            plt.text(1.05, y_max+0.10*(y_max-y_min),
                    r'Vrad $\chi^2=$%4.2f'%chi2, va='top', ha='right', size=10)
    if plot:
        plt.legend(loc='upper left', prop={'size':9},
                   frameon=False, numpoints=1, ncol=2)
        plt.ylabel('velocity (km/s)')
        plt.xlim(-0.1,1.1)
        plt.ylim(y_min-0.05*(y_max-y_min), y_max+0.13*(y_max-y_min))
        ax.set_xticklabels([])
        uni = labelPanel(uni)

    if plot and False:
        # -- persistent L/R and L/Vpuls diagrams
        plt.figure(0, figsize=(8,8))
        plt.clf()
        x0 = np.linspace(0,1,100)

        lum = (iTeff(x0)/C_Teffsol)**4*(iRadius(x0)/C_Rsol)**2
        dlum_v = np.gradient(lum)/a['PERIOD']/iVpuls(x0)
        plt.plot(np.log10(iTeff(x0)), np.log10(lum), '.k-')
        plt.plot(np.log10(iTeff(x0).mean()), np.log10(lum.mean()), '*y')

        # -- instalbility strip
        #plt.plot([3.8, 3.75], [3.15, 3.9], '-b')
        #plt.plot([3.735, 3.68], [3.15, 3.9], '-r')
        # -- Fundamental Blue Edge: Saio & Gautschy 1997
        # -- http://iopscience.iop.org/article/10.1086/305544/fulltext/,
        _P = np.logspace(0.5,2,10)
        l10_L = 2.573 + 1.270*np.log10(_P) # eq 3
        l10_T = -0.036*l10_L + 3.925 # eq 1
        plt.plot(l10_T, l10_L, '.b-')

        for _R in [40, 50, 60, 75, 100, 125]:
            l10_P = (np.log10(_R)-1.10)/0.75 # ref ???

            # -- http://iopscience.iop.org/article/10.1086/305544/fulltext/,
            l10_Lblu = 2.573 + 1.270*l10_P # eq 3
            l10_Tblu = l10_Lblu/4. - 2*np.log10(_R)/4.
            l10_Tblu += np.log10(C_Teffsol)
            plt.plot(l10_Tblu, l10_Lblu, 'ob')

            l10_Lred = 2.326 + 1.244*l10_P # eq 4
            l10_Tred = l10_Lred/4. - 2*np.log10(_R)/4.
            l10_Tred += np.log10(C_Teffsol)
            plt.plot(l10_Tred, l10_Lred, 'or')

            _T = np.linspace(10**l10_Tred, 10**l10_Tblu, 10)
            _L = (_T/C_Teffsol)**4*(_R)**2
            plt.plot(np.log10(_T), np.log10(_L), '--k', alpha=0.5)
            plt.text(np.log10(_T)[0], np.log10(_L)[0], '%2.0fRsol'%_R,
                    ha='left', va='center')


        # -- P/R relation
        _P = [5, 10, 20, 30]
        _R = 10**(0.75*np.log10(_P)+1.10)

        print _P, _R
        plt.xlabel('log10 Teff')
        plt.ylabel('log10 Lum/Lsol')
        plt.xlim(3.82, 3.65)
        #w = np.where(np.gradient(iVpuls(x0))>0)
        #plt.plot(iVpuls(x0[w]), lum[w], '^k')
        #w = np.where(np.gradient(iVpuls(x0))<=0)3.
        #plt.plot(iVpuls(x0[w]), lum[w], 'vk')

        #plt.legend(loc='lower left', prop={'size':10})
        #plt.xlabel('Vpuls (km/s)')
        #plt.ylabel('$\delta$ luminosity (Lsol/day)')
        #plt.grid()
        plt.figure(fignum)

    # ----- diameters ------
    if plot:
        if test1plot:
            # ax2_e = plt.axes([0.06, 0.06, 0.43, 0.08])
            # ax2 = plt.axes([0.06, 0.14, 0.43, 0.35])
            #ax2_e = plt.axes([0.06, win_ym-(2.5+2*win_e)*win_ys-2*win_yo, 0.43, win_ys*win_e]) # residuals
            ax2 = plt.axes([0.06, win_ym-(2.5+win_e)*win_ys-2*win_yo, 0.43, win_ys]) # model and data
        else:
            ax2 = plt.subplot(nplot, 1, 2, sharex=ax)

    y_min = iDiam(X).min()
    y_max = iDiam(X).max()
    if iDiam(X).mean()>1.:
        lab = 'model ptp=%5.2fmas (%3.1f%%)'%(np.ptp(iDiam(X)),
                                        100*np.ptp(iDiam(X))/np.mean(iDiam(X)))
    else:
        lab = 'model ptp=%5.2fuas (%3.1f%%)'%(np.ptp(1000*iDiam(X)),
                                        100*np.ptp(iDiam(X))/np.mean(iDiam(X)))
    if plot:
        plt.plot(X, iDiam(X), '-', label=lab,
                linewidth=3, color=colorModel, alpha=0.5)

    # -- color code for the baseline (in m):
    baselineC = lambda x: plt.cm.get_cmap('gist_stern_r')(x/330.)

    w = np.where(types=='diam')
    plotK, plotH = False, False

    if len(w[0])>0:
        y_min = min(y_min, np.min([data[k]-edata[k] for k in w[0]]))
        y_max = max(y_max, np.max([data[k]+edata[k] for k in w[0]]))
        wi = np.where([edata[k]>0 for k in w[0]])
        if len(wi[0])>1:
            chi2 = np.mean([((data[k]-res[k])/edata[k])**2 for k in w[0][wi]])
        else:
            chi2 = 0.0
        #chi2 = np.mean([((data[k]-res[k])/edata[k])**2 for k in w[0]])
        _label1, _label2 = True, True
        for df in [0,-1,1]:
            if k_excess!=0:
                for i in w[0]:
                    # -- model: to show effect of K excess
                    if not np.isscalar(x[i][2]) and np.abs(x[i][2][0]-2.2)<0.3:
                        plotK = True
                        if plot:
                            ax2.errorbar(phi[i]+df, x[i][-2], yerr=x[i][-1],
                                    fmt='p', color=baselineC(x[i][2][1]),
                                    alpha=0.7, markersize=4,
                                    label=r'$\theta$Ross$_K$ $\chi^2$=%3.1f'%chi2 if _label1 else '')
                        if _label1:
                            allChi2.append(('Ross_K', chi2))
                        _label1 = False
                    elif not np.isscalar(x[i][2]) and np.abs(x[i][2][0]-1.65)<0.3:
                        plotH = True
                        if plot:
                            ax2.errorbar(phi[i]+df, x[i][-2], yerr=x[i][-1],
                                    fmt='d', color=baselineC(x[i][2][1]),
                                    alpha=0.7, markersize=4,
                                    label=r'$\theta$Ross$_H$ $\chi^2$=%3.1f'%chi2 if _label1 else '')
                        if _label1:
                            allChi2.append(('Ross_H', chi2))
                        _label1 = False
                    else:
                        if plot:
                            ax2.errorbar(phi[i]+df, x[i][-2], yerr=x[i][-1],
                                    fmt='o', color='0.5',
                                    alpha=0.7, markersize=4,
                                    label=r'$\theta$Ross $\chi^2$=%3.1f'%chi2 if _label2 else '')
                        if _label2:
                            allChi2.append(('LD', chi2))
                        _label2 = False
            else:
                if plot:
                    ax2.errorbar(phi[w]+df, [data[k] for k in w[0]],
                                yerr=[edata[k] for k in w[0]],
                                fmt='.', color=(0.1, 0.3, 0.8), alpha=0.6,
                                label=r'$\theta$Ross $\chi^2$=%3.1f'%chi2 if _label2 else '',
                                markersize=4)
                if _label2:
                    allChi2.append(('Ross', chi2))
                _label2 = False

    w = np.where(types=='UDdiam')
    if len(w[0])>0:
        y_min = min(y_min, np.min([data[k]-edata[k] for k in w[0]]))
        y_max = max(y_max, np.max([data[k]+edata[k] for k in w[0]]))
        filt = set(filtname[w])
        # -- list of colors for each filter
        colors = plt.cm.get_cmap('gnuplot')(np.linspace(0,1,len(filt)+1)[:-1])
        for i, f in enumerate(filt):
            w__ = np.where((types=='UDdiam')*(filtname==f))
            y_min = min(y_min, np.min([data[k]-edata[k] for k in w[0]]))
            y_max = max(y_max, np.max([data[k]+edata[k] for k in w[0]]))
            _label1, _label2 = True, True
            wi = np.where([edata[k]>0 for k in w__[0]])
            if len(wi[0])>1:
                chi2 = np.mean([((data[k]-res[k])/edata[k])**2 for k in w__[0][wi]])
            else:
                chi2 = 0.0
            for df in [0,-1,1]:
                if k_excess!=0 and np.abs(float(f)-2.2)<0.4:
                    plotK = True
                    for i in w__[0]: # for each data point
                        if _ldCoef=='PHOEBE':
                            UD_LD = ldphoebe.UD_LD_c1c2c3c4(x[i][2],
                                                           iTeff(phi[i]),
                                                           ilogg_m(phi[i]))
                        elif _ldCoef=='ATLAS9':
                            UD_LD = atlas9_cld.UDLD(x[i], iDiam(phi[i]), iTeff(phi[i]))
                        elif _ldCoef=='NEILSON':
                            UD_LD = neilson_cld.UDLD(x[i], iDiam(phi[i]), iTeff(phi[i]))
                        elif _ldCoef=='SATLAS':
                            UD_LD = ldsatlas.UDLD(x[i], iDiam(phi[i]), iTeff(phi[i]))

                        if i==w__[0][0] and df==0:
                            print f, 'UD_LD=', UD_LD
                        # -- baseline given and in the K band:
                        if not np.isscalar(x[i][2]) and np.abs(x[i][2][0]-2.2)<0.4:
                            if plot:
                                ax2.errorbar(phi[i]+df, x[i][-2]/UD_LD, yerr=x[i][-1],
                                        fmt='s', color=baselineC(x[i][2][1]),
                                        alpha=0.7, markersize=4,
                                        label=r'UD$_K$->$\theta$Ross $\chi^2=%3.1f$'%chi2 if _label1 else '')
                            if _label1:
                                allChi2.append(('UD_K', chi2))
                            _label1 = False
                        else:
                            if plot:
                                ax2.errorbar(phi[i]+df, x[i][-2]/UD_LD, yerr=x[i][-1],
                                        fmt='s', color='0.5', alpha=0.7, markersize=4,
                                        label=r'UD$_K$->$\theta$Ross $\chi^2=%3.1f$'%chi2 if _label2 else '')
                            if _label2:
                                allChi2.append(('UD_K', chi2))
                            _label2 = False
                elif h_excess!=0 and np.abs(float(f)-1.65)<0.3:
                    plotH = True
                    for i in w__[0]:
                        if _ldCoef=='PHOEBE':
                            UD_LD = ldphoebe.UD_LD_c1c2c3c4(x[i][2],
                                                           iTeff(phi[i]),
                                                           ilogg_m(phi[i]))
                        elif _ldCoef=='ATLAS9':
                            UD_LD = atlas9_cld.UDLD(x[i], iDiam(phi[i]), iTeff(phi[i]))
                        elif _ldCoef=='NEILSON':
                            UD_LD = neilson_cld.UDLD(x[i], iDiam(phi[i]), iTeff(phi[i]))
                        elif _ldCoef=='SATLAS':
                            UD_LD = ldsatlas.UDLD(x[i], iDiam(phi[i]), iTeff(phi[i]))

                        if i==w__[0][0] and df==0:
                            print f, 'UD_Ross=', UD_LD

                        # -- baseline given and in the H band:
                        if not np.isscalar(x[i][2]) and np.abs(x[i][2][0]-1.65)<0.3:
                            if plot:
                                ax2.errorbar(phi[i]+df, x[i][-2]/UD_LD, yerr=x[i][-1],
                                            fmt='d' if x[i][-1]>0 else 'x',
                                            color=baselineC(x[i][2][1]),
                                            alpha=0.7, markersize=4,
                                            label=r'UD$_H$->$\theta$Ross $\chi^2=%3.1f$'%chi2 if _label1 else '')
                            if _label1:
                                #print '  |%-30s:'%('UD_H'), chi2
                                allChi2.append(('UD_H', chi2))
                            _label1 = False
                        else:
                            if plot:
                                ax2.errorbar(phi[i]+df, x[i][-2]/UD_LD, yerr=x[i][-1],
                                            fmt='d' if x[i][-1]>0 else 'x',
                                            color='0.5', alpha=0.7, markersize=4,
                                            label=r'UD$_H$->$\theta$Ross $\chi^2=%3.1f$'%chi2 if _label2 else '')
                            if _label2:
                                #print '  |%-30s:'%('UD_H'), chi2
                                allChi2.append(('UD_H', chi2))
                            _label2 = False
                else:
                    if _ldCoef=='PHOEBE':
                        UD_LD = [ldphoebe.UD_LD_c1c2c3c4(x[k][2],
                                                           iTeff(phi[k]),
                                                           ilogg_m(phi[k])) for k in w__[0]]
                    elif _ldCoef=='ATLAS9':
                        UD_LD = [atlas9_cld.UDLD(x[k], iDiam(phi[k]), iTeff(phi[k]))
                                    for k in w__[0]]
                    elif _ldCoef=='NEILSON':
                        UD_LD = [neilson_cld.UDLD(x[k], iDiam(phi[k]), iTeff(phi[k]))
                                    for k in w__[0]]
                    elif _ldCoef=='SATLAS':
                        UD_LD = [ldsatlas.UDLD(x[k], iDiam(phi[k]), iTeff(phi[k]))
                                    for k in w__[0]]

                    UD_LD = np.array(UD_LD)
                    if df==0:
                        print f, 'UD -> Ross', UD_LD.mean()
                    # -- show data
                    wi = np.where([edata[k]>0 for k in w__[0]])
                    if plot and len(wi[0])>0:
                        ax2.errorbar(phi[w__]+df, np.array([data[k] for k in w__[0]])/UD_LD,
                                yerr=np.array([edata[k] for k in w__[0]])/UD_LD,
                                fmt='.', color=colors[i%len(colors)],
                                markersize=4, alpha=0.7,
                                label=r'UD$_{%s\mu m}$->$\theta$Ross $\chi^2$=%3.1f'%(f, chi2) if df==0 else '')
                    wi = np.where([edata[k]<=0 for k in w__[0]])
                    if plot and len(wi[0])>0:
                        ax2.errorbar(phi[w__]+df, np.array([data[k] for k in w__[0]])/UD_LD,
                                yerr=np.array([edata[k] for k in w__[0]])/UD_LD,
                                fmt='x', color=colors[i%len(colors)],
                                markersize=4, alpha=0.7,
                                label=r'UD$_{%s\mu m}$->$\theta$Ross [IGNORED]'%(f) if df==0 else '')
                    if _label1:
                        #print '  |%-30s:'%('UD %sum'%(f)), chi2
                        allChi2.append(('UD %sum'%(f), chi2))
                    _label1 = False

    # -- plot K excess effects:
    if k_excess!=0 and plotK:
        for b in [100. , 200. , 300.]:
            bias = np.array([diamBiasK(d, b, 10**(k_excess/2.5)-1) for d in iDiam(X)])
            if plot:
                ax2.plot(X, iDiam(X)*bias, '-', linewidth=2, color=baselineC(b),
                         alpha=0.8, linestyle='dashed')
                if np.interp(0.5, X, iDiam(X)*bias)<y_max+ 0.05*(y_max-y_min):
                    ax2.text(0.5, np.interp(0.5, X, iDiam(X)*bias), 'B=%3.0fm (K)'%b,
                             color=baselineC(b), alpha=0.9, size=10, va='center', ha='left')
            fits_data['diamK %3.0fm'%b] = iDiam(fits_data['PHASE'])*\
                        np.interp(fits_data['PHASE'], X, bias)
    # -- plot H excess effects:
    if h_excess!=0 and plotH and False:
        for b in [100. , 200. , 300.]:
            bias = np.array([diamBiasK(d, b, 10**(h_excess/2.5)-1) for d in iDiam(X)])
            if plot:
                ax2.plot(X, iDiam(X)*bias, '-', linewidth=2, color=baselineC(b),
                         alpha=0.8, linestyle='dotted')
                if np.interp(0.5, X, iDiam(X)*bias)<y_max+ 0.05*(y_max-y_min):
                    ax2.text(0.5, np.interp(0.5, X, iDiam(X)*bias), 'B=%3.0fm (H)'%b,
                             color=baselineC(b), alpha=0.9, size=10, va='center', ha='right')
            fits_data['diamH %3.0fm'%b] = iDiam(fits_data['PHASE'])*\
                        np.interp(fits_data['PHASE'], X, bias)

    if plot and not uncer is None:
        ax2.plot(0,a['DIAM0'], 'pw', markersize=10, label='Ross(t=0)')

    if plot:
        ax2.legend(loc='lower center', prop={'size':9}, numpoints=1,
                   frameon=False, ncol=1)

        ax2.set_xlim(-0.1,1.1)
        ax2.set_ylim(y_min - 0.05*(y_max-y_min),
                 y_max + 0.05*(y_max-y_min))
        uni = labelPanel(uni)

        if y_max>0.1:
            ax2.set_ylabel('Ang. diam. (mas)')
        else:
            ax2.set_ylabel('Ang. diam. (uas)')
            yti = ax2.get_yticks()
            ax2.set_yticklabels([str(__yti*1000) for __yti in yti])


    next_plot = 3

    if False: #-- plot logg
        plt.figure(22)
        plt.clf()

        plt.plot(X, 10**ilogg_m(X), color=colorModel, linewidth=3,
                 alpha=0.5, label='static gravity')
        plt.plot(phi_intern, acc, '.r', label='acceleration')
        #plt.plot(phi_intern, -acc, ',r', label='- acceleration', alpha=0.5)

        plt.hlines(10**ilogg_m(X).mean(), -1, 2, color='k',
                   alpha=0.5, linestyle='dashed', label='logg = %4.2f'%(ilogg_m(X).mean()))
        plt.hlines(0, -1,2, color='k', linestyle='dotted')
        plt.hlines((10**ilogg_m(X)).mean(), -1, 2, color='k',
                   alpha=0.5, linestyle='dashed')

        # -- plot nodes:
        plt.ylabel(r'grav (cm/s2)')
        plt.xlim(-0.1, 1.1)
        plt.xlabel('pulsation phase')
        plt.legend(loc='lower center', prop={'size':12}, frameon=False, numpoints=1)
        plt.grid()
        if plot== True:
            plt.figure(10)
        else:
            plt.figure(plot)

    __np = 2
    # -- Teff on the top right
    if (len(list_filt)+len(list_color)+2)%2 ==0:
        __op = 0
    else:
        __op = 1
    subplot = 7

    showLum = False
    # -- Teff on the bottom left:
    if (len(list_filt)+len(list_color)+showLum)%2 ==0:
        __op = 0
    else:
        __op = 1
    subplot = 3+showLum

    if plot:
        if test1plot:
            # -- Teff on the top right
            #ax3 = plt.subplot((len(list_filt)+len(list_color)+2)/__np+__op, 2, 2)
            # -- Teff on the bottom left:
            ax3 = plt.axes([0.06, win_ym-(1.5+win_e)*win_ys-win_yo, 0.43, win_ys*0.5])

        else:
            ax3 = plt.subplot(nplot, 1, next_plot, sharex=ax)

    # -- from table 15.7 of Astrophysical Quantities
    sp = ['F0','F1','F2','F3','F4','F5','F6','F7','F8',
          'F9','G0','G2','G4','G6',
          'G8','K0','K1','K2','K3','K4','K5']
    sp_t = [7460, 7240, 7030, 6810, 6590, 6370, 6163, 5956, 5750,
            5570, 5370, 5190, 5016, 4853,
            4700, 4550, 4430, 4310, 4203, 4096, 3990]
    Tmax = iTeff(X).max()
    Tmin = iTeff(X).min()
    w = np.where(types=='teff')
    if len(w[0])>0:
        Tmax = max(Tmax, max([data[k]+edata[k] for k in w[0]]))
        Tmin = min(Tmin, min([data[k]-edata[k] for k in w[0]]))
    if False and plot:
        # -- spectral types lines:
        for k in range(len(sp)):
            if sp_t[k]<=Tmax+150 and \
                sp_t[k]>=Tmin-150:
                plt.hlines(sp_t[k]*1e-3, -1,0.97, linewidth=3, alpha=0.5,
                        color=teffColorPalette(sp_t[k]))
                plt.hlines(sp_t[k]*1e-3, 1.03,2, linewidth=3, alpha=0.5,
                              color=teffColorPalette(sp_t[k]))
                plt.text(1.0, sp_t[k]*1e-3, sp[k], ha='center', va='center',
                            fontsize=10, color='0.3', weight='semibold',
                            variant='small-caps')
    if plot:
        # -- Teff model from SPIPS:
        plt.plot(X, iTeff(X)*1e-3, '-', label='model, ptp=%4.0fK'%(np.ptp(iTeff(X))),
                    linewidth=2, alpha=0.5, color=(0.1,0.5,0.25))
    # -- plot nodes:
    if useSplineTeff:
        j = 0
        for i in range(len(xpT)):
            if xpT[i]>=0 and xpT[i]<1:
                fits_model['TEFF SPLINE NODE PHI'+str(j)] = (xpT[i], 'phase')
                fits_model['TEFF SPLINE NODE VAL'+str(j)] = (ypT[i], 'K')
                j+=1
        if plot:
            plt.plot(xpT,ypT*1e-3, 'p', markersize=5,
                     label='Spline Nodes', color=(0.5,0.25,0.1))
            #if nteff%2==0:
            #   plt.plot(phi_intern, Teff/1000., '-', color=(0.5,0.2,0),
            #            linestyle='dashed', label='Spline')
            if not xt_pow is None:
                color=(0.8,0.4,0.0)
                Tmean = iTeff(np.linspace(0,1,100)).mean()*1e-3
                plt.plot(xt_pow, Tmean+0*xt_pow, '|', color=color,
                        linewidth=2, alpha=0.8, markersize=12, label='Spline comb')
                plt.plot(xt_pow-1, Tmean+0*xt_pow, '|', color=color,
                        linewidth=2, alpha=0.8, markersize=12)
                for i in xt_i:
                    if xt_pow[i]<1.1:
                        plt.text(xt_pow[i], Tmean, '%d'%i, color=color,
                                alpha=0.5, size=8)
                    if xt_pow[i]>0.9:
                        plt.text(xt_pow[i]-1, Tmean, '%d'%i, color=color,
                                alpha=0.5, size=8)

    chi2, nchi2 = 0.0, 0
    for i,ori in enumerate(set(orig[np.where(types=='teff')])):
        w = np.where((types=='teff')*(orig==ori))
        # -- plot observations
        wi = np.where([edata[k]>0 for k in w[0]])
        if len(wi[0])>0:
            chi2 += np.sum([((data[k]-res[k])/edata[k])**2 for k in w[0][wi]])
            nchi2 += len( w[0][wi])
            df=0
            #print '#'*6, '"'+ori+'"', chi2, w[0][wi]
            if plot:
                for df in [-1,0,1]:
                    plt.errorbar(phi[w][wi]+df-teff_phase_offset, [data[k]*1e-3 for k in w[0][wi]],
                                 yerr=[edata[k]*1e-3 for k in w[0][wi]],
                                fmt='.', markersize=5, color=colors[i%len(colors)],
                                label='%s'%(ori) if df==0 else '')
        wi = np.where([edata[k]<=0 for k in w[0]])
        if len(wi[0])>0:
            df=0
            if plot:
                plt.plot(phi[w][wi]+df-teff_phase_offset, [data[k]*1e-3 for k in w[0][wi]],
                        'x', markersize=5, label=ori+' ignored' if df==0 else '', color=colors[i%len(colors)])
    if nchi2>0:
        allChi2.append(('TEFF', chi2/nchi2))
    if plot:
        plt.ylabel('Teff (1e3K)')
        plt.xlim(-0.1,1.1)
        Tmin, Tmax = Tmin-0.10*(Tmax-Tmin), Tmax+0.10*(Tmax-Tmin)
        plt.ylim(Tmin*1e-3, Tmax*1e-3)
        plt.legend(loc='upper center', prop={'size':9}, frameon=False, numpoints=1, ncol=2)
        ax3.set_xticks([0,0.2,0.4,0.6,0.8,1.0])
        ax3.set_xticklabels([])
        ax3.tick_params(axis='both', which='major', labelsize=10)
        ax3.set_yticks(ax3.get_yticks()[1:])
        uni = labelPanel(uni)

    #### Luminosity ######
    if showLum and plot:
        ax = plt.subplot((len(list_filt)+len(list_color)+showLum)/__np+__op, 4, subplot-1)
        # -- Lum model from SPIPS:
        ax.plot(X, np.log10(iLum(X)), '-', label='model',
                    linewidth=2, alpha=0.5, color=colorModel)
        ax.set_ylabel('log10(L/Lsol) ')
        ax.set_xlim(-0.1,1.1)
        ax.set_ylim(ax.get_ylim()[1], ax.get_ylim()[0])

    # -- surface brightness plots
    showSurBri = False
    if showSurBri and plot:
        plt.figure(23)
        plt.clf()
        axsbr = plt.subplot(111)
        plt.figure(10)

    ##################################################
    ###### SECOND plot: photometry ###################
    ##################################################
    #colorMap_mags = 'spectral'
    colorMap_mags = 'jet'
    # -- plot magnitudes:
    print 'average absolute magnitudes: True, reddening, IR excess:'
    excess = {'wl':[], 'model avg':[], 'excess':[], 'reddened':[],
                'data med':[], 'data Q1':[], 'data Q3':[], 'err':[], 'sign':[]}

    for i, filt in enumerate(list_filt):
        if plot:
            if test1plot:
                ax = plt.subplot((len(list_filt)+len(list_color)+showLum)/__np+__op, 4, subplot)
                if subplot%4==0:
                    subplot += 3
                else:
                    subplot += 1
                ax.set_xticks([0,0.2,0.4,0.6,0.8,1.0])
            else:
                ax = plt.subplot((len(list_filt)+len(list_color)+1+showLum)/__np+__op, 2, subplot)
                subplot+=1

        w = np.where(np.array(filtname)==filt)[0]
        w = (np.array(w)[phi[w].argsort()], )
        wi = np.where([edata[k]>0 for k in w[0]])
        if len(wi[0])>=1:
            chi2 = np.mean([((data[k]-res[k])/edata[k])**2 for k in w[0][wi]])
            allChi2.append((filt, chi2))
        else:
            chi2=0

        # -- model:
        modl = []
        modlT = [] # only temperature effects
        diamAVG = iDiam(np.linspace(0,1,100)[:-1]).mean()
        modlD = [] # only diam effects
        teffAVG = iTeff(np.linspace(0,1,100)[:-1]).mean()

        xmo = np.linspace(-0.5,1.5,100)[:-1]
        wl0 = photfilt2.effWavelength_um(filt)
        excess['wl'].append(wl0)
        for x_ in xmo: # for each phase of the model
            modl.append(photometrySED([phot_corr*iDiam(x_), COMP_DIAM],
                                     [iTeff(x_), COMP_TEFF], filt,
                                     logg=[ilogg_m(x_), COMP_LOGG],
                                     metal=a['METAL']))
            modl[-1] = float(modl[-1]) # something nasty is going on...
            if splitDTeffects:
                modlT.append(photometrySED([phot_corr*diamAVG, COMP_DIAM],
                                         [iTeff(x_), COMP_TEFF], filt,
                                         logg=[ilogg_m(x_), COMP_LOGG],
                                         metal=a['METAL']))
                modlT[-1] = float(modlT[-1]) # something nasty is going on...
                modlD.append(photometrySED([phot_corr*iDiam(x_), COMP_DIAM],
                                         [teffAVG, COMP_TEFF], filt,
                                         logg=[ilogg_m(x_), COMP_LOGG],
                                         metal=a['METAL']))
                modlD[-1] = float(modlD[-1]) # something nasty is going on...

        modl = np.array(modl)
        if splitDTeffects:
            modlT = np.array(modlT)
            modlD = np.array(modlD)

        excess['model avg'].append(modl.mean())
        if verbose:
            print '%-20s = %5.3f '%(filt, np.mean(modl)-5*np.log10(a['d_kpc']/0.01)),

        if __monochromaticAlambda:
            _red = Alambda_Exctinction(wl0, EB_V=a['E(B-V)'], Rv=Rv, Teff=iTeff(xmo))
        else:
            _red = Alambda_Exctinction(filt, EB_V=a['E(B-V)'], Rv=Rv, Teff=iTeff(xmo))
        modl += _red
        if splitDTeffects:
            modlT += _red
            modlD += _red
        excess['reddened'].append(modl.mean())
        if verbose:
            print '+ %5.3f'%(np.mean(_red)),

        irex = 0
        if not f_excess is None:
            irex = f_excess(wl0)
        else:
            if np.abs(wl0-2.2)<0.3/2:
                irex = k_excess
            if np.abs(wl0-1.6)<0.3/2:
                irex = h_excess
            if np.abs(wl0-1.26)<0.3/2:
                irex = h_excess
        modl -= irex
        if splitDTeffects:
            modlT -= irex
            modlD -= irex
        if verbose:
            print '- %5.3f'%(irex),
            print '= %6.3f'%(modl.mean()-5*np.log10(a['d_kpc']/0.01))

        excess['excess'].append(modl.mean())

        if any(['dMAG ' in k for k in a.keys()]):
            for k in a.keys():
                if 'dMAG ' in k and filt in k:
                    modl += a[k]

        # -- pseudo variance
        _tmp = np.array([data[k] - res[k] for k in w[0]])
        excess['err'].append(np.mean([np.abs(edata[k]) for k in w[0]]))
        excess['data med'].append(np.median(_tmp))
        excess['data Q1'].append(np.percentile(_tmp, 25))
        excess['data Q3'].append(np.percentile(_tmp, 75))
        excess['sign'].append(np.sign(min([edata[k] for k in w[0]])))

        fits_data['MAG '+filt] = np.interp(fits_data['PHASE'], xmo, np.array(modl))
        fits_model['AVG_MAG '+filt] = round(modl.mean(), 3)
        fits_model['ABS_MAG_DERED '+filt] = round(np.mean(modl-_red)-5*np.log10(a['d_kpc']/0.01), 3)

        # -- find outliers:
        w__ = np.where(np.array([np.abs(data[k]-res[k]) for k in w[0]]) >
                        3*np.array([edata[k] for k in w[0]]))

        if len(w__[0])>0 and showOutliers:
            if verbose:
                print 'outliers in ', filt
                print [x[w[0][i]][0] for i in w__[0]]
            MJDoutliers.extend([x[w[0][i]][0] for i in w__[0]])

        tmp = list(set(orig[w])) # each biblio source
        avg = np.mean(modl)
        ptp = np.ptp(modl)
        if plot:
            for j,s in enumerate(np.sort(tmp)):
                # -- for each set of data (each litterature reference)
                color = plt.cm.get_cmap(colorMap)(j/float(max(len(tmp)-1,1)))
                color = [0.8*np.sqrt(c) for c in color[:-1]]
                color.append(1)
                _w = np.where(orig[w]==s)
                wi = np.where([not np.isnan(phi[k]) for k in w[0][_w]])
                if len(wi[0]):
                    for df in [-1,0,1]:
                        # -- typical error bar
                        if df == 0:
                            meanerr = np.mean([np.abs(edata[k]) for k in w[0][_w]])
                            plt.errorbar(0.95+0.1*j/float(len(tmp)),
                                         avg+0.4*ptp-0.2*j*ptp,
                                        yerr=meanerr, linestyle='none',
                                        marker='.', alpha=0.5,
                                        markersize=3, color=color)
                        # -- data points
                        wi = np.where([edata[k]>0 for k in w[0][_w]])
                        if len(wi[0]):
                            plt.plot(phi[w][_w][wi]+df, [data[k] for k in w[0][_w][wi]],
                                linestyle='none', markersize=3, color=color,
                                marker='.', alpha=0.7, label=s if df==0 else '')
                        wi = np.where([edata[k]<=0 for k in w[0][_w]])
                        if len(wi[0]):
                            plt.plot(phi[w][_w][wi]+df, [data[k] for k in w[0][_w][wi]],
                                linestyle='none', markersize=3, color=color,
                                marker='x', alpha=0.5, label=s+' ignored' if df==0 else '')
                # -- case no phase/mjd was given
                wi = np.where([np.isnan(phi[k]) for k in w[0][_w]])
                if len(wi[0]):
                    for k in np.arange(len(data))[w][_w][wi]:
                        plt.fill_between([-0.5,1.5], data[k]-edata[k]*np.array([1,1]),
                                    data[k]+edata[k]*np.array([1,1]),
                                    color='orange' if edata[k]<=0 else color ,
                                    alpha=0.2, #hatch='x' if edata[k]<0 else None,
                                    label=s+' ignored' if edata[k]<=0 else '')
            ### models
            plt.plot(xmo, modl, color=colorModel, linewidth=2, alpha=0.5,
                    label='model')
            if splitDTeffects:
                plt.plot(xmo, np.array(modlT), color='y', linewidth=1, alpha=0.8,
                        linestyle='dotted', label=r'$\Delta$R=0')
                plt.plot(xmo, np.array(modlD), color='g', linewidth=1, alpha=0.8,
                        linestyle='dotted', label=r'$\Delta$Teff=0')

            if showSurBri:
                axsbr.plot(iTeff(xmo), modl+5*np.log10(iDiam(xmo))+f_excess(wl0), '-', label='')
                wi = np.where([not np.isnan(phi[k]) for k in w[0][_w]])
                axsbr.plot(iTeff(phi[w][_w][wi]), [data[k]+5*np.log10(iDiam(phi[k])) for k in w[0][_w][wi]], 'o')

            if not f_excess is None and f_excess(wl0)>0.01:
                plt.plot(xmo, np.array(modl)+f_excess(wl0), color=colorModel,
                         linewidth=1.5, linestyle='dashed',
                         label='no CSE', alpha=0.5)
            else:
                ### K,H excess
                if l_excess!=0 and np.abs(wl0-3.5)<=0.5/2:
                    print '--- I SHOULD NOT PASS HERE!?! ---'
                    plt.plot(xmo, np.array(modl)+l_excess, color=colorModel,
                             linewidth=1.5, linestyle='dashed',
                             label='no CSE', alpha=0.5)
                if k_excess!=0 and np.abs(wl0-2.2)<=0.3/2:
                    print '--- I SHOULD NOT PASS HERE!?! ---'
                    plt.plot(xmo, np.array(modl)+k_excess, color=colorModel,
                             linewidth=1.5, linestyle='dashed',
                             label='no CSE', alpha=0.5)
                if h_excess!=0 and np.abs(wl0-1.6)<=0.3/2:
                    print '--- I SHOULD NOT PASS HERE!?! ---'
                    plt.plot(xmo, np.array(modl)+h_excess, color=colorModel,
                             linewidth=1.5, linestyle='dashed',
                             label='no CSE', alpha=0.5)
                if j_excess!=0 and np.abs(wl0-1.26)<=0.3/2:
                    print '--- I SHOULD NOT PASS HERE!?! ---'
                    plt.plot(xmo, np.array(modl)+j_excess, color=colorModel,
                              linewidth=1.5, linestyle='dashed',
                              label='no CSE', alpha=0.5)
            plt.legend(loc='upper left', prop={'size':9}, numpoints=1,
                        frameon=False)

            y_min = min(min(modl), np.min([data[k]-0*edata[k] for k in w[0]]))
            y_max = max(max(modl), np.max([data[k]+0*edata[k] for k in w[0]]))
            plt.text(xmo[np.argmax(modl)]%1+0.1, y_min, filt,
                     va='bottom', ha='right', size=12, color='k',
                     alpha=0.33, fontweight='bold')
            plt.ylim(y_min-0.15*(y_max-y_min), y_max+0.25*(y_max-y_min))
            plt.xlim(-0.1,1.1)
            uni = labelPanel(uni)
            plt.text(1.05, plt.ylim()[1]-0.05*(plt.ylim()[1]-plt.ylim()[0]),
                     r'$\chi^2=$%4.2f'%chi2, va='top', ha='right',
                     size=10)
            _y = ax.get_yticks()
            if len(_y>7):
                ax.set_yticks(_y[1:-1][::2])
            elif len(_y>5):
                ax.set_yticks(_y[1:-1])
            ax.tick_params(axis='both', which='major', labelsize=10)

    for k in excess.keys():
        excess[k] = np.array(excess[k])

    # == plot colors:
    print 'average dereddened colors:'
    for i, filt in enumerate(list_color):
        if plot:
            if test1plot:
                #ax = plt.subplot((len(list_filt)+len(list_color)+2)/__np+__op, 4, subplot)
                ax = plt.subplot((len(list_filt)+len(list_color)+showLum)/__np+__op, 4, subplot)

                if subplot%4==0:
                    subplot += 3
                else:
                    subplot += 1
                ax.set_xticks([0,0.2,0.4,0.6,0.8,1.0])
            else:
                ax = plt.subplot((len(list_filt)+len(list_color)+2)/__np+__op, 2, subplot)
                subplot+=1

        w = np.where(np.array(filtname)==filt)[0]
        w = (np.array(w)[phi[w].argsort()], )
        wi = np.where([edata[k]>0 for k in w[0]])
        if len(wi[0])>=1:
            chi2 = np.mean([((data[k]-res[k])/edata[k])**2 for k in w[0][wi]])
            allChi2.append((filt, chi2))
        else:
            chi2 = 0.0
        modl = []
        wl = 0.5*(photfilt2.effWavelength_um(filt.split('-')[0])+
                  photfilt2.effWavelength_um(filt.split('-')[1]))
        xmo = np.linspace(-0.5,1.5,100)[:-1]
        for x_ in xmo: # for each phase of the model
            modl.append(photometrySED([iDiam(x_), COMP_DIAM],
                                         [iTeff(x_), COMP_TEFF],
                                         filt.split('-')[0],
                                         logg=[ilogg_m(x_), COMP_LOGG],
                                    metal=a['METAL'] if a.has_key('METAL') else 0.0)-
                        photometrySED([iDiam(x_), COMP_DIAM],
                                         [iTeff(x_), COMP_TEFF],
                                         filt.split('-')[1],
                                         logg=[ilogg_m(x_), COMP_LOGG],
                                    metal=a['METAL'] if a.has_key('METAL') else 0.0))
            wl0 = photfilt2.effWavelength_um(filt.split('-')[0])
            wl1 = photfilt2.effWavelength_um(filt.split('-')[1])
            if __monochromaticAlambda:
                _red = Alambda_Exctinction(wl0, EB_V=a['E(B-V)'], Rv=Rv) -\
                            Alambda_Exctinction(wl1, EB_V=a['E(B-V)'], Rv=Rv)
            else:
                _red = Alambda_Exctinction(filt.split('-')[0], EB_V=a['E(B-V)'], Rv=Rv, Teff=iTeff(x_)) -\
                            Alambda_Exctinction(filt.split('-')[1], EB_V=a['E(B-V)'], Rv=Rv, Teff=iTeff(x_))
            modl[-1] += _red

            if not f_excess is None:
                modl[-1] -= f_excess(wl0)
                modl[-1] += f_excess(wl1)
            else:
                if np.abs(wl0-3.5)<0.5/2:
                    modl[-1] -= l_excess
                if np.abs(wl1-3.5)<0.5/2:
                    modl[-1] += l_excess
                if np.abs(wl0-2.2)<0.3/2:
                    modl[-1] -= k_excess
                if np.abs(wl1-2.2)<0.3/2:
                    modl[-1] += k_excess
                if np.abs(wl0-1.6)<0.3/2:
                    modl[-1] -= h_excess
                if np.abs(wl1-1.6)<0.3/2:
                    modl[-1] += h_excess
                if np.abs(wl0-1.26)<0.3/2:
                    modl[-1] -= j_excess
                if np.abs(wl1-1.26)<0.3/2:
                    modl[-1] += j_excess


            modl[-1] = float(modl[-1]) # something nasty is going on...
        modl = np.array(modl)
        if any(['dMAG ' in k for k in a.keys()]):
            for k in a.keys():
                if 'dMAG ' in k and filt.split('-')[0] in k:
                    modl += a[k]
                elif 'dMAG ' in k and filt.split('-')[1] in k:
                    modl -= a[k]

        fits_data['COLOR '+filt] = np.interp(fits_data['PHASE'], xmo, np.array(modl))
        fits_model['AVG_COLOR '+filt] = round(modl.mean(), 3)
        fits_model['AVG_COLOR_DERED '+filt] = round(np.mean(modl-_red), 3)

        # -- find outliers:
        w__ = np.where(np.array([np.abs(data[k]-res[k]) for k in w[0]])>3.*np.array([edata[k] for k in w[0]]))
        if len(w__[0])>0 and showOutliers:
            print 'outliers in', filt
            print [x[w[0][i]][0] for i in w__[0]]
            MJDoutliers.extend([x[w[0][i]][0] for i in w__[0]])

        tmp = list(set(orig[w]))
        avg = np.mean(modl)
        ptp = np.ptp(modl)
        if plot:
            for j,s in enumerate(np.sort(tmp)):
                color = plt.cm.get_cmap(colorMap)(j/float(max(len(tmp)-1,1)))
                color = [0.8*np.sqrt(c) for c in color[:-1]]
                color.append(1)

                _w = np.where(orig[w]==s)
                for df in [-1,0,1]:
                    # -- typical error bar
                    if df == 0:
                        meanerr = np.mean([np.abs(edata[k]) for k in w[0][_w]])
                        plt.errorbar(0.95+0.1*j/float(len(tmp)), avg+0.3*ptp-0.15*j*ptp,
                            yerr=meanerr, linestyle='none',
                            marker='.', alpha=0.5,
                            markersize=3, color=color)
                    # -- data points
                    wi = np.where([edata[k]>0 for k in w[0][_w]])
                    if len(wi[0]):
                        plt.plot(phi[w][_w][wi]+df, [data[k] for k in w[0][_w][wi]],
                            linestyle='none',
                            marker='.', alpha=0.7, label=s if df==0 else '',
                            markersize=3, color=color)
                    wi = np.where([edata[k]<=0 for k in w[0][_w]])
                    if len(wi[0]):
                        plt.plot(phi[w][_w][wi]+df, [data[k] for k in w[0][_w][wi]],
                            linestyle='none',
                            marker='x', alpha=0.5, label=s+' ignored' if df==0 else '',
                            markersize=3, color=color)

            ### model
            print '%-20s = %5.3f'%(filt, np.mean(modl-_red))
            plt.plot(xmo, np.array(modl), color=colorModel, linewidth=2, alpha=0.5)

            ### K,H excess
            e = 0
            if not f_excess is None:
                e += f_excess(photfilt2.effWavelength_um(filt.split('-')[0]))
                e -= f_excess(photfilt2.effWavelength_um(filt.split('-')[1]))
            else:
                if np.abs(photfilt2.effWavelength_um(filt.split('-')[0])-3.5)<0.5/2:
                    e += l_excess
                if np.abs(photfilt2.effWavelength_um(filt.split('-')[1])-3.5)<0.5/2:
                    e -= l_excess
                if np.abs(photfilt2.effWavelength_um(filt.split('-')[0])-2.2)<0.3/2:
                    e += k_excess
                if np.abs(photfilt2.effWavelength_um(filt.split('-')[1])-2.2)<0.3/2:
                    e -= k_excess
                if np.abs(photfilt2.effWavelength_um(filt.split('-')[0])-1.6)<0.3/2:
                    e += h_excess
                if np.abs(photfilt2.effWavelength_um(filt.split('-')[1])-1.6)<0.3/2:
                    e -= h_excess
                if np.abs(photfilt2.effWavelength_um(filt.split('-')[0])-1.26)<0.3/2:
                    e += j_excess
                if np.abs(photfilt2.effWavelength_um(filt.split('-')[1])-1.26)<0.3/2:
                    e -= j_excess
            if np.abs(e)>0.005:
                plt.plot(xmo, np.array(modl)+e, color=colorModel,
                         linewidth=1.5, linestyle='dashed',
                         label='no CSE', alpha=0.5)
            plt.legend(loc='upper left', prop={'size':9}, numpoints=1,
                       frameon=False)

            y_min = min(min(modl), np.min([data[k]-0*edata[k] for k in w[0]]))
            y_max = max(max(modl), np.max([data[k]+0*edata[k] for k in w[0]]))

            plt.text(xmo[np.argmax(modl)]%1+0.1, y_min, filt.replace('-', ' -\n'),
                     va='bottom', ha='right', size=12, color='k',
                     alpha=0.33, fontweight='bold')
            plt.ylim(y_min-0.15*(y_max-y_min), y_max+0.25*(y_max-y_min))
            plt.xlim(-0.1,1.1)
            uni = labelPanel(uni)
            plt.text(1.05, plt.ylim()[1]-0.05*(plt.ylim()[1]-plt.ylim()[0]),
                     r'$\chi^2=$%4.2f'%chi2, va='top', ha='right',
                     size=10)
            _y = ax.get_yticks()
            if len(_y>7):
                ax.set_yticks(_y[1:-1][::2])
            elif len(_y>5):
                ax.set_yticks(_y[1:-1])
            ax.tick_params(axis='both', which='major', labelsize=10)

            # -- last plot gets the xlabel set
            plt.xlabel('pulsation phase')

    if len(MJDoutliers)>0:
        print 'ALL MJD outliers:', MJDoutliers

    #### THIRD plot: spectra ##############
    show_spectra_res =  True # show residuals
    if plot and any([obs[1].split(';')[0]=='normalized spectrum' for obs in x]):
        plt.close(12)
        if show_spectra_res :
            figures.append(plt.figure(12, figsize=(12,10)))
        else:
            figures.append(plt.figure(12, figsize=(6,8)))
        plt.clf()
        plt.subplots_adjust(right=0.98, top=0.95, left=0.1, bottom=0.07)
        if show_spectra_res:
            ax = plt.subplot(121)
        else:
            ax = plt.subplot(111)
            plt.title(title, weight='semibold')
        spread=8. # spacing between spectra: spread*pulsation_phase
        plt.ylabel('(spectra-1)/%3.1f + pulsating phase'%(spread))
        plt.xlabel('wavelength ($\mu$m)')
        if show_spectra_res :
            plt.subplot(122, sharex=ax, sharey=ax)
            plt.ylabel('residuals/%3.1f + pulsating phase'%(spread))
            plt.xlabel('wavelength ($\mu$m)')
        XMIN = 1e8
        XMAX = 0
        _scale = True
        for k, o in enumerate(x):
            if o[1]=='normalized spectrum':
                if o[2].min()<XMIN:
                    XMIN = o[2].min()
                if o[2].max()>XMAX:
                    XMAX = o[2].max()
                if show_spectra_res:
                    plt.subplot(121)
                # -- plot observed spectrum at phase phi=o[0] ---------------
                plt.plot(o[2], (o[3]-1)/spread+phi[k], '-', alpha=0.7,
                    color=teffColorPalette(iTeff(phi[k])), linewidth=2.5)

                # -- model spectra: -----------------------------------------
                plt.plot(o[2], (res[k]-1)/spread+phi[k],
                           color= 'k', alpha=0.9, linewidth=1,
                           label=filt)
                # -- scale --------------------------------------------------
                if _scale:
                    #print XMIN, phi[k], phi[k]-spread
                    plt.vlines(XMIN+0.02*(XMAX-XMIN),
                               phi[k], phi[k]-1./spread, linewidth=5)

                # -- residuals: ---------------------------------------------
                if show_spectra_res:
                    plt.subplot(122, sharex=ax, sharey=ax)
                    plt.hlines(phi[k], o[2].min(), o[2].max(), color='k',
                                  alpha=0.5, linewidth=1)
                    plt.plot(o[2], (o[3]-res[k])/spread+phi[k], alpha=1.0,
                                color=teffColorPalette(iTeff(phi[k])), linewidth=2)
                    if _scale:
                        plt.vlines(XMIN+0.02*(XMAX-XMIN),
                                   phi[k], phi[k]-1./spread, linewidth=5)
                if _scale:
                    _scale = False

        if show_spectra_res:
            plt.subplot(121)
        xt = np.linspace(XMIN, XMAX, 20)

        # -- vertical lines showing vrad
        y = np.linspace(-1,2,300)
        for _x in xt:
            if show_spectra_res:
                plt.subplot(121)
            plt.plot(_x*(1+iVpuls(y%1)/3e5), y, 'k', alpha=0.5,
                     linestyle='dotted')
            if show_spectra_res:
                plt.subplot(122, sharex=ax, sharey=ax)
                plt.plot(_x*(1+iVpuls(y%1)/3e5), y, 'k', alpha=0.5,
                         linestyle='dotted')
        plt.ylim(ymin=-0.8/spread, ymax=1+0.2/spread)
        plt.xlim(XMIN, XMAX)
    else:
        plt.close(12)

    chi2 = np.mean([z[1] for z in allChi2 if z[0]!='TOTAL'])
    allChi2.append(('AVG', chi2))

    n = np.max([len(z[0]) for z in allChi2])
    form = ' | %%-%ds : %%5.2f'%(n)
    print '='*6, 'CHI2', '='*n
    for i in range(len(allChi2)-2):
        print form%allChi2[i+1]
    form = form.replace(' |', '>>')
    print '-'*(n+12)
    print form%allChi2[0]
    print form%allChi2[-1]
    print 'number of data points:', len(x)

    if 'PERIOD1' in a.keys():
        mjd = np.array([o[0] if not o[0] is None else np.nan for o in x])

        phi, period = phaseFunc(mjd, a)
        period[np.isnan(period)] = np.nanmean(period)
        phi[mjd<1] = mjd[mjd<1]
        dphi, dperiod = phaseFunc(mjd+365.25, a)
        dperiod[np.isnan(dperiod)] = np.nanmean(dperiod)

        phi0, period0 = phaseFunc(mjd, {'MJD0':a['MJD0'],
                                        'PERIOD':a['PERIOD']})
        period0[np.isnan(period0)] = np.nanmean(period0)

        _y = (phi[mjd>1]-phi0[mjd>1]+0.5)%1.-0.5
        _x = mjd[mjd>1]

        print ' > observed period change: %3.2f s/yr'%a['PERIOD1']
        print ' > model prediction from Fadeyev 2014 (blue edge -> red edge)'

        for i in [1,2,3]:
            tmp = periodChange(mass_pr, i=i)
            if i==2:
                tmp = (tmp[1], tmp[0])
            print ' crossing %d'%i,
            print 'from %7.2f -> %7.2f s/yr'%tmp

    if not f_excess is None:
        tmp = tuple([f_excess(l) for l in [2., 5., 10., 24]])
        print '--------------------------------------------------------------'
        print 'excess at 2, 5, 10 and 24 um (mag): %5.3f, %5.3f, %5.3f, %5.3f'%tmp
        print '--------------------------------------------------------------'
        for l in [1.,1.5, 2.,3, 4.,6.,8.,12., 16., 20., 30., 50.]:
            fits_model['IR EXCESS %4.1fUM'%l] = (round(f_excess(l), 3), 'magnitude')
    if plot:
        figures.append(plt.figure(99))
        plt.clf()
        ax2 = plt.subplot(111)
        plt.grid()
        plt.xscale('log')
        plt.ylabel('apparent magnitude')

        _X = [0.3, 0.5, 1, 2, 3, 5, 10,20, 50, 100]
        _X = filter(lambda x: x<=1.2*max(excess['wl']) and
                              x>=0.8*min(excess['wl']), _X)
        ax2.set_xticks(_X)
        ax2.set_xticklabels([str(_x) for _x in _X])
        # - data

        for k in range(len(excess['data med'])):
            off = excess['excess'][k] - excess['reddened'][k]
            color = 'b' if excess['sign'][k]>=0 else 'orange'
            plt.plot(excess['wl'][k], -excess['data med'][k] - off,
                    'h' if excess['sign'][k]>=0 else 'd',
                     color=color, alpha=0.5, label='model - photometric data' if k==0 else '')
            Q1 = excess['data med'][k] - np.sqrt((excess['data Q1'][k] -
                                                  excess['data med'][k])**2 + excess['err'][k]**2)
            Q3 = excess['data med'][k] + np.sqrt((excess['data Q3'][k] -
                                                  excess['data med'][k])**2 + excess['err'][k]**2)
            plt.plot(excess['wl'][k], -Q1 - off,
                    marker='_', color=color, alpha=0.5)
            plt.plot(excess['wl'][k], -Q3 - off,
                    marker='_', color=color, alpha=0.5)
            plt.plot([excess['wl'][k],excess['wl'][k]],
                    [-Q1-off, -Q3-off], '-', color=color, alpha=0.5)
        plt.hlines(0, excess['wl'].min(),excess['wl'].max(), linestyle='dotted')
        wl = np.logspace(np.log10(min(excess['wl'])),
                         np.log10(max(excess['wl'])), 100)
        if not f_excess is None:
            plt.plot(wl, [f_excess(z) for z in wl], '--y',label='modeled IR excess')
        plt.legend(loc='upper left')
        plt.xlabel('effective wavelength (um)')
        plt.ylabel('$\Delta$ magnitude')
        if title[0] != ' ':
            plt.title(title, fontsize=12, fontweight='bold')

    # === save FITS and plots =============================================
    if exportFits:
        filename = fitsName(starName)
        if os.path.exists(filename+'.fits'):
            os.remove(filename+'.fits')
        print '--EXPORT--:', filename+'.fits'

        # -- FITS
        hdu = pyfits.PrimaryHDU()
        hdu.header['AUTHOR'] = getpass.getuser()
        hdu.header['RUNDATE'] = time.asctime()
        hdu.header['COMMENT'] = 'generated with SPIPS: spectrophoto-interferometry of pulsating stars'
        hdu.header['COMMENT'] = 'https://github.com/amerand/SPIPS'
        if not starName is None:
            hdu.header['STARNAME'] = starName

        for k in np.sort(a.keys()):
            hdu.header['HIERARCH PARAM '+k] = a[k]
        for k in np.sort(fits_model.keys()):
            hdu.header['HIERARCH MODEL '+k] = fits_model[k]
        for k in allChi2:
            hdu.header['HIERARCH CHI2 '+k[0]] = round(k[1], 2)
        # -- model
        cols=[]
        # TODO
        units = {'Vpuls': 'km/s', 'Vrad':'km/s',
                'diam':'mas', 'logg':'cm/s2', 'Lum':'Lsol',
                'Teff':'K', 'R':'Rsol'}
        for k in fits_data.keys():
            unit = None
            for u in units.keys():
                if u in k:
                    unit=units[u]
            cols.append(pyfits.Column(name=k, format='E', unit=unit,
                                      array=fits_data[k]))
        hdum = pyfits.BinTableHDU.from_columns(cols)

        hdum.header['EXTNAME'] = 'MODEL'

        # -- data
        cols=[]
        cols.append(pyfits.Column(name='MJD', format='E',
                                      array=np.array([o[0] for o in x])))
        # -- this encodes lots of different possible things...:
        tmp = []
        for o in x:
            _t = list(o[1:-2])
            # -- to avoid problems while reading
            _t[0] = _t[0].replace('(', '')
            _t[0] = _t[0].replace(')', '')
            _t[0] = _t[0].replace('[', '')
            _t[0] = _t[0].replace(']', '')
            tmp.append('|'.join([str(q) for q in _t]))
        tmp = np.array(tmp)
        n = max([len(s) for s in tmp])
        # -- build data table
        cols.append(pyfits.Column(name='OBS', format='A'+str(n), array=tmp))
        cols.append(pyfits.Column(name='MEAS', format='E',
                                      array=np.array([o[-2] for o in x])))
        cols.append(pyfits.Column(name='ERR', format='E',
                                      array=[o[-1] for o in x]))
        cols.append(pyfits.Column(name='MODEL', format='E',
                                      array=np.array(res)))
        cols.append(pyfits.Column(name='PHASE', format='E',
                                      array=phi))
        cols.append(pyfits.Column(name='PERIOD', format='E11.8',
                                      array=period))

        #hducols = pyfits.ColDefs(cols)
        #hdud = pyfits.new_table(hducols)
        hdud = pyfits.BinTableHDU.from_columns(cols)
        hdud.header['EXTNAME'] = 'DATA'
        hdud.header['COMMENT'] = 'OBS colums contains a description of the data'
        hdud.header['COMMENT'] = 'the string before ; defines the type, after ; is the source'
        hdud.header['COMMENT'] = 'after | are anciliary data:'
        hdud.header['COMMENT'] = 'for diam, UDdiam: [wavelength_um, interf_baseline_m]'
        hdud.header['COMMENT'] = 'for mag: photometric band'
        hdud.header['COMMENT'] = 'for color: photometric band1 - photometric band2'

        thdulist = pyfits.HDUList([hdu, hdud, hdum])

        if not os.path.isdir(_dir_export):
            os.path.mkdir(_dir_export)
        export = {'FITS':os.path.join(_dir_export, filename+'.fits')}

        thdulist.writeto(export['FITS'])
        if len(figures)>0:
            export['FIG'] = []

            for k, f in enumerate(figures):
                export['FIG'].append(os.path.join(_dir_export, filename+'_Fig'+str(k)+'.pdf'))
                f.savefig(export['FIG'][-1])
        return export
    return res

def importFits(fitsname, runSPIPS=False):
    """
    read fits and extract parameters dictionnary and data vector
    """
    f = pyfits.open(fitsname)
    if 'STARNAME' in f[0].header.keys():
        starName = f[0].header['STARNAME']
    else:
        starName = ''

    # -- build dict of parameters:
    a = {}
    for k in filter(lambda x: x.startswith('PARAM '), f[0].header):
        a[k.split('PARAM ')[1]] = f[0].header[k]

    # -- build data list:
    obs = []
    for k in range(len(f['DATA'].data)):
        tmp = [f['DATA'].data['MJD'][k]] # MJD
        if not '|' in f['DATA'].data['OBS'][k]:
            tmp.append(f['DATA'].data['OBS'][k])
        else:
            for x in f['DATA'].data['OBS'][k].split('|'):
                if x.startswith('(') or x.startswith('['):
                    t = []
                    for _t in x[1:-1].split(','):
                        try:
                            t.append(float(_t))
                        except:
                            t.append(_t)
                    tmp.append(t)
                else:
                    try:
                        tmp.append(float(x))
                    except:
                        tmp.append(x)
        tmp.append(f['DATA'].data['MEAS'][k]) # measurement
        tmp.append(f['DATA'].data['ERR'][k]) # error
        obs.append(tuple(tmp))
    f.close()
    if runSPIPS:
        model(obs, a, plot=True, starName=starName, verbose=True)
    else:
        return a, obs

def datasetPhaseCoverageQuality(obs):
    Nvrad = np.sum([(o[1].startswith('vrad') or o[1].startswith('vpuls'))
                    and o[-1]>0 for o in obs ])
    Nteff = np.sum([o[1].startswith('teff') and o[-1]>0 for o in obs])
    Ninterf = np.sum(['diam' in o[1].split(';')[0]  for o in obs])
    mags = filter(lambda o: o[1].startswith('mag') and o[-1]>0, obs)
    Nvis, Nir = 0, 0
    if len(mags)>0:
        wl = np.array([photfilt2.effWavelength_um(o[2]) for o in mags])
        Nvis += np.sum(wl<1.0)
        Nir  += np.sum(wl>=1.0)
    # -- colors constrain more Teff than radius, just like vis photometry
    Nvis += np.sum(['color' in o[1].split(';') and o[-1]>0 for o in obs])
    return qualityPhaseFuntion(Nvrad, Nvis, Nir, Ninterf, Nteff)

def qualityPhaseCoverage(Nvrad=0, Nvis=0, Nir=0, Ninterf=0, Nteff=0):
    """
    Nvrad: number of radial velocity measurements
    Nvis: number of visible photometric measurements
    Nir: number of IR photometric measurements
    Ninterf: number of interferometric measurements
    Nteff: number of spectra and Teff measurements

    stats on max gaps:
    gap        50%   90%
    0.20 for N> 17 N> 26
    0.10 for N> 42 N> 64
    0.05 for N>102 N>147
    """
    # -- data where good phase coverage is needed:
    Q1 = [0, 24, 64, 144]
    qual1 = lambda n: np.sum([n>q for q in Q1])/(len(Q1)-1.)
    # -- data where any is needed, 10x less than q1 basically
    Q2 = [0, 2, 6, 14]
    qual2 = lambda n: np.sum([n>q for q in Q2])/(len(Q2)-1.)
    # -- total quality
    res = qual1(Nvrad) + qual2(Nir) + qual2(Ninterf)
    if Nteff>0:
        res += qual2(Nvis) + qual2(Nteff)
    else:
        res += qual1(Nvis)
    return res


def pseudoStat(x):
    # -- pseudo variance
    _s = np.argsort(x)
    res = {'median': np.median(x),
            'mean': np.mean(x),
            'pseudo std': 0.5*(x[_s[int(len(_s)*1-.68/2)]] -
               x[_s[int(len(_s)*.68/2)]])}
    return res

_maxCores = None

__bunch = 1
def modelM(x,p, bunch=None, maxCores=None):
    """
    parallelized version of model. 'bunch' is the size of the data to
    be treated by each subprocess.
    """
    global __bunch, phaseOffset, _maxCores
    cb_1([], init=len(x))
    if not maxCores is None:
        _maxCores=maxCores
    nCores = multiprocessing.cpu_count()
    if not _maxCores is None:
        po = Pool(min(nCores, _maxCores))
    else:
        po = Pool()
    if bunch is None:
        bunch=max(int(np.ceil(len(x)/float(po._processes))),1)
        #print 'BUNCH:', bunch, len(x)
    __bunch=bunch
    if po._processes==1:
        # -- single thread:
        for k,o in enumerate(x):
            cb_1(f_1(k, x[__bunch*k:__bunch*(k+1)], p))
        po.close()
        po.join()
    else:
        # -- multithread:
        if bunch==1:
            for k,o in enumerate(x):
                po.apply_async(f_1, (k, o, p), callback=cb_1)
        else:
            for k in range(int(np.ceil(len(x)/float(__bunch)))):
                #print 'ASYNC:', k
                po.apply_async(f_1, (k, x[__bunch*k:__bunch*(k+1)], p),
                                callback=cb_1)
        po.close()
        po.join()
    return cb_1([], output=True)

def f_1(k,x,a):
    """
    sub function for modelM
    """
    global __bunch
    return (k, model(x, a, plot=False))
    try:
        if __bunch==1:
            r = (k, model(x, a, plot=False)[0])
        else:
            r = (k, model(x, a, plot=False))
        return r
    except:
        print 'F_1 ERROR!', __bunch
        print k, len(x),
        print len(x), len(model(x, a, plot=False))
        pass
__bunch = 1


def cb_1(x, output=False, init=None):
    """
    sub function for modelM

    if INIT is a integer, initialize the result with an array on length INIT

    if OUTPUT is set to True, will return the result

    else, expects x=(k, value), where value can be anything. Will set
    res1[k] to value.
    """
    global res1, __bunch
    try:
        if output:
            #print '-- RETURN'
            return res1
        if not init is None:
            #print '-- INIT'
            res1 = [0.0 for k in range(init)]
            return
        if __bunch==1:
            res1[x[0]] = x[1]
            #print '-- FILL:', x[0], __bunch, len(x[1])
        else:
            res1[x[0]*__bunch:x[0]*__bunch+len(x[1])] = x[1]
            #print '-- FILL BUNCH:', x[0]
    except:
        #print 'CB_1:', len(res1), x, output, init
        pass
    return

def computePhaseOffset(obs, a, deltaMJD=1000., mjds=None):
    """
    for each deltaMJD range (in days), compute best period offset.

    obs -> array of tuples, describing the dataset
    a -> parameters

    -> assumes first elements of x are MJD, not phases

    """
    mjd_min = np.min([x[0] for x in obs])
    mjd_max = np.max([x[0] for x in obs])

    if mjds is None:
        mjds = np.linspace(mjd_min, mjd_max, int((mjd_max-mjd_min)/deltaMJD))

    dP = a['PERIOD']*np.linspace(-.5, .5, 50)

    dphi = []
    for i in range(len(mjds)-1):
        obs_ = filter(lambda o: o[0]>=mjds[i] and o[0]<mjds[i+1], obs)
        if len(obs_)>0:
            mjd_mean = np.mean([x[0] for x in obs_])
            print '%9.3f -> %9.3f (%8.3f) N=%3d'%(mjds[i], mjds[i+1], mjds[i+1]-mjds[i], len(obs_)),
            chi2 = []
            for d in dP:
                obs_2 = [list(o) for o in obs_]
                for i in range(len(obs_2)):
                    obs_2[i][0]+= d
                mod = modelM(obs_2, a)
                chi2.append(np.mean([(obs_[k][-2]-mod[k])**2/obs_[k][-1]**2 for k in range(len(mod))]))
            #print dP, chi2
            dphi.append((mjd_mean, findMin(np.array(dP), np.array(chi2))))
            print dphi[-1][-1]
        else:
            pass
    return dphi

def findMin(x, y):
    i0 = np.argmin(y)
    if i0==0:
        i = [i0, i0+1, i0+2]
    elif i0==len(y)-1:
        i = [i0-2, i0-1, i0]
    else:
        i = [i0-1, i0, i0+1]
    c = np.polyfit(x[i], y[i], 2)
    return -c[1]/(2*c[0])

def phaseFunc(mjd, p, vgamma=0.0):
    """
    assumes that all MJD are MJDs, not phase (0-1)
    """
    global __period, __mjd_mjd0
    __mjd_mjd0 = mjd
    __period = 0.0

    if p.has_key('MJD0'):
        __mjd_mjd0 = mjd - p['MJD0']

    # -- compute polynomial period
    if p.has_key('PERIOD'):
        __period = p['PERIOD']*np.ones(len(mjd))
    if p.has_key('PERIOD1'): # in s/year
        __period += (__mjd_mjd0)*p['PERIOD1']/(24*3600*365.25)
    if p.has_key('PERIOD2'):
        __period += ((__mjd_mjd0)/1e4)**2*p['PERIOD2']
    if p.has_key('PERIOD3'):
        __period += ((__mjd_mjd0)/1e4)**3*p['PERIOD3']
    if p.has_key('PERIOD4'):
        __period += ((__mjd_mjd0)/1e4)**4*p['PERIOD4']
    if p.has_key('PERIOD5'):
        __period += ((__mjd_mjd0)/1e4)**5*p['PERIOD5']

    # -- check for keys: "PERIOD MJDx", "PERIOD VALx"
    # -- for piecewise lineate interpolations
    p_ = []
    mjd_ = []
    for k in p.keys():
        if k.startswith('PERIOD MJD'):
            mjd_.append(p[k])
            p_.append(p['PERIOD VAL'+k.split('MJD')[1]])
    if len(p_)>0:
        p_ = np.array(p_)[np.argsort(mjd_)]
        mjd_ = np.array(mjd_)[np.argsort(mjd_)]
        __period = np.interp(mjd, mjd_, p_)

    # -- sin variation of period
    if p.has_key('PERIOD SIN AMP') and \
        p.has_key('PERIOD SIN PHI') and \
         p.has_key('PERIOD SIN PER'):
         # -- amplitude in seconds:
         s = 1/(24*3600.)
         __period += s*p['PERIOD SIN AMP']*np.sin(2*np.pi*__mjd_mjd0/p['PERIOD SIN PER'] +
                                                p['PERIOD SIN PHI'])

    # -- phase shift due to Vgamma and speed of light.
    # -- phase shift / MJD0
    # -- Vgamma in km/s
    # -- Vgamma >0 means coming toward Earth, early after MJD0, so - sign
    __corr = 0.0
    #__corr = -vgamma/(300.*1e3)*__mjd_mjd0
    #print 'max_correction:', np.abs(__corr).max(), 'days'

    _x, _y = [], []
    for k in p.keys():
        if 'DELTA PHI MJD' in k:
            _x.append(float(k.split('MJD')[1]))
            _y.append(p[k])
    if len(_x)>0:
        _x = np.array(_x)
        _y = np.array(_y)
        # -- find closest phase offset
        dphi = np.array([_y[np.argmin(np.abs(_x-m))] for m in mjd])
    else:
        dphi = __period*0.0

    return ((__mjd_mjd0+__corr)/__period + dphi)%1.0, __period

def cleanFourierCoef(aBW):
    """
    set phases between 0 and 2pi and corrects negative amplitudes
    """
    for k in aBW.keys():
        if 'VPULS A' in k and k.split('A')[1]!='0':
            aBW['VPULS PHI'+k.split('A')[1]]=(aBW['VPULS PHI'+k.split('A')[1]]+
                                              np.pi/2*(np.sign(aBW[k])-1))%(2*np.pi)
            aBW[k] = np.abs(aBW[k])
        if 'TEFF A' in k and k.split('A')[1]!='0':
            aBW['TEFF PHI'+k.split('A')[1]]=(aBW['TEFF PHI'+k.split('A')[1]]+
                                              np.pi/2*(np.sign(aBW[k])-1))%(2*np.pi)
            aBW[k] = np.abs(aBW[k])
    return aBW

def dephaseParam(a, dphi):
    """
    dephase the model by dphi (0->1). This is usefull to phase the model so maximum luminosity is attained for phi==0
    """
    a['MJD0'] -= dphi*a['PERIOD']
    if any(['VPULS VAL' in x for x in a.keys()]):
        # -- VPULS in spline
        for k in a.keys():
            if 'VPULS PHI' in k:
                a[k] += dphi
    else:
        # -- VPULS in Fourier
        for k in a.keys():
            if 'VPULS PHI' in k:
                a[k] -= int(k.split('PHI')[1])*2*np.pi*dphi
                a[k] = a[k]%(2*np.pi)
    if any(['TEFF VAL' in x for x in a.keys()]):
        # -- TEFF in spline
        for k in a.keys():
            if 'TEFF PHI' in k:
                a[k] += dphi
    else:
        # -- TEFF in Fourier
        for k in a.keys():
            if 'TEFF PHI' in k:
                a[k] -= int(k.split('PHI')[1])*2*np.pi*dphi
                a[k] = a[k]%(2*np.pi)
    return a

def _make_photometryGrid(filtname, teff=None, logg=None, metal=None):
    """
    Teff and logg are 1D array. Returns grid of magnitudes for 1mas star
    """
    if teff is None:
        if __SEDmodel=='atlas9' or __SEDmodel=='BOSZ':
            teff = atlas9._teff
        elif __SEDmodel=='phoenix2':
            teff = phoenix2.data['Teff']
            teff = np.array(teff)
            teff = teff[teff<8000]
        teff.sort()
    if logg is None:
        if __SEDmodel=='atlas9' or __SEDmodel=='BOSZ':
            logg = atlas9._logg
        elif __SEDmodel=='phoenix2':
            logg = phoenix2.data['logg']
        logg.sort()
    if metal is None:
        if __SEDmodel=='atlas9' or __SEDmodel=='BOSZ':
            metal = atlas9._metal
        metal.sort()

    res = np.zeros((len(teff), len(logg), len(metal)))
    for i,t in enumerate(teff):
        for j,g in enumerate(logg):
            for k,m in enumerate(metal):
                res[i,j,k] = photometrySED(1.0, t, filtname,
                                           logg=g, metal=m,
                                           useGrid=False)
    return {'mag':res,
            'teff':teff,
            'logg':logg,
            'metal':metal}

def _makeAll_photometryGrid():
    res = {}
    global __SEDmodel # tell which grid of model to use
    print '='*5, 'computing photometric grid for', __SEDmodel, '='*5
    for f in np.sort(photfilt2._data.keys()): # for each filtername
        print f
        res[f]=_make_photometryGrid(f)
    return res

def photometrySED(diam, teff, filtname, metal=0.0, plot=False, Nwl=100,
                  logg=1.0, SED=False, useGrid=True, jy=False):
    """
    compute photometry for diam (mas), teff (K) and filter name. If
    SED==True, returns the wavelength table and SED (in W/m2/um),
    instead of the integrated value.
    """
    global __SEDmodel # tell which grid of model to use
    global __MAGgrid, _dir_data

    if isinstance(diam, list):
        if not isinstance(logg, list):
            logg = logg*np.ones(len(diam))
        res = 0.0
        for k in range(len(diam)):
            res += 10.**(-photometrySED(diam[k], teff[k],
                                          filtname, metal=metal,
                                          Nwl=Nwl, logg=logg[k],
                                          SED=False)/2.5)
        return -2.5*np.log10(res)

    if useGrid:
        if filtname in __MAGgrid.keys():
            # -- find closests metalicity:
            k = np.abs(metal-__MAGgrid[filtname]['metal']).argsort()
            m0 = RectBivariateSpline(__MAGgrid[filtname]['teff'],
                                __MAGgrid[filtname]['logg'],
                                __MAGgrid[filtname]['mag'][:,:,k[0]],
                                kx=1,ky=1)(teff, logg) - 5*np.log10(np.abs(diam))
            m1 = RectBivariateSpline(__MAGgrid[filtname]['teff'],
                                __MAGgrid[filtname]['logg'],
                                __MAGgrid[filtname]['mag'][:,:,k[1]],
                                kx=1,ky=1)(teff, logg) - 5*np.log10(np.abs(diam))
            res = m0 + (metal - __MAGgrid[filtname]['metal'][k[0]])*(m1-m0)/\
                            (__MAGgrid[filtname]['metal'][k[1]] -
                             __MAGgrid[filtname]['metal'][k[0]])
            if jy:
                return photfilt2.convert_mag_to_Jy(res, filtname)
            return res
        else:
            print 'WARNING:', filtname, 'not in the grid!'

    # -- wavelength range of the filter
    wl = np.linspace(photfilt2.wavelRange(filtname)[0],
                     photfilt2.wavelRange(filtname)[1],
                     Nwl)

    # -- energy counter
    T = photfilt2.Transmission(filtname)(wl) # includes earth atmo is needed

    # -- photon counting!
    if photfilt2.effWavelength_um(filtname)<3.:
        T *= wl

    # -- stellar spectrum
    if __SEDmodel=='atlas9' or __SEDmodel=='BOSZ':
        import atlas9
        F = atlas9.flambda(wl, teff, logg, metal=metal)

    elif __SEDmodel=='phoenix2':
        F = phoenix2.flambda(wl*n_air_P_T(wl), teff, logg)

    # -- facilities in space or which already took into account atmo
    # space = ['IRAS', 'Spitzer', 'Hipparcos', '2MASS', 'DENIS', 'TYCHO',
    #         'TESS', 'GAIA', 'DIRBE']
    # if not any([s.lower() in filtname.lower() for s in space]):
    #     F *= photfilt2._atmoTrans(wl)

    # -- integrated
    iF = np.trapz(F*T, wl)/np.trapz(T, wl)

    # -- solid angle
    iF *= np.pi*(diam/2.0)**2 # angular surface, in mas**2
    iF *= (np.pi/(180.0*3600*1000))**2 # conversion in rad**2

    # -- erg/cm2/s/m -> W/m2/um
    iF *= 10**-9

    if plot:
        plt.clf()
        plt.plot(wl, F)
        plt.plot(wl, T/T.max()*F.mean())
        plt.title(__SEDmodel)

    if SED:
        return wl, iF
    else:
        if jy:
            return photfilt2.convert_Wm2um_to_Jy(iF,
                            photfilt2.effWavelength_um(filtname))

        else:
            return photfilt2.convert_Wm2um_to_mag(iF, filtname)

def minMag(a, filters):
    """
    a -> parameters for a model
    filters -> list of filters
    """
    obs = []
    plt.figure(0)
    plt.clf()
    phis = np.linspace(-0.5,0.5,2000)
    col = plt.cm.get_cmap('jet')
    data = []
    plt.subplot(121)
    for i, f in enumerate(filters):
        obs = []
        for phi in phis:
            # -- fake data
            obs.append((phi%1., 'mag', f, 0.0, 1.0))
        mod = np.array(model(obs, a, plot=False))
        plt.plot(phis, mod-mod.mean(), label=f,
                 color=col(i/float(len(filters)-1)),
                 linewidth=3, alpha=0.5)
        print '%-20s (%5.3fum): %7.4f'%(f, photfilt2.effWavelength_um(f),
                                       phis[np.argmin(mod)])
        data.append((photfilt2.effWavelength_um(f),
                                       phis[np.argmin(mod)]))
    plt.legend(loc='upper right', prop={'size':10})
    plt.xlabel('phase')
    plt.ylabel('mag - <mag>')
    plt.xlim(phis.min(), phis.max())

    plt.subplot(122)
    plt.plot([d[0] for d in data], [d[1] for d in data],
             'k-', alpha=0.5)
    for i, d in enumerate(data):
        plt.plot(d[0], d[1], 'o', color=col(i/float(len(filters)-1)),
                 markersize=10, alpha=0.5)

    plt.xlabel('effective wavelength (um)')
    plt.ylabel('phase of minimum')
    plt.grid()
    plt.legend()
    return

def testColors():
    """
    Compare colors computed from ATLAS9 models and the tabulation found in
    Astrophysical Quantities
    """

    # -- Table 15.7 in Astrophysical quantities:
    _T = [4310, 4550, 4700, 4930, 5190, 5370, 5750, 6360,
          7030, 7460, 8610, 9380, 9980, 11100, 13600]
    B_V = [1.36, 1.25, 1.14, 1.02, 0.87, 0.76, 0.56, 0.32,
            0.23, 0.17, 0.09, 0.03, -0.01, -0.03, -.10]
    U_B = [1.32, 1.17, 1.07, 0.83, 0.63, 0.52, .41, 0.27,
            0.18, 0.15, -0.08, -0.25, -0.38, -0.55, -0.72]
    V_R = [0.85, 0.76, 0.69, 0.67, 0.58, 0.51, 0.45, 0.35,
            0.26, 0.21, 0.12, 0.07, 0.03, 0.02, -0.05]

    # -- Table 7.8 in Astrophysical quantities:
    _T2 = [4330, 4420, 4590, 4980, 5510, 6100, 6640, 7170,
            7700, 8510, 9080, 9230, 9730, 10500, 11200, 12000,
            12700, 13400]
    V_K = [2.28, 2.15, 1.99, 1.67, 1.44, 1.21, 0.93, 0.75,
            0.64, 0.48, 0.32, 0.26, 0.19, 0.13, 0.07, 0.01,
            -0.07, -0.13]
    J_H = [0.49, 0.46, 0.43, 0.38, 0.33, 0.28, 0.22, 0.18,
            0.15, 0.13, 0.12, 0.11, 0.09, 0.08, 0.07, 0.06,
            0.04, 0.01]
    H_K = [0.13, 0.12, 0.11, 0.09, 0.08, 0.07, 0.06, 0.05,
           0.04, 0.02, 0.01, 0.0, -0.01, -0.02, -0.02, -0.02,
           -0.02, 0.0]

    try:
        teff = __MAGgrid['I_COUSIN']['teff']
    except:
        if __SEDmodel=='atlas9' or __SEDmodel=='BOSZ':
            teff = atlas9._teff
        elif __SEDmodel=='phoenix2':
            teff = phoenix2.data['Teff']

    teff = teff[teff>=np.min(_T)-500]
    teff = teff[teff<=np.max(_T)+500]

    U = np.array([photometrySED(1.0, t, 'U_JOHNSON',
                 logg=1.5, useGrid=True)[0] for t in teff])
    B = np.array([photometrySED(1.0, t, 'B_JOHNSON',
                 logg=1.5, useGrid=True)[0] for t in teff])
    V = np.array([photometrySED(1.0, t, 'V_JOHNSON',
                 logg=1.5, useGrid=True)[0] for t in teff])
    R = np.array([photometrySED(1.0, t, 'R_JOHNSON',
                 logg=1.5, useGrid=True)[0] for t in teff])
    J = np.array([photometrySED(1.0, t, 'J_CTIO',
                 logg=1.5, useGrid=True)[0] for t in teff])
    H = np.array([photometrySED(1.0, t, 'H_CTIO',
                 logg=1.5, useGrid=True)[0] for t in teff])
    K = np.array([photometrySED(1.0, t, 'K_CTIO',
                 logg=1.5, useGrid=True)[0] for t in teff])

    plt.figure(0)
    plt.clf()
    ax0 = plt.subplot(231)
    plt.plot(teff, U - B, '-oc', label='U-B (ATLAS9)', alpha=0.5)
    plt.plot(_T, U_B, '-cv', linewidth=3, label='U-B (Astr. Q)')
    plt.legend()
    plt.xlabel('Teff')

    plt.subplot(232, sharex=ax0)
    plt.plot(teff, B - V, '-ob', label='B-V (ATLAS9)', alpha=0.5)
    plt.plot(_T, B_V, '-bv', linewidth=3, label='B-V (Astr. Q)')
    plt.legend()
    plt.xlabel('Teff')

    plt.subplot(233, sharex=ax0)
    plt.plot(teff, V - R, '-o', color='g', label='V-R (ATLAS9)', alpha=0.5)
    plt.plot(_T, V_R, '-v', color='g', linewidth=3, label='V-R (Astr. Q)')
    plt.legend()
    plt.xlabel('Teff')

    plt.subplot(234, sharex=ax0)
    plt.plot(teff, J - H, '-o', color='y', label='J-H (ATLAS9)', alpha=0.5)
    plt.plot(_T2, J_H, '-v', color='y', linewidth=3, label='J-H (Astr. Q)')
    plt.legend()
    plt.xlabel('Teff')

    plt.subplot(235, sharex=ax0)
    plt.plot(teff, H - K, '-o', color='orange', label='H-K (ATLAS9)', alpha=0.5)
    plt.plot(_T2, H_K, '-v', color='orange', linewidth=3, label='H-K (Astr. Q)')
    plt.legend()
    plt.xlabel('Teff')

    plt.subplot(236, sharex=ax0)
    plt.plot(teff, V - K, '-o', color='r', label='V-K (ATLAS9)', alpha=0.5)
    plt.plot(_T2, V_K, '-v', color='r', linewidth=3, label='V-K (Astr. Q)')
    plt.legend()
    plt.xlabel('Teff')
    return

def allFilter():
    wl = np.linspace(0.3, 3.5, 1000)

    filters = ['B_MVB_TYCHO', 'B_W', 'B_GCPD', 'B_ST_GCPD',
                'HP_MVB_HIPPARCOS', 'V_MVB_TYCHO', 'V_W', 'V_GCPD',
                'Y_ST_GCPD', 'R_GCPD', 'I_GCPD',
                'J_CTIO', 'H_CTIO', 'K_CTIO',]

    ref = [1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]
    for i,f in enumerate(filters):
        print f.split('_')[0]+'$_\mathrm{'+f.split('_')[1]+'}$', '&',
        print round(photfilt2.effWavelength_um(f)*1000, 1), '&',
        tmp ='%5.3e'%(photfilt2.zeroPoint_Wm2um(f))
        tmp = tmp.split('e-')[0]+r'\times10^{-'+tmp.split('e-')[1]+'}'
        print '$'+tmp+'$', '&',
        # print '%5.3f, %5.3f, %5.3f'%(photometrySED(1.0, 4500.0, f, logg=1.5),
        #                              photometrySED(1.0, 5500.0, f, logg=1.5),
        #                              photometrySED(1.0, 6500.0, f, logg=1.5),), '&',
        # print '%5.3f, %5.3f, %5.3f'%(Alambda_Exctinction(f, EB_V=1, Rv=3.1, Teff=4500),
        #                             Alambda_Exctinction(f, EB_V=1, Rv=3.1, Teff=5500),
        #                              Alambda_Exctinction(f, EB_V=1, Rv=3.1, Teff=6500)),
        print photfilt2._data[f]['ID'], '&',
        print photfilt2._data[f]['Description'],
        print '&', '(%d)'%ref[i], r'\\'

    #
    for i,f in enumerate(filters):
        print f.split('_')[0]+'$_\mathrm{'+f.split('_')[1]+'}$', '&',
        print '%5.3f, %5.3f, %5.3f'%(photometrySED(1.0, 4500.0, f, logg=1.5),
                                     photometrySED(1.0, 5500.0, f, logg=1.5),
                                     photometrySED(1.0, 6500.0, f, logg=1.5),), '&',
        print '%5.3f, %5.3f, %5.3f'%(Alambda_Exctinction(f, EB_V=1, Rv=3.1, Teff=4500),
                                    Alambda_Exctinction(f, EB_V=1, Rv=3.1, Teff=5500),
                                     Alambda_Exctinction(f, EB_V=1, Rv=3.1, Teff=6500)),
        print '\\'

    plt.close(0)
    plt.figure(0, figsize=(12,4))
    ax = plt.subplot(111)
    p = 0.2
    for i, f in enumerate(filters):
        tmp = photfilt2.Transmission(f)(wl)
        plt.plot(wl**p, tmp/tmp.max(), linewidth=2,
                 color=plt.cm.rainbow(i/float(len(filters))),
                 linestyle='dashed' if "WAL" in f else '-',
                 label=f)
    plt.legend()
    X = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
    ax.set_xticks([x**p for x in X])
    ax.set_xticklabels(['%3.1f'%x for x in X])
    plt.xlim(wl.min()**p, wl.max()**p)
    return

def magmodel(teffMin=4700., teffMax=7000.):
    filters = ['B_JOHNSON', 'V_JOHNSON', 'R_BESSELL', 'I_BESSELL',
                'J_CTIO', 'H_CTIO', 'K_CTIO', ]
    Teff = np.linspace(teffMin, teffMax, 20)
    logTeff = np.log10(Teff)
    plt.figure(0, figsize=(4,8))
    plt.clf()
    ax0 = plt.subplot(211)
    ax1 = plt.subplot(212, sharex=ax0)
    t0 = np.log10(5400.)
    X = -2.5*(logTeff-t0)
    X = Teff - 0.5*(teffMin+teffMax)

    print 't = -2.5(log10(Teff) - %5.3f)'%(t0)
    for i,f in enumerate(filters):
        m = np.array([photometrySED(1.0, t, f, logg=1.5)[0,0] for t in Teff])
        print f.split('_')[0]+r'&',
        for n in [1,2]:
            c = np.polyfit(X, m, n)
            for k in range(len(c)):
                c[k] = round(c[k], 2 if k!=len(c)-1 else 3)
            for k,x in enumerate(c[::-1]):
                if k==0:
                    print '$%4.3f'%x,
                elif k==1:
                    print '%s%4.2f t'%('+' if x>=0 else '', x),
                else:
                    print '%s%4.2f t^%d'%('+' if x>=0 else '', x,k),
            # -- residuals
            res =  m-np.polyval(c, X)
            print r'$ & %4.3f '%(res.ptp()),
            if n!=2:
                print '&',
        print r'\\'

        ax0.plot(Teff, m,  '.-', label=f,
                 color=plt.cm.rainbow(i/float(len(filters))))
        ax1.plot(Teff, res,  '.-', label=f,
                 color=plt.cm.rainbow(i/float(len(filters))))
    ax0.legend(loc='upper right', prop={'size':8}, numpoints=1)
    ax0.set_ylabel('magnitude')
    ax1.set_ylabel('residuals to poly-%d fit'%n)
    ax1.set_xlabel('log10 Teff')
    return

# -- A&A 428, 587-593 (2004), table 4
K04 = {('B','V'):(-0.2944, 3.8813, 0.017),
       ('B','R'):(-0.1978, 3.8719, 0.008),
       ('B','I'):(-0.1800, 3.9283, 0.015),
       ('B','J'):(-0.1401, 3.9297, 0.015),
       ('B','H'):(-0.1224, 3.9423, 0.014),
       ('B','K'):(-0.1199, 3.9460, 0.015),
       ('V','R'):(-0.3789, 3.8516, 0.014),
       ('V','I'):(-0.3077, 3.9617, 0.016),
       ('V','J'):(-0.1759, 3.9407, 0.015),
       ('V','H'):(-0.1379, 3.9490, 0.014),
       ('V','K'):(-0.1336, 3.9530, 0.015),
       ('R','I'):(-1.2894, 4.3248, 0.060),
       ('R','J'):(-0.2240, 3.9532, 0.006),
       ('R','H'):(-0.1458, 3.9426, 0.005),
       ('R','K'):(-0.1386, 3.9430, 0.005),
       ('I','J'):(-0.2854, 3.9323, 0.013),
       ('I','H'):(-0.1713, 3.9491, 0.010),
       ('I','K'):(-0.1630, 3.9458, 0.013),
       ('J','H'):(-0.2988, 3.9679, 0.015),
       ('J','K'):(-0.2614, 3.9722, 0.016),
       ('H','K'):(-2.3858, 4.0653, 0.029),
    }

def allSBRelations(teffMin=4800., teffMax=7000., forceLin=False):
    filters = ['B_GCPD', 'V_GCPD', 'R_GCPD', 'I_GCPD',
                'J_CTIO', 'H_CTIO', 'K_CTIO']


    # filters = ['B_MVB_TYCHO', 'V_MVB_TYCHO', 'R_JOHNSON',
    #         'I_JOHNSON', 'K_CTIO']

    print 'Surface Brightness for Teff=%4.0f-%4.0fK'%(teffMin, teffMax)
    if forceLin:
        print '  F1 = a1*(M1-M2) + a0'
    else:
        print '  F1 = a2*(M1-M2)**2 + a1*(M1-M2) + a0'

    print '     = 4.2207 - 0.1*M1 - 0.5log(diam)'
    print '-'*(6+8*(len(filters)-1))
    #print ' '*8,
    print 'M1 \\ M2 ',
    for f2 in filters[1:]:
        print '%-7s'%f2.split('_')[0],
    print ''
    print '-'*(6+8*(len(filters)-1))
    for k1, f1 in enumerate(filters[:-1]):
        if forceLin:
            line =  ['%1s: a1'%f1.split('_')[0], '   a0', '  ptp']
        else:
            line =  ['%1s: a2'%f1.split('_')[0], '   a1', '   a0', '  ptp']
        for k2, f2 in enumerate(filters[1:]):
            if k1<=k2:
                tmp = SBRelation(f1, f2, teffMin=teffMin, teffMax=teffMax,
                                 forceLin=forceLin)
                if not forceLin:
                    if 'a2' in tmp[0].keys():
                        line[0] += ' %7.4f'%tmp[0]['a2']
                    else:
                        line[0] += '   0    '
                i = 1-int(forceLin)
                line[i] += '  %6.3f'%tmp[0]['a1']
                line[i+1] += '  %6.3f'%tmp[0]['a0']
                line[i+2] += '   %4.2f%%'%(100*tmp[1])
            else:
                for i in range(len(line)):
                    line[i] += ' '+'.'*7
        for i in range(len(line)):
            print line[i]
        print '-'*(6+8*(len(filters)-1))
    return

def SBRelation(filter1, filter2, teffMin=5000., teffMax=7000.,
                verbose=False, plot=False, forceLin=True):
    """
    A*(M1-M2)+B = 4.2207 - 0.1*M1 + 0.5*log10(diam)

    (A+0.1)*M1 - A*M2 + (B-4.2207) = 0.5*log10(diam)

    if diam == 0

    (A+0.1)/A*M1 + (B-4.22027)/A = M2

    c[0]*M1 + c[1] = M2
    """
    # -- fit
    N=100
    #Teff = np.linspace(teffMin, teffMax, N)
    Teff = np.logspace(np.log10(teffMin), np.log10(teffMax), N)
    diam = 1+0*np.random.rand(len(Teff))
    mag1 = np.array([photometrySED(diam[k], Teff[k], filter1, logg=1.5)[0,0] for k in range(N)])
    mag2 = np.array([photometrySED(diam[k], Teff[k], filter2, logg=1.5)[0,0] for k in range(N)])
    f = dpfit.leastsqFit(diamSB, [mag1, mag2], {'a1':1.0, 'a0':1.0}, diam, verbose=verbose)
    form = r'F'+filter1+' = %4.3f*('+filter1+' - '+filter2+') + %4.3f'
    form = form%(f['best']['a1'], f['best']['a0'])
    if ((diam-f['model'])/diam).ptp()>0.001 and not forceLin:
        f = dpfit.leastsqFit(diamSB, [mag1, mag2], {'a2':0.0, 'a1':1.0, 'a0':1.0},
                             diam, verbose=verbose)
        form = r'F'+filter1+' = %4.3f*('+filter1+' - '+filter2+')**2 + %4.3f*('+filter1+' - '+filter2+') + %4.3f'
        form = form%(f['best']['a2'], f['best']['a1'], f['best']['a0'])


    c = K04[(filter1[0], filter2[0])]
    print c
    diamK04 = diamSB((mag1, mag2), {'a0':c[1], 'a1':c[0]} )

    if plot:
        plt.close(0)
        plt.figure(0, figsize=(5,3))
        plt.plot(mag1-mag2, 100*(diam-f['model'])/diam, '-k', label='SPIPS')
        plt.plot(mag1-mag2, 100*(diam-diamK04)/diam, '-r', label='Kervella+ 04')

        plt.xlabel(filter1+' - '+filter2)
        plt.ylabel('relative error on diameter (%)')
        #plt.text(plt.xlim()[1]-(plt.xlim()[1]-plt.xlim()[0])*0.02,
        #     plt.ylim()[1]-(plt.ylim()[1]-plt.ylim()[0])*0.02,
        #     form, ha='right', va='top')
        plt.legend()
        plt.hlines((-1,1), (mag1-mag2).min(), (mag1-mag2).max(),
                    linestyle='dotted')
        plt.ylim(-10,10)
    if verbose:
        print form
    return f['best'], ((diam-f['model'])/diam).ptp()

def diamSB(mags, c):
    tmp = -4.2207+0.1*mags[0]
    #tmp = 0.1*mags[0]
    for k in c.keys():
        tmp += c[k]*(mags[0]-mags[1])**float(k[1:])
    return 10**(-tmp/0.5)

def n_air_P_T(wl, P=743.2, T=290, e=74.32):
    """
    wl in um
    P, in mbar (default 743.2mbar)
    T, in K    (default 290.0K)
    e partial pressure of water vapour, in mbar (default 74.32)
    """
    return 1 + 1e-6*(1+0.00752/np.array(wl)**2)*\
           (77.6*np.array(P)/np.array(T)
                             + 3.73e-5*e/np.array(T)**2)

def Alambda_Exctinction(wl, EB_V=0.0, Rv=3.1, Teff=6000.):
    """
    wl in microns of filter name. Returns A(lambda) for each wavelength.

    Here A(lambda) is the extinction, in magnitudes, or 1.086tau(lambda), where tau
    is the optical depth in dust.

    see Astrophysical Quantities
        8.4 INTERSTELLAR EXTINCTION IN THE ULTRAVIOLET
        21.2 GALACTIC INTERSTELLAR EXTINCTION [[[not used anymore!]]]

    TABLE 3 and 4 from FITZPATRICK: 1999 PASP, 111-63 for visible, IR
    """
    global __SPE
    ### res == A_lambda/E(B-V)

    if isinstance(Teff, list):
        return [Alambda_Exctinction(wl, EB_V=EB_V, Rv=Rv, Teff=t) for t in Teff]
    if isinstance(Teff, np.ndarray):
        #print 'DEBUG:', np.isscalar(Teff), Teff.shape
        try:
            return np.array([Alambda_Exctinction(wl, EB_V=EB_V, Rv=Rv, Teff=t) for t in Teff])
        except:
            return Alambda_Exctinction(wl, EB_V=EB_V, Rv=Rv, Teff=float(Teff))

    if isinstance(wl, str):
        # -- wl is the filter name
        l = np.linspace(photfilt2.wavelRange(wl)[0],
                        photfilt2.wavelRange(wl)[1],
                        100)
        # -- interpolation:
        k0 = np.array(__SPE.keys())[np.abs(Teff-np.array(__SPE.keys())).argsort()[0]]
        s = np.interp(l, __SPE[k0]['WAVEL'], __SPE[k0]['FLAMBDA'])
        t = photfilt2.Transmission(wl)(l)
        a0 = np.sum(Alambda_Exctinction(l, EB_V=EB_V, Rv=Rv)*s*t*l)/np.sum(s*t*l)

        k1 = np.array(__SPE.keys())[np.abs(Teff-np.array(__SPE.keys())).argsort()[1]]
        s = np.interp(l, __SPE[k1]['WAVEL'], __SPE[k1]['FLAMBDA'])
        t = photfilt2.Transmission(wl)(l)
        a1 = np.sum(Alambda_Exctinction(l, EB_V=EB_V, Rv=Rv)*s*t*l)/np.sum(s*t*l)
        return a0 + (Teff-k0)*(a1-a0)/(k1-k0)

    if np.isscalar(wl):
        wl = np.array([wl])

    if isinstance(wl, list):
        wl = np.array(wl)
    if isinstance(wl, float):
        wl = np.array([wl])

    res = np.zeros(len(wl))

    if False:
        ### 21.6 GALACTIC INTERSTELLAR EXTINCTION
        ### Rv = A(V)/[A(B) - A(V)] = A(V)/E(B-V) -> A(V) = Rv*E(B-V)
        wl_31 = [0.002, 0.004, 0.023, 0.041, 0.073, 0.091, 0.12, 0.13, 0.15, 0.18, 0.20, 0.218, 0.24,
               0.26, 0.28, 0.33, 0.365, 0.44, 0.55, 0.7, 0.9, 1.25, 1.65, 2.2, 3.4, 5,
               7, 9, 9.7, 10, 12, 15, 18, 20, 25, 35, 60, 100, 250]
        ### A_lambda / A_V for Rv = 3.1
        AlAv_31 = [0.38, 0.96, 2.06, 2.58, 5.38, 4.85, 3.58, 3.12, 2.66, 2.52, 2.84, 3.18, 2.54,
               2.15, 1.94, 1.65, 1.56, 1.31, 1.0, 0.749, 0.479, 0.282, 0.176, 0.108, 0.051, 0.027,
               0.020, 0.042, 0.059, 0.054, 0.028, 0.015, 0.023, 0.021, 0.014, 3.7e-3, 2e-3, 1.2e-3, 4.2e-4]
        ### extrapolation in log/log
        res_31 = np.exp(np.interp(np.log(wl), np.log(wl_31),np.log(np.array(AlAv_31))))*3.1

        wl_50 = [0.12, 0.13, 0.15, 0.18, 0.20, 0.218, 0.24,
               0.26, 0.28, 0.33, 0.365, 0.44, 0.55, 0.7, 0.9, 1.25, 1.65, 2.2, 3.4, 5,
               7, 9, 9.7, 10, 12, 15, 18, 20, 25, 35, 60, 100, 250]
        ### A_lambda / A_V for Rv = 3.1
        AlAv_50 = [1.74, 1.6, 1.49, 1.52, 1.74, 1.97, 1.68,
                1.5, 1.42, 1.35, 1.33, 1.20, 1.0, .794, .556, .327, .204, .125, .059, .031,
                0.23, 0.51, 0.68, .063, 0.032, 0.017, 0.027, 0.025, 0.016, 0.0042, 0.0023, 0.0013, 0.0049]
        ### extrapolation in log/log
        res_50 = np.exp(np.interp(np.log(wl), np.log(wl_50),np.log(np.array(AlAv_50))))*5.0

        res = res_31 + (Rv-3.1)*(res_50-res_31)/(5.0-3.1)
    else:
        # A(lambda)/E(B-V) as a function of 1/lambda (1/um), cubic spline:
        # TABLE 3 and 4 from FITZPATRICK: 1999 PASP, 111-63
        x = [0, 0.377, 0.820, 1.667, 1.828, 2.141, 2.433, 3.704, 3.846]
        y = [0, 0.265, 0.829, -0.426+1.0044*Rv, -0.050+1.0016*Rv, 0.701+1.0016*Rv,
             1.208+1.0032*Rv-0.00033*Rv**2, 6.265, 6.591]
        res[wl>=0.26] = interp1d(x, y, kind='cubic')(1/wl[wl>=0.26])
        #res[wl>=0.26] = np.interp(1/wl[wl>=0.26], x, y)

    ### 8.4 INTERSTELLAR EXTINCTION IN THE ULTRAVIOLET
    wn = 1./wl # wavenumber
    res[(wn>=2.70)*(wn<=10.)] = (1.56 + 1.048*wn[(wn>=2.70)*(wn<=10.)] +\
                    1.01/((wn[(wn>=2.70)*(wn<=10.)]-4.60)**2+0.280))*Rv/3.1
    res[(wn>=3.65)*(wn<=10.)] = (2.29 + 0.848*wn[(wn>=3.65)*(wn<=10.)] +\
            1.01/((wn[(wn>=3.65)*(wn<=10.)]-4.60)**2+0.280))*Rv/3.1
    res[(wn>=7.14)*(wn<=10.)] = (16.17 - 3.20*wn[(wn>=7.14)*(wn<=10.)] +
                                 0.2975*wn[(wn>=7.14)*(wn<=10.)]**2)*Rv/3.1
    # -- final result
    if len(wl)==1:
        return res[0]*EB_V
    else:
        return res*EB_V

if False:
    filt = ['B_WALRAVEN.B', 'B_JOHNSON', 'V_WALRAVEN.V', 'V_JOHNSON',
            'R_COUSIN', 'I_COUSIN', 'J_CTIO', 'H_CTIO', 'K_CTIO']
    print 'filter eff.wave. Al/Av'
    wv = photfilt2.effWavelength_um('V_JOHNSON')
    for f in filt:
        wl = photfilt2.effWavelength_um(f)
        print '%12s %5.3f %5.3f'%(f, wl, Alambda_Exctinction(f, 1.0, Rv=3.1)/
                                         Alambda_Exctinction('V_JOHNSON', 1.0, Rv=3.1))

# == Teff=6000K, logg=1.5 spectra for computing reddenning:
filename = 'BW2_ATLAS9_aLambdaSPE.dpy'
if os.path.exists(os.path.join(_dir_data, filename)):
    print 'loading', filename, '...',
    f = open(os.path.join(_dir_data, filename))
    __SPE = cPickle.load(f)
    f.close()
    print 'OK'
else:
    print 'reloading atlas9 files...',
    import atlas9
    __SPE={}
    for T in [4000, 4500, 5000, 5500, 6000, 6500, 7000]:
        __SPE[T] = atlas9.ReadOneFile(os.path.join(_dir_data,
                                       'ATLAS9/gridp00k2odfnew/fp00t%sg15k2odfnew.dat'%str(T)))
    f = open(os.path.join(_dir_data, filename), 'wb')
    cPickle.dump(__SPE, f, 2)
    f.close()
    print 'saving', filename
# =========================================================

# == Computation of the GRID of magnitudes at theta=1mas ========
# -- check if variables already loaded:
# -- try to read from file

filename = 'BW2_maggrid_%s.dpy'%__SEDmodel

try:
    print 'loading', filename,' ...',
    f = open(os.path.join(_dir_data, filename))
    __MAGgrid = cPickle.load(f)
    f.close()
    print 'OK'
    _computeGrid = False
    # -- check we have all the filters
    for k in photfilt2._data.keys():
        if not k in __MAGgrid.keys():
            print ' >> filter', k, 'is missing in the grid -> recomputing grid now!'
            _computeGrid = True
except:
    _computeGrid = True

#_computeGrid = True
if _computeGrid:
    # -- compute grid as it does not seem to exist
    print ' >> generating file:', filename
    if __SEDmodel=='atlas9' or __SEDmodel=='BOSZ':
        import atlas9
        if atlas9.useBOSZ:
            filename = filename.replace('_atlas9', '_BOSZ')
    elif __SEDmodel=='phoenix2':
        import phoenix2
    __MAGgrid = _makeAll_photometryGrid()
    print ' >> saving', filename
    f = open(os.path.join(_dir_data, filename), 'wb')
    cPickle.dump(__MAGgrid, f, 2)
    f.close()

# =======================================================================

def testWesenheit(B_V0=1.0, Rv=3.1):
    """
    test the robustness of the Wesenheit relations to reddenning.
    """
    EB_V = np.linspace(0, 0.6, 7)
    print '# reddennings for Rv=', Rv
    print '# E(B-V) B      V      I      WVI    WBI    WBVI   WBV'
    for e in EB_V:
        MB = Alambda_Exctinction(0.44, e, Rv=Rv)
        MV = Alambda_Exctinction(0.551, e, Rv=Rv)
        MI = Alambda_Exctinction(0.806, e, Rv=Rv)
        form = '  %4.2f  '+'%6.3f '*7
        # -- classical relations
        print form%(e, MB, MV, MI, MI-1.53*(MV-MI), MI-0.83*(MB-MI),
            MI-1.96*(MB-MV), MV-3.24*(MB-MV))
    return

def findWesenheit(B1='V_JOHNSON', B2='I_BESSELL', EB_Vmax=0.6, Rv=3.1, plot=True):
    """
    comes from hypothesis that Rv = Av/E(B-V) so the expression "V - Rv*(B-V)"
    is free of redenning. Using this function, one can compute the Wesenheit
    relation for various pairs of photometric systems.

    Weseinheit ~= Intrinsic, Essential in German
    """
    EB_V = np.linspace(0, EB_Vmax, 20)
    if isinstance(B1, float):
        wl1 = B1
    else:
        wl1 = photfilt2.effWavelength_um(B1)
    if isinstance(B2, float):
        wl2 = B2
    else:
        wl2 = photfilt2.effWavelength_um(B2)

    R1 = np.array([Alambda_Exctinction(wl1, e, Rv=Rv) for e in EB_V])
    R2 = np.array([Alambda_Exctinction(wl2, e, Rv=Rv) for e in EB_V])
    c1 = np.polyfit(EB_V, R1, 1)
    c2 = np.polyfit(EB_V, R2, 1)

    s = '%s - %4.2f*%s'%(B1, c1[0]/c2[0], B2)

    if plot:
        plt.close(0)
        plt.figure(0)
        plt.subplot(211)
        plt.plot(EB_V, R1, 'ob', label=B1)
        plt.plot(EB_V, np.polyval(c1, EB_V), '-b')
        plt.plot(EB_V, R2, 'or', label=B2)
        plt.plot(EB_V, np.polyval(c2, EB_V), '-r')
        plt.legend(loc='lower right')
        plt.ylabel(r'A$_\lambda$')

        plt.subplot(212)
        plt.plot(EB_V, R1 - c1[0]*R2/c2[0], 'ok-',
                 label=s)
        plt.legend(loc='lower right')
        plt.xlabel('E(B-V)')
        plt.ylabel(r'A$_\lambda$')

    print '%s %s %5.2f*(%s-%s)'%(B1, '+' if np.sign(c1[0]/(c1[0]-c2[0]))>0 else '-',
                                 np.abs(c1[0]/(c1[0]-c2[0])), B2, B1)
    print s

    print 'for E(B-V) from %4.2f to %4.2f'%(EB_V.min(), EB_V.max())
    return

def V2centro(baselines, params, wavel='FLUOR'):
    """
    centro symetric squared visibility function.

    params = {'UD DIAM':} # UD diam in mas
           = {'nu DIAM':, 'nu':} # nu=1 -> UD case; nu>1 LD case
           = {'nu DIAM':, 'nu':, 'CSE':} # CSE in % of flux

    wavel in um or name of passband (e.g. FLUOR)
    """
    c = np.pi*np.pi/(180*3600*1000.0)*1e6
    #baselines=np.array(baselines)
    #print baselines
    res = []
    # -- set v2 function
    if params.has_key('UD DIAM'):
        v2func  = lambda x: (2*special.j1(x*params['UD DIAM'])/(x*params['UD DIAM']))**2
    elif params.has_key('nu DIAM') and \
        params.has_key('nu') and \
        not params.has_key('CSE'):
        v2func  = lambda x: (special.gamma(params['nu']+1)*
                             special.jv(params['nu'],x*params['nu DIAM'])/
                             (x*params['nu DIAM']/2.)**params['nu'])**2
    elif params.has_key('nu DIAM') and params.has_key('nu') and params.has_key('CSE'):
        # -- complex visibility of the star
        Vs = lambda x: (special.gamma(params['nu']+1)*
                        special.jv(params['nu'],x*params['nu DIAM'])/
                        (x*params['nu DIAM']/2.)**params['nu'])
        # -- CSE from 3 to 4 stellar radii
        Dout = 5; Din = 4
        Vcse = lambda x: ((2*special.j1(x*Dout*params['nu DIAM'])/
                           (x*Dout*params['nu DIAM']))*Dout**2-
                          (2*special.j1(x*Din*params['nu DIAM'])/
                           (x*Din*params['nu DIAM']))*Din**2)/(Dout**2-Din**2)
        v2func = lambda x: (Vs(x) + params['CSE']/100.*Vcse(x))**2/\
                 (1 + params['CSE']/100.)**2
    else:
        print 'UNKNOWN squared visibility case'

    if isinstance(wavel, float):
        # monochromatic
        x =  c*baselines/wavel
        x += (x==0)*1e-6 # avoid /0
        res = v2func(x)
    if wavel=='FLUOR':
        # -- planck function
        _h = 6.63e-34
        _c = 3.e8
        _k = 1.38e-23
        c1 = 2*_h*_c**2
        c2 = _h*_c/_k
        Bl = lambda l,t: c1/(l*1e-6)**5*1./(np.exp(c2/(l*1e-6*t))-1)

        # transmission of FLUOR
        l=[1.85, 1.86966, 1.88931, 1.90897, 1.92862, 1.94828,
           1.96793, 1.98759, 2.00724, 2.0269, 2.04655, 2.06621,
           2.08586, 2.10552, 2.12517, 2.14483, 2.16448, 2.18414,
           2.20379, 2.22345, 2.2431, 2.26276, 2.28241, 2.30207,
           2.32172, 2.34138, 2.36103, 2.38069, 2.40034, 2.42]
        t =[0, 0.000151781, 0.00142689, 0.00873798, 0.0356695,
            0.1008, 0.200336, 0.288131, 0.338292, 0.374736,
            0.39877, 0.416136, 0.446993, 0.479307, 0.500674,
            0.513605, 0.519658, 0.514367, 0.493725, 0.465865,
            0.447081, 0.432149, 0.381693, 0.27687, 0.156203,
            0.0663728, 0.0199908, 0.00388601, 0.000443628,
            0]
        l = np.array(l); t = np.array(t)
        # -- V2_FLUOR = int(T**2 * WL**2 * V2) / int(T**2 * WL**2)
        if not np.isscalar(baselines):
            x = c*baselines[:,np.newaxis]/l[np.newaxis,:]
            x += (x==0)*1e-6 # avoid /0
            tmp = (Bl(l, 6000.)*t*l)**2
            #tmp = (t*l)**2
            res = np.sum(tmp[None,:]*v2func(x), axis=1)/np.sum(tmp)
        else:
            x = c*baselines/l
            x += (x==0)*1e-6 # avoid /0
            tmp = (Bl(l, 6000.)*t*l)**2
            #tmp = (t*l)**2
            res = np.sum(v2func(x)*tmp)/np.sum(tmp)
    return res

def V2udFluor(B, params):
    """
    # from Yorick code  "prisme.i":
    > V2_nu_FLUOR([100,200,300,400,500,600], [1., 6000, 1.])
    [0.879385,0.587678,0.279224,0.077648,0.00537152,0.00604318]
    """
    v2func  = lambda x: (2*special.j1(x*params['diam'])/(x*params['diam']))**2
    # -- planck function
    _h = 6.63e-34
    _c = 3.e8
    _k = 1.38e-23
    c1 = 2*_h*_c**2
    c2 = _h*_c/_k
    Bl = lambda l,t: c1/(l*1e-6)**5*1./(np.exp(c2/(l*1e-6*t))-1)
    c = np.pi*np.pi/(180*3600*1000.0)*1e6

    # -------------------------------------------------
    c *= 0.997 # to match original function in Yorick!
    # -------------------------------------------------

    # -- transmission of FLUOR
    l=[1.85,1.865,1.88,1.895,1.91,1.925,1.94,1.955,1.97,1.985,2,2.015,2.03,2.045,
       2.06,2.075,2.09,2.105,2.12,2.135,2.15,2.165,2.18,2.195,2.21,2.225,2.24,2.255,
       2.27,2.285,2.3,2.315,2.33,2.345,2.36,2.375,2.39,2.405,2.42,2.435,2.45]
    t =[0.0,8.38474e-05,0.00051966,0.00251838,0.00950313,0.0283148,0.0679561,
        0.132616,0.211101,0.278914,0.322857,0.353358,0.379684,0.397452,0.409636,
        0.428328,0.454399,0.478587,0.496058,0.507955,0.515948,0.519693,0.516744,
        0.504697,0.484915,0.463881,0.449165,0.4397,0.41994,0.370665,0.289627,0.195792,
        0.113061,0.0547599,0.0215168,0.00657107,0.00149611,0.000243976,2.76526e-05,
        2.15892e-06,0.0]
    l = np.array(l); t = np.array(t)

    if 'Teff' in params.keys():
        Teff = params['Teff']
    else:
        Teff = 6000.

    if not np.isscalar(B):
        x = c*B[:,np.newaxis]/l[np.newaxis,:]
        x += (x==0)*1e-6 # avoid /0
        tmp = (Bl(l, Teff)*t*l)**2
        res = np.sum(tmp[None,:]*v2func(x), axis=1)/np.sum(tmp)
    else:
        x = c*B/l
        x += (x==0)*1e-6 # avoid /0
        tmp = (Bl(l, Teff)*t*l)**2
        res = np.sum(v2func(x)*tmp)/np.sum(tmp)
    return res

def vStarShell(B, params, plot=False, bias=False):
    """
    complex visibility of star+shell

    B: array of baselines, in meters
    params: {'wavel':in um,
             'Tstar':, 'Tshell': in K,
             'LD DIAM': in mas,
             'nu': LD coef for the star (1 is UD, default),
             'Rshell/Rstar':>1,
             'tau': optical depth of the layer}
    """
    r = np.linspace(0,1-1e-5, 1000)
    if 'h' in params.keys():
        r = np.linspace(0,4*params['h'], 1000)
    V2star, V2all, fr = 0, 0, 0
    wl0 = params['wavel']
    #mear = [0.6, 0.8, 1.0, 1.2, 1.4]
    smear = [0.99, 1.01]
    smear = [1]
    for _l in smear: # -- semaring
        params['wavel'] = _l*wl0
        I = iStarShell(r, params, plot=False)
        fr += np.trapz(I[1]*r, r)/np.trapz((I[0]+I[1])*r, r)/len(smear)
        c = np.pi*np.pi/(180*3600*1000.0)*1e6
        if 'h' in params.keys():
            x = c*B*params['LD DIAM']/params['wavel']
        else:
            x = c*B*params['LD DIAM']/params['wavel']*params['Rshell/Rstar']

        # -- Hankel transform, star only:
        Vstar = np.trapz(special.j0(x[np.newaxis,:]*r[:,np.newaxis])*
                     (I[0]*r)[:,np.newaxis], r[:,np.newaxis], axis=0)
        Vstar /= np.trapz(I[0]*r, r)
        V2star += Vstar**2/len(smear)

        # -- Hankel transform, star+shell:
        Vall = np.trapz(special.j0(x[np.newaxis,:]*r[:,np.newaxis])*
                     ((I[0]+I[1])*r)[:,np.newaxis], r[:,np.newaxis], axis=0)
        Vall /= np.trapz((I[0]+I[1])*r, r)
        V2all += Vall**2/len(smear)

    if plot:
        plt.figure(4)
        plt.clf()
        plt.subplot(211)
        plt.plot(B*params['LD DIAM'],V2all, label='star + %4.2f%% shell'%(fr*100))
        plt.plot(B*params['LD DIAM'], V2star, label='star only')
        plt.ylabel(r'V$^2$')
        plt.legend()
        plt.subplot(212)
        plt.plot(B*params['LD DIAM'], np.interp(V2all[::-1], V2star[::-1], B[::-1])[::-1]/B-1)
        plt.ylim(-0.05, 10*fr)
        plt.ylabel('angular diameter bias')
        plt.plot(B*params['LD DIAM'], 0*B, linestyle='dashed')
        plt.xlabel(r'B$\theta$ (m.mas)')
    else:
        if bias:
            # -- diam bias and flux ratio
            return (np.interp(V2all[::-1], V2star[::-1], B[::-1])[::-1]/B, fr)
        else:
            return Vall

def v2StarShell(B, params, smear=np.linspace(0.88, 1.12, 8)):
    if smear is None:
        return vStarShell(B, params, plot=False, bias=False)**2
    else:
        v2 = 0
        for s in smear:
            v2 += vStarShell(s*B, params, plot=False, bias=False)**2/len(smear)
        return v2

def frStarShell(params):
    r = np.linspace(0,1-1e-5, 1000)
    if 'h' in params.keys():
        r = np.linspace(0,4*params['h'], 1000)
    I = iStarShell(r, params)
    return np.trapz(I[1]*r, r)/np.trapz((I[0]+I[1])*r, r)

def iStarShell(r_rshell, params, plot=False):
    """
    Intesity profile for a star surrounded by a shell.
    params: {'wavel':in um,
             'Tstar':, 'Tshell': in K,
             'nu': LD coef for the star (1 is UD, default)
             'Rshell/Rstar':>1,
             'tau': optical depth of the layer}
    r_rshell is an array of values in [0..1]

    see Perrin et al. A&A 436-317 (2005), eq 1 and 2

    returns [Istar, Ishell]
    """
    h = 6.63e-34
    c = 3.e8
    k = 1.38e-23
    c1 = 2*h*c**2
    c2 = h*c/k
    # -- planck function
    Bl = lambda l,t: c1/(l*1e-6)**5*1./(np.exp(c2/(l*1e-6*t))-1)

    if not params.has_key('nu'):
        params['nu']=1.0

    # see Perrin et al. A&A 436-317 (2005), eq 1 and 2:
    if 'tau' in params.keys():
        Istar = [Bl(params['wavel'], params['Tstar'])*
                 np.sqrt(1-(r*params['Rshell/Rstar'])**2)**((params['nu']-1)/2.)*
                 np.exp(-params['tau']/np.sqrt(1-r**2)) if
                 r<1./params['Rshell/Rstar'] else
                 0.0 for r in r_rshell]
        Ishell=[ Bl(params['wavel'], params['Tshell'])*
                 (1-np.exp(-params['tau']/np.sqrt(1-r**2))) if
                 r<1./params['Rshell/Rstar'] else
                 Bl(params['wavel'], params['Tshell'])*
                 (1-np.exp(-2*params['tau']/np.sqrt(1-r**2))) for r in r_rshell]
    elif 'h' in params.keys():
        Istar = [Bl(params['wavel'], params['Tstar'])*
                 np.sqrt(1-r**2)**((params['nu']-1)/2.)
                 if r<1. else 0.0 for r in r_rshell]
        Ishell= Bl(params['wavel'], params['Tshell'])*\
            np.exp(-(r/params['h'])**2)

    Istar = np.array(Istar); Ishell = np.array(Ishell)

    if plot:
        plt.clf()
        plt.plot(r_rshell, Istar,  'r', label='star', linewidth=4)
        plt.plot(r_rshell, Ishell, 'b', label='shell', linewidth=4)
        plt.plot(r_rshell, Ishell+Istar, 'g', label='total',
                    linewidth=4, linestyle='dashed')

        plt.legend()
        plt.xlabel('$R/R_{shell}$ = $R/R_{\star}$/'+'%4.2f'%(params['Rshell/Rstar']))
        print 'flux ratio (shell/total): %4.2f %%'%(100*np.trapz(Ishell*r_rshell, r_rshell)/
              np.trapz((Istar+Ishell)*r_rshell, r_rshell))

    else:
        return [Istar, Ishell]

filename = 'BW2_diambiask.dpy'
try:
    print 'loading %sy ...'%filename,
    f = open(os.path.join(_dir_data,filename))
    __biasData = cPickle.load(f)
    f.close()
    print 'OK'
except:
    print 'Failed'
    # -- compute grid as it does not seem to exist
    print 'computing', filename
    # =================================
    params = {'LD DIAM':1.0, 'wavel':2.133, 'Tstar':6000.,
              'Tshell':1200., 'nu':1.0, 'Rshell/Rstar':2.,
              'tau':0.3}
    Ntau = 25
    Nrsrs = 10
    NB = 100
    B = np.linspace(1., 500., NB) # in meters
    __biasData = {'fr':np.zeros((Nrsrs,Ntau)),
                  'bias':np.zeros((Nrsrs,Ntau,NB)),
                  'Rshell/Rstar':np.linspace(1.1,6,Nrsrs),
                  'Bdw':B*params['LD DIAM']/params['wavel']}
    for j, rsrs in enumerate(__biasData['Rshell/Rstar']):
        print j, Nrsrs
        for i, tau in enumerate(np.logspace(-4, 0, Ntau)):
            params['tau'] = tau
            params['Rshell/Rstar'] = rsrs
            tmp = vStarShell(B, params, bias=True)
            __biasData['fr'][j,i] = tmp[1]
            __biasData['bias'][j,i,:] = tmp[0]
    print 'saving', filename
    f = open(os.path.join(_dir_data, filename), 'wb')
    cPickle.dump(__biasData, f, 2)
    f.close()

KLUDGE = 0.5
if KLUDGE!=1:
    print '\033[41m### bw2.diamBiasK has a kludge factor of %f ###\033[0m'%float(KLUDGE)

def diamBiasK(diam, B, Kexcess, RshellRstar=2.5):
    """
    diameter bias (>1) due to the presence of a shell

    only works for scalar diam, B and Kexcess

    validity: Kexcess>0 and Kexcess<0.1 and B*diam <~ 500

    return 1 if Kexcess <= 0
    """
    global __biasData, KLUDGE
    d = np.abs(__biasData['Rshell/Rstar']-RshellRstar)
    j0 = np.argsort(d)[0]
    j1 = np.argsort(d)[1]
    if Kexcess>0 and diam*B/2.2<__biasData['Bdw'][-1]:
        tmp = [np.interp(diam*B/2.2, __biasData['Bdw'],
                         __biasData['bias'][j0,k,:])
                for k in range(__biasData['bias'].shape[1]) ]
        r0 = np.interp(np.log10(KLUDGE*Kexcess), np.log10(__biasData['fr'][j0,:]), tmp)

        tmp = [np.interp(diam*B/2.2, __biasData['Bdw'],
                         __biasData['bias'][j1,k,:])
                for k in range(__biasData['bias'].shape[1]) ]
        r1 = np.interp(np.log10(KLUDGE*Kexcess), np.log10(__biasData['fr'][j0,:]), tmp)

        return r0 + (r1-r0)*(RshellRstar-__biasData['Rshell/Rstar'][j0])/\
                (__biasData['Rshell/Rstar'][j1]-__biasData['Rshell/Rstar'][j0])
    else:
        return 1

def diamBiasK_fig():
    diam = 1.0
    B = np.linspace(0,500,100)
    plt.close(0)
    plt.figure(0, figsize=(6,5))
    sty = ['-', '--', '-.', ':']
    for i, K in enumerate([0.0, 0.01, 0.02, 0.04]):
        plt.plot(np.pi*B*diam*np.pi/(180.*3600*1000.)/2.2e-6,
                 [diamBiasK(diam, b, K) for b in B], sty[i],
                 color='k', linewidth=2,
                 label = 'K$_\mathrm{ex}$ = %4.2fmag'%K)
    plt.legend()
    plt.ylim(0.98, 1.3)
    plt.xlim(0, 3.5)
    plt.xlabel(r'x = $\pi$B$\theta$/$\lambda$')
    plt.ylabel(r'$\theta_{observed}$ / $\theta_{real}$')
    return

# ============

def periodChange(mass=7., i=2, plot=False):
    """
    - mass: in solar mass
    - i: crossing of the instability strip (default=2)

    based on Fadeyev 2014:
    "Theoretical Rates of Pulsation Period Change in the Galactic Cepheids"
    https://arxiv.org/pdf/1401.6547.pdf
    """
    M = np.array([6., 8., 10., 12.])
    # -- Periods, in days, 2 edges of instability strip:
    # -- blue to red for crossing 1 and 3, red to blue for crossing 2
    P = {1: (np.array([1.89, 6.39, 12.56, 22.12]),
             np.array([2.652, 8.615, 19.229, 32.867])),
         2: (np.array([6.82, 20.06, 48.35, 92.98]),
             np.array([4.436, 9.229, 26.831, 51.125])),
         3: (np.array([6.52, 14.28, 31.49, 49.96]),
             np.array([9.864, 22.515, 46.645, 72.64]))}
    # -- change of period, in s/yr
    dP = {1: (np.array([3.3, 46.2, 382.6, 895.9]),
              np.array([6.04, 82.10, 385.32, 1167.15])),
          2: (-np.array([1.6, 55, 460.3, 2165.6]),
              -np.array([0.487, 17.0, 283.3, 1121.45])),
          3: (np.array([0.46, 3.9, 66.7, 113.2]),
              np.array([1.56, 23.64, 151.52, 539.8]))}

    if not plot:
        if i==2:
            s = -1.
        else:
            s = 1.
        return (s*10**np.interp(mass, M, np.log10(s*dP[i][0])),
                s*10**np.interp(mass, M, np.log10(s*dP[i][1])))

    if plot:
        plt.figure(22)
        plt.clf()
        colors = {6:'k', 8:'r', 10:'orange', 12:'y'}
        styles = {1:'dotted', 2:'dashed', 3:'solid'}
        for k,m in enumerate(M):
            for i in [1,2,3]:
                if i==2:
                    s = -1.
                    plt.subplot(212, sharex=ax1)
                    plt.xlabel('log10 P (days)')
                else:
                    s = 1.
                    ax1 = plt.subplot(211)
                plt.plot(np.log10([P[i][0][k], P[i][1][k]]),
                         np.log10([s*dP[i][0][k], s*dP[i][1][k]]),
                         '-o', color=colors[m],
                         linestyle=styles[i])
                plt.ylabel('log10 dP (s/yr)')
    return
