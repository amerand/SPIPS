"""
reads an interpolate files obtained here:

http://wwwuser.oats.inaf.it/castelli/grids.html

In each file: col.3=wavelength in nm
              col.4=frequency nu (s^-1)
              col.5=Hnu (erg/cm^2/s/Hz/ster)
              col.6=Hcont (erg/cm^2/s/Hz/ster)
              col.7=Hnu/Hcont

Flambda=4*Hnu*c/wavelength^2  c=light velocity

if c m/s and wavelength in m, then
Flambda in erg/cm2/s/ster/m
usual unit in erg/cm2/s/ster/A

obsolete: Reads also LD coef from PHOEBE

AUTHOR: Antoine MERAND (2011-2017)
"""
import os, sys

_dir_data = '../SPIPS/DATA/'
for d in sys.path:
    if os.path.isdir(d) and 'SPIPS' in d:
        _data_dir = os.path.join(d, 'DATA')
        print '\033[43m', _data_dir, '\033[0m'

if not os.path.exists(_dir_data):
    os.makedirs(_dir_data)

print 'ATLAS9: _dir_data=', _dir_data


import numpy as np
import photfilt2
from scipy.optimize import leastsq
import urllib2
import cPickle

try:
    from matplotlib import pyplot
except:
    print 'no plots'

_warnings=True

def ReadAllFiles_Imu(directory, TeffMax=12000):
    """
    returns a dictionnary of interpolation function. Interpolation is
    FLAMBDA (erg/s/cm2/m/ster) as a function of wavelength (um). Keys
    of the dictionnary are (teff, logg, metalicity).
    """
    global LD_teff, LD_logg, LD_MH, LD_wl, LD_data
    global _dir_data
    if not os.path.exists(directory):
        os.makedirs(directory)
    files = os.listdir(directory)
    res = {}
    print 'ALTLAS9: reading files in', directory
    # -- read all SED:
    mu_ = np.logspace(-2,0,200)[::-1]
    r_ = np.sqrt(1-mu_**2)

    files = filter(lambda x: x.endswith('odfnew.dat'), files)
    if len(files)==0:
        print 'could not find any ATLAS9 "odfnew.dat" models in '+directory
        print ' -> downloading models...'
        getAllFiles(directory.split('/')[-1])
        files = os.listdir(directory)
        files = filter(lambda x: x.endswith('odfnew.dat'), files)

    for f in files:
        _Teff = int(f.split('t')[1].split('g')[0])
        if _Teff <= TeffMax:
            tmp = ReadOneFile(os.path.join(directory,f))
            #res[(tmp['TEFF'], tmp['LOGG'], tmp['METAL'])] = \
            #    (tmp['WAVEL'], tmp['FLAMBDA'])
            if not LD_data.has_key((tmp['TEFF'],
                                    tmp['LOGG'],
                                    tmp['METAL'])):
                w = np.where(tmp['WAVEL']>0)
                tmp['FLAMBDA'] = tmp['FLAMBDA'][w]
                tmp['WAVEL'] = tmp['WAVEL'][w]
                key = (tmp['TEFF'], tmp['LOGG'], tmp['METAL'])
                res[key] = (r_, # radius 0->1
                            tmp['WAVEL'], # wl in um
                            [], # Imu(wl,r)
                            np.ones(len(tmp['WAVEL'])), # Ross, 1 here
                            tmp['FLAMBDA']) # total flux
            else:
                key = (tmp['TEFF'], tmp['LOGG'], tmp['METAL'])
                imu = []
                # -- keep wavelength were LD is defined
                # -- WARNING: cuts long wavelengths!
                #w = np.where((tmp['WAVEL']>=min(LD_wl)*0.8)*
                #             (tmp['WAVEL']<=max(LD_wl)*1.5))
                w = np.where(tmp['WAVEL']>0)
                tmp['FLAMBDA'] = tmp['FLAMBDA'][w]
                tmp['WAVEL'] = tmp['WAVEL'][w]
                k = 0
                I1 = Imu_I0c1c2c3c4(mu_, LD_data[key][LD_wl[k]])
                I2 = Imu_I0c1c2c3c4(mu_, LD_data[key][LD_wl[k+1]])
                for i,wl in enumerate(tmp['WAVEL']):
                    # -- follow the WL progression
                    if k!=len(LD_wl)-2 and wl>LD_wl[k+1]:
                        k+=1
                        I1 = Imu_I0c1c2c3c4(mu_, LD_data[key][LD_wl[k]])
                        I2 = Imu_I0c1c2c3c4(mu_, LD_data[key][LD_wl[k+1]])
                    imu_ = (I1 + (wl-LD_wl[k])*(I2-I1)/
                                    (LD_wl[k+1]-LD_wl[k]))
                    imu_ = np.maximum(0, imu_)
                    imu_ = np.minimum(1, imu_)
                    imu.append(imu_*tmp['FLAMBDA'][i])
                # -- same structure as for PHOENIX models
                res[key] = (r_, # radius 0->1
                            tmp['WAVEL'], # wl in um
                            np.array(imu), # Imu(wl,r)
                            np.ones(len(tmp['WAVEL'])), # Ross, 1 here
                            tmp['FLAMBDA']) # total flux
        else:
            print 'ignoring', f
    return res

def ReadAllFiles_BOSZ():
    import bosz
    files = os.listdir(bosz.dirData)
    files = filter(lambda x: x.endswith('.asc'), files)
    res = {}
    for f in files:
        m = bosz.readOneFile(os.path.join(bosz.dirData, f))
        key = (m['Teff'], m['logg'], m['metal'])
        res[key] = (None, m['wlA']*1e-4, None, None, 1e10*m['H_BOSZ'])
    return res


def Imu_FAST(wavel, mu, teff, logg, debug=False):
    """
    interpolate Imu(wavel) for array of 'wavel' but scalar mu, teff,
    logg. Requires global variables 'model', '_teff' and '_logg' to be set.

    limitation: there is a lot of check for NaNs and Infs that appear
    from some reason.
    """
    global models, _teff, _logg, _warnings

    if teff<_teff.min() or teff>_teff.max():
        if _warnings:
            print 'atlas.py: WARNING -> TEFF:', teff, 'not in [',\
                  _teff.min(), '-', _teff.max(), ']'
            _warnings=False
        #teff = np.clip(teff,_teff.min()+1,_teff.max()-1 )
    if logg<_logg.min()-0.25 or logg>_logg.max()+0.25:
        if _warnings:
            print 'atlas.py: WARNING -> LOGG:', logg, 'not in [',\
                  _logg.min(), '-', _logg.max(), ']'
            _warnings=False
        #logg = np.clip(teff,_logg.min()+.25,_logg.max()-.25 )

    # -- two closest tabulated Teff and logg
    it = np.abs(teff-_teff).argsort()[:2]
    ig = np.abs(logg-_logg).argsort()[:2]
    if debug:
        print 'it, ig', it, ig
        print teff, ':', _teff[it[0]], _teff[it[1]]
        print logg, ':', _logg[ig[0]], _logg[ig[1]]
    rmu = np.sqrt(1-mu**2)

    rossMean = 1. # assumes input 'mu' are wrt to ross.mean() already
    # -- interpolat in rmu, for the 4 (teff, logg)
    i0 = 0; i1=1 # range for interpolation

    # -- get closest models and the Rosseland radius for each wavelength, 4 quad
    # of the grid: normalize all R's to the Rosseland radius in continuum
    # spherical models have ross < 1

    k00 = (_teff[it[0]], _logg[ig[0]], 0)
    if models.has_key(k00):
        a00 = models[k00]
        r00 = sliding_avg_interp1d(a00[1], a00[3], wavel)
        r00Mean = np.median(r00[np.isfinite(r00)])
        ir00 = np.abs(a00[0]/r00Mean-rmu/rossMean).argsort()[:3]
        Imu00 = a00[2][:,ir00[i0]] +\
                (rmu/rossMean - a00[0][ir00[i0]]/r00Mean)*\
                (a00[2][:,ir00[i1]] - a00[2][:,ir00[i0]])/\
                (a00[0][ir00[i1]]/r00Mean- a00[0][ir00[i0]]/r00Mean)
        ### bin in wavelength:
        Imu00 = sliding_avg_interp1d(a00[1], Imu00, wavel)

    k01 = (_teff[it[0]], _logg[ig[1]], 0)
    if models.has_key(k01):
        a01 = models[k01]
        r01 = sliding_avg_interp1d(a01[1], a01[3], wavel)
        r01Mean = np.median(r01[np.isfinite(r01)])
        ir01 = np.abs(a01[0]/r01.mean()-rmu/rossMean).argsort()[:3]
        Imu01 = a01[2][:,ir01[i0]] +\
                (rmu/rossMean - a01[0][ir01[i0]]/r01Mean)*\
                (a01[2][:,ir01[i1]] - a01[2][:,ir01[i0]])/\
                   (a01[0][ir01[i1]]/r01Mean- a01[0][ir01[i0]]/r01Mean)
        Imu01 = sliding_avg_interp1d(a01[1], Imu01, wavel)


    k10 = (_teff[it[1]], _logg[ig[0]], 0)
    if models.has_key(k10):
        a10 = models[k10]
        r10 = sliding_avg_interp1d(a10[1], a10[3], wavel)
        r10Mean = np.median(r10[np.isfinite(r10)])
        ir10 = np.abs(a10[0]/r10Mean-rmu/rossMean).argsort()[:3]
        Imu10 = a10[2][:,ir10[i0]] +\
                (rmu/rossMean - a10[0][ir10[i0]]/r10Mean)*\
                (a10[2][:,ir10[i1]] - a10[2][:,ir10[i0]])/\
                   (a10[0][ir10[i1]]/r10Mean- a10[0][ir10[i0]]/r10Mean)
        Imu10 = sliding_avg_interp1d(a10[1], Imu10, wavel)


    k11 = (_teff[it[1]], _logg[ig[1]], 0)
    if models.has_key(k11):
        a11 = models[k11]
        r11 = sliding_avg_interp1d(a11[1], a11[3], wavel)
        r11Mean = np.median(r11[np.isfinite(r11)])
        ir11 = np.abs(a11[0]/r11Mean-rmu/rossMean).argsort()[:3]
        Imu11 = a11[2][:,ir11[i0]] +\
                (rmu/rossMean - a11[0][ir11[i0]]/r11Mean)*\
                (a11[2][:,ir11[i1]] - a11[2][:,ir11[i0]])/\
                   (a11[0][ir11[i1]]/r11Mean- a11[0][ir11[i0]]/r11Mean)
        Imu11 = sliding_avg_interp1d(a11[1], Imu11, wavel)

    try:
        ### first linear interp in Teff
        Imu0 = Imu00 + (Imu10-Imu00)*(teff-_teff[it[0]])/\
                (_teff[it[1]]-_teff[it[0]])
    except:
        Imu0 = None

    try:
        ### second linear interp in Teff
        Imu1 = Imu01 + (Imu11-Imu01)*(teff-_teff[it[0]])/\
                (_teff[it[1]]-_teff[it[0]])
    except:
        Imu1 = None

    if Imu0 is None and not Imu1 is None:
        return Imu1
    if Imu1 is None and not Imu0 is None:
        return Imu0
    if Imu0 is None and Imu1 is None:
        print 'OUPS!', teff, logg

    ### final interpolation (logg)
    Imu = Imu0 + (Imu1-Imu0)*(logg-_logg[ig[0]])/\
           (_logg[ig[1]]-_logg[ig[0]])

    if debug:
        pyplot.figure(0)
        pyplot.clf()
        pyplot.plot(wavel, Imu00, label='Imu00')
        pyplot.plot(wavel, Imu01, label='Imu01')
        pyplot.plot(wavel, Imu10, label='Imu10')
        pyplot.plot(wavel, Imu11, label='Imu11')
        pyplot.plot(wavel, Imu, linewidth=2, label='Imu')
        pyplot.legend()

    ### final interpolation
    return Imu

def multi_Imu_FAST(k, wavel, mu, teff, logg):
    """
    """
    return (k, Imu_FAST(wavel, mu, teff, logg))

def sliding_avg_interp1d(x,y,x0,dx0=None):
    """
    use to interpret y(x) for x0+-dx0/2.

    use if y is over-resolved compared to dx0 (same shape as x0)

    assumes x and x0 sorted!!!
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    if not isinstance(x0, np.ndarray):
        x0 = np.array(x0)
    if dx0==None:
        dx0 = np.gradient(x0)
    y0 = np.zeros(len(x0))

    for i in range(len(x0)):
        # -- slow :(
        #y0[i] = np.mean(y[(x>=x0[i]-dx0[i]/2)*(x<x0[i]+dx0[i]/2)])
        # -- faster
        y0[i] = np.mean(y[np.searchsorted(x, x0[i]-dx0[i]/2., side='left'):
                          np.searchsorted(x, x0[i]+dx0[i]/2., side='right')])

    if not all(np.isfinite(y0)):
        wf = np.where(np.isfinite(y0))
        wi = np.where(1-np.isfinite(y0))
        y0[wi] = np.interp(x0[wi], x0[wf], y0[wf])
    return y0

def sliding_avg_interp1d_WEAVE(x,y,x0,dx0=None):
    """
    use to interpret y(x) for x0+-dx0/2.

    use if y is over-resolved compared to dx0 (same shape as x0)

    assumes x and x0 sorted!!!
    """
    from scipy import weave
    from scipy.weave import converters

    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    if not isinstance(x0, np.ndarray):
        x0 = np.array(x0)
    if dx0==None:
        dx0 = np.gradient(x0)

    LX = len(x)
    LX0 = len(x0)
    y0 = np.zeros(len(x0))
    # using WEAVE
    code =\
    """
    double n;
    int j;
    int i;
    int j0;
    int inloop;
    j0=1;

    for (i=0; i<LX0; i++){
        n=0.0;
        j=j0;
        inloop=0;
        while ((x(j)<=x0(i)+dx0(i)/2) && (j<LX-1) ) {
            if ((x(j)>=x0(i)-dx0(i)/2.) && (x(j)<=x0(i)+dx0(i)/2.)) {
               n=n+(x(j+1)-x(j-1))/2.0;
               y0(i)=y0(i)+y(j)*(x(j+1)-x(j-1))/2.0;
               inloop=1;
            }
            if (inloop==0) {
                j0=j; // memorize where we started before
                }
            j++;
        }
        y0(i) /= n;
    }
    """
    err = weave.inline(code,
                       ['x', 'y', 'x0', 'dx0', 'LX', 'LX0','y0'],
                       type_converters=converters.blitz,
                       compiler = 'gcc')
    # try to catch 'nan' and 'inf'
    if not all(np.isfinite(y0)):
        wf = np.where(np.isfinite(y0))
        wi = np.where(1-np.isfinite(y0))
        y0[wi] = np.interp(x0[wi], x0[wf], y0[wf])
    return y0

def ReadOneFile(filename, plot=False):
    """
    returns a dictionnary with keys 'TEFF', 'LOGG', 'VTURB', 'LH' (all
    floats) and 'HNU', 'WAVEL', 'FLAMBDA' (np.array) units: TEFF in K,
    LOGG in log10(m/s2), VTURB in km/s, WAVEL in um, HNU in
    erg/s/cm/Hz/ster and FLAMBDA in erg/s/cm2/m/ster

    L/H is mixing length

    In each file: col.3=wavelength in nm
              col.4=frequency nu (s^-1)
              col.5=Hnu (erg/cm^2/s/Hz/ster)
              col.6=Hcont (erg/cm^2/s/Hz/ster)
              col.7=Hnu/Hcont
    Flambda=4*Hnu*c/wavelength^2  c=light velocity
    """
    f = open(filename, 'rU')
    lines = f.readlines()

    WAVEL = []
    HNU = []
    FLAMBDA = []

    for l in lines:
        if 'TEFF' in l:
            tmp = l.split()
            TEFF = float(tmp[1])
            LOGG = float(tmp[3])
        if 'TITLE' in l:
            tmp = l.split()
            METAL = float(l.split('[')[1].split(']')[0])
            VTURB = float(tmp[2].split('=')[1])
            LH = float(tmp[3].split('=')[1])
        if 'FLUX' in l and len(l.strip()) > 4:
            l = l[9:] # remove beginning of the line, not used
            tmp = l.split()
            wl = float(tmp[0])
            if wl >50.0:
                ### there is a typo for short WL in files
                ### too lazy to correct it
                WAVEL.append(wl/1000.) # in um
                HNU.append(float(tmp[2]))

    FLAMBDA = 4*np.array(HNU)*299792458.0/\
              (np.array(WAVEL)*1e-6)**2 #erg/s/cm2/m/ster
    f.close()

    if plot:
        pyplot.figure(1)
        pyplot.clf()
        pyplot.plot(WAVEL, FLAMBDA, 'r', linewidth=2)
        pyplot.yscale('log')
        pyplot.xscale('log')

    return {'TEFF':TEFF, 'LOGG': LOGG, 'VTURB':VTURB, 'LH':LH,\
            'HNU':np.array(HNU), 'WAVEL':np.array(WAVEL),\
            'FLAMBDA':FLAMBDA, 'METAL':METAL}

def readLD():
    """
    computes the LD profile I(mu) for Teff, Logg, and M/H. mu and wl are 1D
    ndarray.

    LD_data[(teff, logg, M/H, wl)] = (I0, c1, c2, c3, c4)
    all Teff, logg, M/H and wavelength values are tabulated in
    LD_teff, LD_logg, LD_MH, LD_wl.
    """
    global LD_teff, LD_logg, LD_MH, LD_wl, LD_data , _dir_data

    #print 'INFO:', __name__,': reading LD Data'
    LD_data = {}
    di = os.path.join(_dir_data, 'LD_PHOEBE')
    print 'ATLAS9: LD_PHOEBE/ ?', os.path.isdir(di)
    print 'ATLAS9: LD_PHOEBE/models.list ?', os.path.isfile(os.path.join(di, 'models.list'))
    # -- read model parameters:
    lines = open(os.path.join(di, 'models.list')).readlines()
    LD_teff = np.array([float(l.split()[0]) for l in lines])
    LD_logg = np.array([float(l.split()[1])/10. for l in lines])
    LD_MH   = np.array([float(l.split()[2])/10. for l in lines])
    LD_mu = np.logspace(-2,0,100)[::-1]

    # -- list of models:
    # -- http://phoebe.fiz.uni-lj.si/?q=node/110
    files = ['johnson_u.ld',
             'johnson_b.ld',
             'johnson_v.ld',
             'cousins_r.ld',
             'cousins_i.ld',
             '2mass_j.ld',
             '2mass_h.ld',
             '2mass_ks.ld']

    # -- effective wavelength of models
    LD_wl = [0.365, 0.445, 0.551, 0.71, 0.97, 1.2, 1.65, 2.17]
    for i, f in enumerate(files):
        lines = open(os.path.join(di, f)).readlines()
        lines = filter(lambda x: len(x)>1 and x[0]!='#', lines)
        for k,l in enumerate(lines):
            tmpC = [1,float(l.split()[7]),
                                        float(l.split()[8]),
                                        float(l.split()[9]),
                                        float(l.split()[10])]
            Imu = Imu_I0c1c2c3c4(LD_mu,tmpC)
            # -- normalize to integral over disk
            I0 = np.trapz(np.sqrt(1-LD_mu**2),np.sqrt(1-LD_mu**2))/\
                np.trapz(Imu*np.sqrt(1-LD_mu**2),np.sqrt(1-LD_mu**2))
            I0 = np.abs(I0)
            key = (LD_teff[k], LD_logg[k], LD_MH[k])
            ### not so sure about the following. I would assume I need to
            ### normalize:
            tmpC[0] = I0

            if LD_data.has_key(key):
                # -- add current wavelength to existing dic
                LD_data[key][LD_wl[i]]=tmpC
            else:
                # -- new dic
                LD_data[key]={LD_wl[i]:tmpC}
    return

def Imu_I0c1c2c3c4(mu, c):
    """
    I(mu, c) where c = (I0, c1, c2, c3, c4)
    """
    return c[0]*(1-c[1]*(1-mu**0.5)-c[2]*(1-mu)-c[3]*(1-mu**1.5)-c[4]*(1-mu**2.))

def filtMag(filter, teff, logg, diameter_mas):
    """
    returns the magnitude of a star of TEFF, LOGG and angular
    diameter DIAMTER_MASS (in mas).

    relies on 'photfilt2py'
    """
    # -- get range of the filter
    wr = photfilt2.wavelRange(filter)
    # -- wavelength range
    wl = np.linspace(wr[0], wr[1], 100)
    # -- SED
    sp = flambda(wl, teff, logg)
    # -- multiply by angular cone
    sp *= np.pi*(diameter_mas/2*np.pi/(180*3600*1000.0))**2
    # -- erg/cm2/s/m -> W/m2/um
    sp *= 10**-9
    # -- filter transmission
    T = photfilt2.Transmission(filter)(wl)

    # -- normalized total flux
    res = np.trapz(sp*T, wl)/np.trapz(T, wl)
    return photfilt2.convert_Wm2um_to_mag(res, filter)

def chi2func(p_fit, fit, p_fixed, obs, verbose=False):
    """
    p[0] = DIAM (mas)
    p[1] = TEFF (K)
    p[2] = logg
    obs = [('fitername', measurement, error), ()...]
    """
    try:
        p = np.zeros(fit.size)
        p[np.where(fit == 1)] = p_fit
        p[np.where(fit == 0)] = p_fixed
    except:
        p = p_fit
    res = []
    for o in obs:
        res.append((filtMag(o[0], p[1], p[2], p[0])-o[1])/o[2])
        if verbose:
            print o[0], '| OBS:', o[1], '+-', o[2], 'MODEL:',\
                  filtMag(o[0], p[1], p[2], p[0])
    if verbose:
        print 'CHI2=', (np.array(res)**2).sum()
        print 'CHI2_red=', (np.array(res)**2).sum()/\
              float(len(obs)-len(p_fit))
    return res

def fitPhotDiam(obs, fixedTeff=None, teffFirstGuess=10e3):
    """
    fit the photometric diameters to magnitudes. if fixedTeff is set,
    the Teff is set to the given value and not fitted

    obs = [('filtername', measurement, error), ()...]
    """
    p0 = np.array([1.0, teffFirstGuess, 4.0])
    fit = np.array([1,1,0])
    if not fixedTeff is None:
        p0[1] = fixedTeff
        fit[1] = 0

    p_fit = p0[ np.where(fit ==1)]
    p_fixed = p0[ np.where(fit == 0)]
    plsq, cov, info, mesg, ier =\
          leastsq(chi2func, p_fit, args=(fit, p_fixed, obs),
                  full_output=True)
    chi2_final = (np.array(chi2func(plsq, fit, p_fixed,
                                obs, verbose=True))**2).sum()
    cov *= chi2_final
    # params_names = ['DIAM (mas)', 'TEFF (K)', 'LOGG']
    for k in range(len(plsq)):
        print round(plsq[k], 5),'+-',\
              round(np.sqrt(np.diag(cov))[k], 5)

def flambda(wavel, teff, logg, metal=0.0, plot=False):
    """
    bi-linear interpolation in teff (K) and logg (log10 m/s2) as a
    function of wavelength (um).

    result in erg/s/cm2/m/ster
    """
    global models, _teff, _logg, _metal

    if not metal in _metal:
        # -- closest 2 in metalicity
        if metal>max(_metal):
            print 'ATLAS9 warning: metallicity too high'
        if metal<min(_metal):
            print 'ATLAS9 warning: metallicity too low'
        im = np.abs(metal-_metal).argsort()[:2]
        return flambda(wavel, teff, logg, metal=_metal[im[0]])+\
               (metal-_metal[im[0]])/(_metal[im[1]]-_metal[im[0]])*\
               (flambda(wavel, teff, logg, metal=_metal[im[1]]) -
                flambda(wavel, teff, logg, metal=_metal[im[0]]))

    # -- interpolate in Teff and Logg
    if teff>max(_teff):
        print 'ATLAS9 warning: Teff too high'
    if teff<min(_teff):
        print 'ATLAS9 warning: Teff too low'

    if logg>max(_logg):
        print 'ATLAS9 warning: logg too high'
    if logg<min(_logg):
        print 'ATLAS9 warning: logg too low'

    # -- 2 closest indexes in each direction
    it = np.abs(teff-_teff).argsort()[:2]
    ig = np.abs(logg-_logg).argsort()[:2]

    # -- linear interpolation
    #spectr = lambda x,wavel: np.interp(wavel, x[1], x[4])
    # -- log/log interpolation
    spectr = lambda x,wavel: 10**np.interp(np.log10(wavel),
                                           np.log10(x[1]), np.log10(x[4]))
    # -- optimized for Rayleigh-Jeans:
    #spectr = lambda x,wavel: np.interp(wavel, x[1], x[4]*x[1]**4)/wavel**4

    # -- check that the 4 models exist
    for t in _teff[it]:
        for g in _logg[ig]:
            if not models.has_key((t,g,metal)):
                #print '!no model for: Teff=', t, 'logg=', g
                # get logg of models for this Teff
                mg = []
                for k in models.keys():
                    if k[0]==t and k[2]==metal:
                        mg.append(k[1])
                mg = np.array(mg)
                mg = mg[np.abs(g-mg).argsort()]
                # -- interpolate
                tmp_ = spectr(models[(t,mg[0],metal)], wavel) + \
                       (spectr(models[(t,mg[1],metal)], wavel)-
                        spectr(models[(t,mg[0],metal)], wavel))*\
                       (g-mg[0])/(mg[1]-mg[0])
                if t== _teff[it[0]] and g==_logg[ig[0]]:
                    tmp00=tmp_
                elif t== _teff[it[1]] and g==_logg[ig[0]]:
                    tmp10=tmp_
                elif t== _teff[it[0]] and g==_logg[ig[1]]:
                    tmp01=tmp_
                elif t== _teff[it[1]] and g==_logg[ig[1]]:
                    tmp11=tmp_
            else:
                if t== _teff[it[0]] and g==_logg[ig[0]]:
                    tmp00 = spectr(models[(_teff[it[0]], _logg[ig[0]], metal)], wavel)
                elif t== _teff[it[1]] and g==_logg[ig[0]]:
                    tmp10 = spectr(models[(_teff[it[1]], _logg[ig[0]], metal)], wavel)
                elif t== _teff[it[0]] and g==_logg[ig[1]]:
                    tmp01 = spectr(models[(_teff[it[0]], _logg[ig[1]], metal)], wavel)
                elif t== _teff[it[1]] and g==_logg[ig[1]]:
                    tmp11 = spectr(models[(_teff[it[1]], _logg[ig[1]], metal)], wavel)
    ### 4 corners in the grid

    #tmp00 = spectr(models[(_teff[it[0]], _logg[ig[0]], metal)], wavel)
    #tmp10 = spectr(models[(_teff[it[1]], _logg[ig[0]], metal)], wavel)
    #tmp01 = spectr(models[(_teff[it[0]], _logg[ig[1]], metal)], wavel)
    #tmp11 = spectr(models[(_teff[it[1]], _logg[ig[1]], metal)], wavel)

    ### interpolation in Teff
    tmp0 = tmp00 + (tmp10-tmp00)*(teff-_teff[it[0]])/(_teff[it[1]]-_teff[it[0]])
    tmp1 = tmp01 + (tmp11-tmp01)*(teff-_teff[it[0]])/(_teff[it[1]]-_teff[it[0]])
    ### interpolation in logg
    res = tmp0 + (tmp1-tmp0)*(logg-_logg[ig[0]])/(_logg[ig[1]]-_logg[ig[0]])

    if plot:
        print 'Teff=', teff, _teff[it]
        print 'logg=', logg, _logg[ig]
        pyplot.figure(1)
        pyplot.clf()
        pyplot.plot(wavel, tmp00, 'r')
        pyplot.plot(wavel, tmp10, 'r')
        pyplot.plot(wavel, tmp0,  'r', linewidth=2)
        pyplot.plot(wavel, tmp01, 'b')
        pyplot.plot(wavel, tmp11, 'b')
        pyplot.plot(wavel, tmp1,  'b', linewidth=2)
        pyplot.plot(wavel, res,   'k')
        pyplot.xscale('log')
        pyplot.xlim(wavel.min(), wavel.max())
    else:
        return res

def getAllFiles(directory='gridp00k2odfnew', table=None, TeffMax=12000, TeffMin=4000):
    global _dir_data
    url='http://wwwuser.oats.inaf.it/castelli/grids/'+directory+'/'
    if table is None:
        table = directory.split('odfnew')[0].split('grid')[-1]
        table= 'f'+table+'tab.html'

    print url+table
    lines = urllib2.urlopen(url+table).read().split('\n')
    lines = filter(lambda x: '.dat' in x, lines)
    lines = [l.split('"')[1] for l in lines]
    directory = url.split('/')[-1]
    if directory=='':
        directory = url.split('/')[-2]
    print 'directory:', directory
    directory = os.path.join(_dir_data+'ATLAS9', directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    for l in lines:
        teff = int(l.split('t')[1].split('g')[0])
        if teff<=TeffMax and teff>=TeffMin:
            print directory+'/'+l
            tmp = urllib2.urlopen(url+'/'+l, timeout=60).read()
            f = open(os.path.join(directory,l), 'w')
            f.write(tmp)
            f.close()
    try:
        os.remove('../SPIPS/DATA/BW2_maggrid_atlas9.dpy')
    except:
        pass
    try:
        os.remove('../SPIPS/DATA/BW2_ATLAS9_aLambdaSPE.dpy')
    except:
        pass
    return

atlas9__dir = [#'gridm25k2odfnew', # F/H = -2.5
               #'gridm20k2odfnew', # F/H = -2.0
               #'gridm15k2odfnew', # F/H = -1.5
               'gridm10k2odfnew', # F/H = -1.0
               'gridm05k2odfnew', # F/H = -0.5
               'gridp00k2odfnew', # <- SOLAR METALLICITY
               #'gridp02k2odfnew', # F/H =  0.2
               'gridp05k2odfnew' # F/H =  0.5
    ]

atlas9__dir = [os.path.join(_dir_data+'/ATLAS9',d) for d in atlas9__dir]

def test(key=None):
    global models, _teff, _logg
    if key is None:
        key = (8000, 4.0, 0.0)

    r   = models[k][0]
    wl  = models[k][1]
    Imu = models[k][2]
    wl_, r_ = np.meshgrid(r,wl)

    pyplot.close(0)
    pyplot.figure(0)
    pyplot.subplot(311)
    pyplot.pcolormesh(r_, wl_, Imu/Imu[:,0][:,np.newaxis],vmin=0)
    pyplot.colorbar()
    pyplot.subplot(312)
    pyplot.plot(wl, Imu[:,0], 'r-', label='Imu(r=0)')
    pyplot.plot(wl, Imu[:,len(r)/2], 'r--', label='Imu(r=1/2)')
    pyplot.plot(wl, models[k][4], 'k-', label='total flux')
    pyplot.legend()
    pyplot.subplot(313)
    pyplot.plot(wl, Imu[:,len(r)/2]/Imu[:,0], 'r-',
                label='Imu(r=1/2)/Imu(r=0)')
    pyplot.legend(loc='lower right')
    return

useBOSZ = False # True for test
if useBOSZ:
    import bosz
    gridfile = 'bosz_models.dpy'
else:
    gridfile = 'atlas9_models.dpy'

try:
    tmp = len(models.keys())
    print tmp, 'ATLAS9 models already loaded'
except:
    # -- reading ATLAS9 models
    try:
        # -- read binary file
        print 'ATLAS9 reading file', gridfile

        if not useBOSZ:
            f = open(os.path.join(bosz.dirData,gridfile))
        else:
            f = open(os.path.join(_dir_data+'/ATLAS9/',gridfile))

        models = cPickle.load(f)
        f.close()
        print '   -> DONE'
    except:
        print '   -> FAILED'
        # -- generate binary file
        # -- read LD files
        #readLD() # not needed anymore
        LD_data = {}
        models = {}
        if not useBOSZ:
            # -- read SEDs and add LD
            for d in atlas9__dir:
                tmp = ReadAllFiles_Imu(d, TeffMax=12000)
                for t in tmp.keys():
                    models[t] = tmp[t]
            f = open(os.path.join(_dir_data,'ATLAS9', gridfile), 'wb')
        else:
            models = ReadAllFiles_BOSZ()
            f = open(os.path.join(bosz.dirData, gridfile), 'wb')
        cPickle.dump(models, f, 2)
        f.close()
        LD_data = [] # free memory
    # -- create lists of all teff, logg and metalicities
    _teff = []
    _logg = []
    _metal = []
    for k in models.keys():
        _teff.append(k[0])
        _logg.append(k[1])
        _metal.append(k[2])
    _teff = np.array(list(set(_teff)))
    _logg = np.array(list(set(_logg)))
    _metal = np.array(list(set(_metal)))
