import numpy as np
#from numpy import polynomials
from matplotlib import pyplot as plt
import dpfit
from scipy import special
import os
import cPickle
import astropy.io.fits as pyfits
import scipy.stats
import itertools
import sys

"""
ldsatlas._udld('../SPIPS/DATA/LD_NEILSON/GIANTS/spheric/ld_satlas_surface.2t5700g100m100.dat', plot=1)
ldsatlas._udld_B74(0.5)
"""

_data_dir = '../SPIPS/DATA/'

for d in sys.path:
    if os.path.isdir(d) and 'SPIPS' in d:
        _data_dir = os.path.join(d, 'DATA')
        print '\033[43m', _data_dir, '\033[0m'


col5 = ['Teff', 'logg', 'mass']
for b in ['B','V','R','I','H','K','CoR','Kep']:
    for i in ['1','2','3','4']:
        col5.append('f%s(%s)'%(i,b))
nei5 = {c:[] for c in col5}
col16 = ['Teff', 'logg', 'mass', 'B','V','R','I','H','K','CoR','Kep']
nei16 = {c:[] for c in col16}

TeffMin, TeffMax = 4000, 20000
ftpServer = 'cdsarc.u-strasbg.fr'

def downloadAllModels():
    global _data_dir

    _files_dir = 'LD_NEILSON/DWARFS/'
    # ================ DWARFS ==================================
    # -- download data is not present:
    #if not os.path.exists(os.path.join(_data_dir, _files_dir)):
    from ftplib import FTP
    try:
        print ' > making', os.path.join(_data_dir, _files_dir)
        os.makedirs(os.path.join(_data_dir, _files_dir))
    except:
        pass
    print ' > connecting to Vizier FTP...'
    ftp = FTP(ftpServer)     # connect to host, default port
    ftp.login()                     # user anonymous, passwd anonymous@
    directories = ['J', 'A+A', '556', 'A86']
    for d in directories:
        ftp.cwd(d)
    print 'http://cdsarc.u-strasbg.fr/viz-bin/qcat?'.join(directories)
    for f in ['ReadMe', 'table5.dat', 'table16.dat']:
        if not os.path.exists(os.path.join(_data_dir, _files_dir, f)):
            print 'downloading', os.path.join(_data_dir, _files_dir, f)
            ftp.retrbinary('RETR '+f,
                           open(os.path.join(_data_dir, _files_dir, f), 'wb').write)
    print ' > loading spherical models...'
    if not os.path.exists(os.path.join(_data_dir, _files_dir, 'spheric')):
        os.makedirs(os.path.join(_data_dir, _files_dir, 'spheric'))
    parent = ftp.pwd()
    ftp.cwd('{}/{}'.format(parent, 'spheric'))
    files = ftp.nlst()
    for k,f in enumerate(files):
        Teff = int(f.split('surface.2t')[1].split('g')[0])
        logg = int(f.split('surface.2t')[1].split('g')[1].split('m')[0])
        mass = int(f.split('surface.2t')[1].split('g')[1].split('m')[1].split('.')[0])
        if Teff>=TeffMin and Teff<=TeffMax and Teff%300==0:
        #if True:
            if not os.path.exists(os.path.join(_data_dir, _files_dir, 'spheric', f)):
                print 'downloading %3d/%3d'%(k+1, len(files)),
                print f
                ftp.retrbinary('RETR '+f,
                           open(os.path.join(_data_dir, _files_dir, 'spheric', f), 'wb').write)
    print ' > loading planar models...'
    if not os.path.exists(os.path.join(_data_dir, _files_dir, 'planar')):
        os.makedirs(os.path.join(_data_dir, _files_dir, 'planar'))
    ftp.cwd('{}/{}'.format(parent, 'planar'))
    files = ftp.nlst()
    for k,f in enumerate(files):
        Teff = int(f.split('ld_t')[1].split('g')[0])
        logg = int(f.split('ld_t')[1].split('g')[1].split('_')[0])
        if Teff>=TeffMin and Teff<=TeffMax and Teff%300==0 and logg>=75 and logg<=500:
            if not os.path.exists(os.path.join(_data_dir, _files_dir, 'planar', f)):
                print 'downloading %3d/%3d'%(k+1, len(files)),
                print f
                ftp.retrbinary('RETR '+f,
                           open(os.path.join(_data_dir, _files_dir, 'planar', f), 'wb').write)
    ftp.close()

    # ================ GIANTS ==================================
    _files_dir = 'LD_NEILSON/GIANTS/'
    #if not os.path.exists(os.path.join(_data_dir, _files_dir)):
    from ftplib import FTP
    try:
        print ' > making', os.path.join(_data_dir, _files_dir)
        os.makedirs(os.path.join(_data_dir, _files_dir))
    except:
        pass
    print ' > connecting to Vizier FTP...'
    ftp = FTP('cdsarc.u-strasbg.fr')     # connect to host, default port
    ftp.login()                     # user anonymous, passwd anonymous@
    for d in ['J', 'A+A', '554', 'A98']:
        ftp.cwd(d)
    for f in ['ReadMe', 'table5.dat', 'table16.dat']:
        if not os.path.exists(os.path.join(_data_dir, _files_dir, f)):
            print 'downloading', os.path.join(_data_dir, _files_dir, f)
            ftp.retrbinary('RETR '+f,
                           open(os.path.join(_data_dir, _files_dir, f), 'wb').write)
    print ' > loading spherical models...'
    if not os.path.exists(os.path.join(_data_dir, _files_dir, 'spheric')):
        os.makedirs(os.path.join(_data_dir, _files_dir, 'spheric'))
    parent = ftp.pwd()
    ftp.cwd('{}/{}'.format(parent, 'spheric'))
    files = ftp.nlst()
    for k,f in enumerate(files):
        Teff = int(f.split('surface.2t')[1].split('g')[0])
        logg = int(f.split('surface.2t')[1].split('g')[1].split('m')[0])
        mass = int(f.split('surface.2t')[1].split('g')[1].split('m')[1].split('.')[0])
        if Teff>=TeffMin and Teff<=TeffMax and Teff%300==0 and logg>=75 and logg<=500 and mass >=50 and mass%10==0:
        #if True:
            if not os.path.exists(os.path.join(_data_dir, _files_dir, 'spheric', f)):
                print 'downloading %3d/%3d'%(k+1, len(files)),
                print f
                ftp.retrbinary('RETR '+f,
                           open(os.path.join(_data_dir, _files_dir, 'spheric', f), 'wb').write)
    print ' > loading planar models...'
    if not os.path.exists(os.path.join(_data_dir, _files_dir, 'planar')):
        os.makedirs(os.path.join(_data_dir, _files_dir, 'planar'))
    ftp.cwd('{}/{}'.format(parent, 'planar'))
    files = ftp.nlst()
    for k,f in enumerate(files):
        Teff = int(f.split('ld_t')[1].split('g')[0])
        logg = int(f.split('ld_t')[1].split('g')[1].split('_')[0])
        if Teff>=TeffMin and Teff<=TeffMax and Teff%300==0 and logg>=75 and logg<=500:
            if not os.path.exists(os.path.join(_data_dir, _files_dir, 'planar', f)):
                print 'downloading %3d/%3d'%(k+1, len(files)),
                print f
                ftp.retrbinary('RETR '+f,
                           open(os.path.join(_data_dir, _files_dir, 'planar', f), 'wb').write)
    ftp.close()

#downloadAllModels()

_files_dir = './LD_NEILSON/DWARFS/'
# -- read table5.dat
# -- read the 4 parameters for each band
for l in open(os.path.join(_data_dir, _files_dir, 'table5.dat')).readlines():
    for k,w in enumerate(l.split()):
        nei5[col5[k]].append(float(w))

_files_dir = './LD_NEILSON/GIANTS/'
# -- read table16.dat
try:
    for l in open(os.path.join(_data_dir, _files_dir, 'table16.dat')).readlines():
        for k,w in enumerate(l.split()):
            nei16[col16[k]].append(float(w))
    for c in col16:
        nei16[c] = np.array(nei16[c])
except:
    pass

# -- continue reading the 4 parameters for each band
for l in open(os.path.join(_data_dir, _files_dir, 'table5.dat')).readlines():
    for k,w in enumerate(l.split()):
        nei5[col5[k]].append(float(w))
for c in col5:
    nei5[c] = np.array(nei5[c])

# -- Load the LD/Ross sent by Hilding, where LD is the outer diameter
cols = ['mass', 'L', 'Teff', 'logg', 'Ross', 'Outer', 'Ross/Outer']
rossTable = {k:[] for k in cols}
f = open(os.path.join(_data_dir, 'LD_NEILSON', 'Rosseland_SATLAS.dat'))
for l in f.readlines():
    if not l.strip().startswith('#'):
        for i,c in enumerate(cols):
            rossTable[c].append(float(l.split()[i]))
f.close()
f = open(os.path.join(_data_dir, 'LD_NEILSON', 'Rosseland_SATLAS_dwarfs.dat'))
cols = ['mass', 'L', 'Teff', 'logg', 'Ross/Outer']
for l in f.readlines():
    if not l.strip().startswith('#'):
        for i,c in enumerate(cols):
            rossTable[c].append(float(l.split()[i]))
f.close()

for c in cols:
    rossTable[c] = np.array(rossTable[c])

# ===============================================================================

# -- Claret 4-coef, for comparison ---------------------------------------
C4 = {}
if os.path.exists('../SPIPS/ATMO/tableeq5.dat'):
    print 'reading Claret+ 2011 4-coef table'
    f = open(os.path.join(_data_dir, 'ATMO/tableeq5.dat'))
    for l in f.readlines():
        # -- Teff, logg, Fe/H, Xi
        key = (float(l.split()[1]), float(l.split()[0]),
               float(l.split()[2]), float(l.split()[3]))
        coef = [float(l.split()[i]) for i in [4,5,6,7]]
        band = l.split()[8]
        model = l.split()[10]
        if C4.has_key(band+'_'+model):
            C4[band+'_'+model][key] = coef
        else:
            C4[band+'_'+model] = {key:coef}
    f.close()
# --------------------------------------------------------

def UDLD(o, diam, Teff):
    global _data
    try:
        _data.keys()
    except:
        _data={}
        datafile = 'BW2_satlas_models.dpy'
        if os.path.exists(os.path.join(_data_dir, datafile)):
            f = open(os.path.join(_data_dir, datafile))
            _data = cPickle.load(f)
            f.close()
        else:
            print ' > loading SATLAS models for logg=1.5 and mass=15Msol'
            for filename in os.listdir(os.path.join(_data_dir, _files_dir, 'spheric')):
                if filename.endswith('.dat'):
                    Teff = int(filename.split('surface.2t')[1].split('g')[0])
                    logg = float(filename.split('surface.2t')[1].split('g')[1].split('m')[0])/100.
                    mass = float(filename.split('surface.2t')[1].split('g')[1].split('m')[1].split('.')[0])/10.
                    if Teff%300==0 and logg==1.5 and mass==15.:
                        print filename
                        _data[Teff] = _udld(os.path.join(_data_dir, _files_dir, 'spheric', filename), plot=False)
            f = open(os.path.join(_data_dir, datafile), 'wb')
            cPickle.dump(_data, f, 2)
            f.close()

    if isinstance(o[2], list) or isinstance(o[2], tuple):
        wl = o[2][0]
        B = o[2][1]
    else:
        wl = o[2]
        B = 100.
        print 'warning, assumes B=100m for the UD/LD correction!'
    # --
    key0, key1 = '', None
    if 'fluor' in o[1].lower():
        key0 = 'K'
    if 'pti' in o[1].lower():
        key0 = 'H'
    if 'pionier' in o[1].lower():
        if 'free' in o[1].lower():
            key0 = 'H' # -- assumed H, low spectral resolution
    if 'amber' in o[1].lower():
        key0 = 'K' # assumes R=50, LR
    if 'gravity' in o[1].lower():
        key0 = 'K'

    if key0=='': # -- could not find the instrument, use wavelength
        bands = {0.45:'B', 0.55:'V', 0.65:'R', 1.0:'I', 1.65:'H', 2.2:'K'}
        # -- closest band
        wl0 = bands.keys()[np.argmin(np.abs(np.array(bands.keys())-wl))]
        key0 = bands[wl0]
        wl1 = bands.keys()[np.argsort(np.abs(np.array(bands.keys())-wl))[1]]
        key1 = bands[wl1]

    # -- interpolates in Teff:
    t0 = _data.keys()[np.abs(np.array(_data.keys())-Teff).argmin()]
    t1 = _data.keys()[np.abs(np.array(_data.keys())-Teff).argsort()[1]]
    res0 = UDLD_x(np.pi*B*diam/wl*(np.pi/(180*3600)*1e-6), _data[t0][key0])[0]
    res1 = UDLD_x(np.pi*B*diam/wl*(np.pi/(180*3600)*1e-6), _data[t1][key0])[0]
    res = res0 + (Teff-t0)*(res1-res0)/(t1-t0)
    if not key1 is None:
        res0 = UDLD_x(np.pi*B*diam/wl*(np.pi/(180*3600)*1e-6), _data[t0][key1])[0]
        res1 = UDLD_x(np.pi*B*diam/wl*(np.pi/(180*3600)*1e-6), _data[t1][key1])[0]
        res1 = res0 + (Teff-t0)*(res1-res0)/(t1-t0)
        res += (wl - wl0)*(res1-res)/(wl1 - wl0)

    return res

wl_b = {'B':0.45, 'V':0.55, 'R':0.65, 'I':1.0, 'H':1.65, 'K':2.2, 'CoR':0.1, 'Kep':0.1}
band = {wl_b[c]:c for c in wl_b.keys()}

def referee():
    """
    explanation for referee that scaling != clipping
    """
    n = 200
    r = np.linspace(0, 1,n)
    a, r0, rr = 0.1, 0.8, 0.9
    Ir = 1-a*r
    Ir[r>r0] = (1-a*r0)*np.exp(-100*(r[r>r0]-r0)**2)
    rp = np.linspace(0, 1/rr, n)

    plt.figure(0)
    plt.clf()
    plt.subplot(211)
    plt.ylabel('I(r)'); plt.xlabel('r')
    plt.plot(r, Ir, '-k', label='original')
    plt.plot(rp, Ir, '-b', label='scaled')
    plt.plot(r, Ir*(rp<1), '-r', label='clipped', linestyle='dashed')
    plt.plot(rp, Ir*(rp<1), '-r', label='clipped & scalled')

    plt.legend(loc='lower left')
    plt.plot([1,1], [0,1], color='0.5', linestyle='dotted')

    x = np.linspace(0, 8, n)
    V = np.trapz(special.jv(0,x[None,:]*r[:,None])*
                 (Ir[:,None]*r[:,None]), r[:,None], axis=0)**2
    V /= V[0]

    Vc = np.trapz(special.jv(0,x[None,:]*r[:,None])*
                 (Ir[:,None]*r[:,None]*(rp[:,None]<=1.0)), r[:,None], axis=0)**2
    Vc /= Vc[0]

    Vcs = np.trapz(special.jv(0,x[None,:]*rp[:,None])*
                 (Ir[:,None]*rp[:,None]*(rp[:,None]<=1.0)), rp[:,None], axis=0)**2
    Vcs /= Vcs[0]

    Vs = np.trapz(special.jv(0,x[None,:]*rp[:,None])*
                 (Ir[:,None]*rp[:,None]), rp[:,None], axis=0)**2
    Vs /= Vs[0]

    plt.subplot(212)
    plt.ylabel('Visibility'); plt.xlabel(r'X = $\pi$B$\theta$/$\lambda$')
    plt.plot(x, np.abs(V), '-k', label='original')
    plt.plot(x, np.abs(Vs), '-b', label='scaled')
    plt.plot(x/rr, np.abs(Vs), '-b', linestyle='dashed', linewidth=3,
             label='V and x scaled')
    plt.plot(x, np.abs(Vc), '-r', label='clipped', linestyle='dashed')
    plt.plot(x, np.abs(Vcs), '-r', label='clipped & scaled')
    plt.ylim(0.0,0.1)
    plt.legend(loc='lower left')
    return

def _rossWL(wl, param):
    return param['r0'] + param['r1']/wl**param['p']

def _udld(filename, plot=False, planar=False, showNeilson=False, reslim=0.05):
    global nei16, nei5, wl_b, rossTable
    # -- read file:
    f = open(filename)
    cols = ['mu','B','V','R','I','H','K','CoR','Kep']
    cols = ['mu','B','V','R','I','H','K']
    wl_b = {'B':0.45, 'V':0.55, 'R':0.65, 'I':1.0, 'H':1.65, 'K':2.2, 'CoR':0.1, 'Kep':0.1}

    data = {c:[] for c in cols}
    for l in f.readlines():
        for k,c in enumerate(cols):
            data[c].append(float(l.split()[k]))
    for c in cols:
        data[c] = np.array(data[c])
    f.close()
    colors = {'B':'m', 'V':'b', 'R':'g', 'I':(0.8,0.7,0.0), 'H':'orange', 'K':'red', 'CoR':'0.5', 'Kep':'0.5'}

    if planar:
        nei=None
        rossFromTable = 1.0
    else:
        Teff = int(filename.split('surface.2t')[1].split('g')[0])
        logg = float(filename.split('surface.2t')[1].split('g')[1].split('m')[0])/100.
        mass = float(filename.split('surface.2t')[1].split('g')[1].split('m')[1].split('.')[0])/10.
        try:
            i = np.argmin(((nei16['Teff']-Teff)/1000.)**2 +
                              (nei16['logg']-logg)**2+
                              (nei16['mass']-mass)**2)
            nei = {c:nei16[c][i] for c in cols[1:]}
        except:
            pass
        # -- find Rosseland diam from files sent by Hilding:
        i = np.argmin(((rossTable['Teff']-Teff)/1000.)**2 +
                          (rossTable['logg']-logg)**2+
                          (rossTable['mass']-mass)**2)
        rossFromTable = rossTable['Ross/Outer'][i]
        if plot:
            print "Rosseland from:"
            print "Teff=%4.0fK (instead of %4.0f)"%( rossTable['Teff'][i], Teff)
            print "logg=%4.2f (instead of %4.2f)"%( rossTable['logg'][i], logg)
            print "mass=%4.2f (instead of %4.2f)"%( rossTable['mass'][i], mass)

    # keep only bands
    cols = cols[1:]

    if plot:
        # -- setting up plots:
        if not isinstance(plot, int):
            plot=1

        # -- main plot:
        plt.close(plot)
        plt.figure(plot, figsize=(8,8))
        plt.subplots_adjust(left=0.1, bottom=0.06,
                            top=0.92, right=0.95,
                            wspace=0.26)
        plt.suptitle(os.path.basename(filename), size=20)
        # ax1 = plt.subplot(221)
        # ax2 = plt.subplot(222)
        # ax3 = plt.subplot(224, sharex=ax2)
        # ax4 = plt.subplot(223)

        ax1 = plt.subplot(221)
        ax2 = plt.subplot(422); ax3 = plt.subplot(424, sharex=ax2)
        ax2 =None; ax3 = plt.subplot(222)
        ax4 = plt.subplot(413)
        ax4r = plt.subplot(414, sharex=ax4)


        ax1.set_ylim(0,0.85)
        ax1.set_xlim(0.985, 1.02)
        ax1.set_xticks([0.99, 1.0, 1.01])
        ax1.set_xlabel('$r/r_\mathrm{Ross}$')
        ax1.set_ylabel('intensity: I / I(max)')

        if not ax2 is None:
            ax2.set_ylabel(r'V$^2$')
            ax2.set_ylim(0.0, 1.0)

        ax3.set_ylabel(r'UD / Ross. (single B)')
        ax3.set_xlabel(r'$\pi$B$\theta$/$\lambda$')
        ax3.set_ylim(0.88, 0.99)

        ax4r.set_xlabel(r'$\pi$B$\theta$/$\lambda$')
        ax4.set_ylabel(r'V$^2$')
        ax4r.grid()
        ax4r.set_ylim(-1.5, 1.5)
        ax4r.set_ylabel(r'$\Delta V^2$ (1e-4)')

        # -- LD analytical check
        if False:
            plt.close(1+plot)
            plt.figure(1+plot)
            plt.suptitle(os.path.basename(filename), size=20)
            plt.subplots_adjust(left=0.1, bottom=0.10,
                                top=0.90, right=0.95,
                                wspace=0.26)
            ax0 = plt.subplot(121)
            ax0.set_ylabel('Intensity: I / I(max)')
            ax0.set_ylim(reslim/5, 1.0)
            ax0.semilogy()
            axp = plt.subplot(122)
            axp.set_xlabel(r' $\delta$ = (model - analytic) / analytic (%)')
            axp.set_ylim(reslim/5,1)
            if showNeilson:
                axp.set_xlim(-100, 50)
            else:
                axp.set_xlim(-20, 40)
        else:
            ax0, axp= None, None
        if 'Kp' in cols:
            fig3 = [plt.axes([0.15, 0.4, 0.8, 0.55])]
            fig3.append(plt.axes([0.15, 0.1, 0.8, 0.25], sharex=fig3[-1]))

    res = {}
    r = np.sqrt(1-data['mu']**2)
    if not planar:
        ross = {}
        for j,b in enumerate(cols):
            ross[b] = maxVar(data[b], r)
        x, y = np.array([wl_b[k] for k in cols]), np.array([ross[k] for k in cols])

        w = np.linspace(0.3, 3, 100)
        _p = 0.6 # see loop below, this leads to the best value for logg = 1.0
        aL = np.polyfit(1/x**_p, y, 1)

        if False:
            plt.figure(20)
            plt.clf()
            chi2 = []
            ps = np.linspace(0.4, 0.8, 20)
            for p in ps:
                X, Y = 1/x**p, y; X = np.append(X, 0.); Y = np.append(Y, rossFromTable)
                aL = np.polyfit(X, Y, 1)
                chi2.append(np.mean((Y-np.polyval(aL,X))**2))
            plt.subplot(122)
            plt.plot(ps, chi2, 'ok')
            plt.subplot(121)
            plt.plot(w, np.polyval(aL, 1/w**p))
            plt.plot(x, y, 'ok')

        ross = np.mean([ross[k] for k in ross.keys()])
        # print 'rossFromTable  :', rossFromTable
        # print '-'*12
        # print ' <ross max var>:', ross
        # print '  -> ratio     :', ross/rossFromTable
        # print ' ross_infty    :', aL[1]
        # print '  -> ratio     :', aL[1]/rossFromTable
        mu0 = np.sqrt(1-rossFromTable**2)
        rossMaxVar = ross/rossFromTable
        rossInfty = aL[1]/rossFromTable
        ross = None
    else:
        rossMaxVar = 1.
        rossInfty = 1.
        mu0 = 0.0
        aL = [0,1]

    # -- Compute the visibility profile
    r = np.sqrt(1-data['mu']**2)/np.sqrt(1-mu0**2)
    Imu_ud = (r<=1)#*np.abs(np.gradient(data['mu'])/np.gradient(r))

    # -- Changes of variable mu->r
    #dr_dmu = np.abs(np.gradient(r)/np.gradient(data['mu']))

    # -- dr_dmu does not work :(
    dr_dmu = np.ones(len(r))

    x1 = np.linspace(0, 4.2, 100) # first lobe, x1[0] needs to be 0!
    # -- range for power law fit in [pi B theta / lambda]
    x2 = np.linspace(.5, 8., 300) # first and second lobes

    # -- no dr_dmu here because the intensity is already in r
    V2ud = np.trapz(special.jv(0,x1[None,:]*r[:,None])*
                    Imu_ud[:,None]*r[:,None]*dr_dmu[:,None],
                    r[:,None], axis=0)**2
    # -- normalize by 0-frequency
    V2ud /= np.trapz(special.jv(0,0)*Imu_ud*r*dr_dmu, r)**2

    if plot:
        # -- Rosseland
        ax1.plot([1., 1], [0, 1], '-', color='k', alpha=0.5)
        # -- Outer Diameter
        #ax1.plot([1/rossFromTable, 1/rossFromTable], [0,1], '-', color='0.5')
        #ax1.text(1/rossFromTable,0.02, "Model's outer radius", color='0.5',
        #        rotation=90, va='bottom')
        if not ax2 is None:
            ax2.plot(x1, V2ud, '-k', linestyle='dashed')
        if not ax0 is None:
            ax0.vlines(mu0, 0.001, 0.9, color='k', linestyle='dashed')
            ax0.text(mu0, 0.72, 'limb', ha='center', va='bottom')
            if mu0>0:
                ax0.set_xlim(max(0,mu0-0.05), mu0+0.05)
            else:
                ax0.set_xlim(0., 0.05)

    for j,b in enumerate(['B','V','R','I','H','K'][::-1]):
        if planar:
            p = {'f1':0.0, 'f2':0.0, 'f3':0.0, 'f4':0.0}
        else:
            p = {'mu0':mu0, 'c':.002, 'e':0.4, 'ap':0.1, 'p':.7}
            #p = {'mu0':mu0, 'c':.002, 'e':0.4}

        f = dpfit.leastsqFit(Imu_an, data['mu'][data[b]>reslim], p,
                             data[b][data[b]>reslim],
                             data[b][data[b]>reslim])
        model = Imu_an(data['mu'], f['best'])

        maxres = (data[b]-model)/data[b]
        maxres = maxres[data[b]>reslim]
        maxres = np.abs(maxres).max()

        if plot:
            i = np.where((nei5['Teff']==Teff)*
                         (nei5['logg']==logg)*
                         (nei5['mass']==mass))[0][0]
            p4 = {'f1': nei5['f1(%s)'%b][i],
                  'f2': nei5['f2(%s)'%b][i],
                  'f3': nei5['f3(%s)'%b][i],
                  'f4': nei5['f4(%s)'%b][i]}
            pro4 = Imu_an(data['mu'], p4)
            maxres2 = (data[b]-pro4)/data[b]
            maxres2 = maxres2[data[b]>reslim]
            maxres2 = np.abs(maxres2).max()

            # -- analytical fit
            ax1.plot(r, data[b], '-', color=colors[b], label=b,
                     linewidth=2)
            #axv.plot(r, -np.gradient(data[b])/np.gradient(r), '.-', color=colors[b], label=b)
            if not ax0 is None:
                ax0.plot(data['mu'], data[b], '.', color=colors[b],
                         alpha=0.5, label='data' if j==0 else '')
                ax0.plot(data['mu'], model, '-',
                         color=colors[b], label='5P Merand' if j==0 else '')
                if showNeilson:
                    ax0.plot(data['mu'], pro4, '--',
                             color=colors[b], label='4P' if j==0 else '')
            if b=='Kep':
                fig3[0].plot(data['mu'], data[b], '-k', linewidth=2, label='SATLAS',
                             alpha=0.2)
                fig3[0].plot(data['mu'], model, '-r', linewidth=2, label='5P Merand',
                             linestyle='dotted', alpha=0.8)
                fig3[0].plot(data['mu'], pro4, '-k', linewidth=2, label='4P Neilson',
                             linestyle='dashed', alpha=0.6)
                fig3[0].set_ylim(-0.1,1)
                fig3[1].plot(data['mu'], data[b]-model, '-r', linewidth=2, label='5P Merand',
                             linestyle='dotted', alpha=0.8)
                fig3[1].plot(data['mu'], data[b]-pro4, '-k', linewidth=2, label='4P Neilson',
                             linestyle='dashed', alpha=0.6)
                fig3[0].set_xlim(1,0)
                fig3[1].set_ylim(-0.1,0.1)
                fig3[0].set_ylabel(r'Kepler I / Imax')
                fig3[1].set_xlabel(r'$\mu$')
                fig3[1].set_ylabel(r'$\Delta$ = SATLAS - Analytical')
                fig3[0].set_title(os.path.basename(filename))
                fig3[0].legend(loc='lower left')

            if not axp is None:
                axp.plot(np.maximum(np.minimum(100*(data[b]-model)/data[b], 1000),-1000),
                            data[b], '-', color=colors[b],
                             label=b+' $|\delta|$<%2.0f%%'%(100*maxres))
                if showNeilson:
                    axp.plot(np.maximum(np.minimum(100*(data[b]-pro4)/data[b], 1000), -1000),
                             data[b], '-', color=colors[b], linestyle='dashed',
                             label=b+' $|\delta|$<%2.0f%%'%round(100*maxres2,-2))

        # -- compute visibility of true profile, in the first lobe
        V2 = np.trapz(special.jv(0,x1[None,:]*r[:,None])*
                 (data[b][:,None]*r[:,None])*dr_dmu[:,None], r[:,None], axis=0)**2
        V2 /= V2[0]

        if plot and not ax2 is None:
           ax2.plot(x1, V2, color=colors[b])

        c = np.interp(V2[::-1], V2ud_x(x1[::-1]), x1[::-1])[::-1]/x1
        fit = dpfit.leastsqFit(UDLD_x, x1[len(c)/4:],
                         {'A0':c[2:].mean(), 'Ap':0.05, 'p':2.8},
                         c[len(c)/4:])
        res[b] = fit['best']
        res[b]['_V2'] = V2
        res[b]['_x'] = x1
        res[b]['maxres %3.1e'%reslim] = maxres
        res[b]['Imu_an'] = f['best']
        if plot:
            # -- UD/Ross
            ax3.plot(x1[len(c)/8:], c[len(c)/8:], '.',
                     color=colors[b], label=b)

        # -- determine alpha LD
        # -- compute visibility of true profile, in the 1rst & 2nd lobe
        V2 = np.trapz(special.jv(0,x2[None,:]*r[:,None])*
                 (data[b][:,None]*r[:,None]), r[:,None], axis=0)**2
        V2 /= np.trapz(special.jv(0,0)*data[b]*r, r)**2

        #V2 /= V2[0]
        _c = np.pi*np.pi/(180*3600.*1000)
        alpha = dpfit.leastsqFit(v2Alpha, x2/_c, {'diam':1, 'alpha':0.2},
                                V2, verbose=0)
        res[b]['alpha'] = alpha['best']
        if plot:
            ax4.plot(x2, V2, '-', color='k', alpha=0.5)
            lab = ''#b+':'
            lab += r'$\alpha$='
            #lab += '%5.3f $\pm$ %5.3f'%(alpha['best']['alpha'], alpha['uncer']['alpha'])
            lab += '%5.3f'%(alpha['best']['alpha'])
            lab += r' LD-Ross='
            lab += '%5.2f'%(100*(alpha['best']['diam']-1))
            lab += '%'
            ax4.plot(x2, alpha['model'], '-', color=colors[b],
                linestyle='dashed', label=lab)
            #ax4r.plot(x2, (V2-alpha['model'])/V2, color=colors[b],)
            ax4r.plot(x2, 1e4*(V2-alpha['model']), color=colors[b],)
            _r = np.linspace(0.95, 1.0, 100)
            _mu = np.sqrt(1-_r**2)
            #ax1.plot(_r*alpha['best']['diam'], _mu**alpha['best']['alpha'],
            #    color=colors[b], linestyle='dotted')

    if plot:
        if not ax0 is None:
            ax0.legend(loc='upper left', prop={'size':10})
            axp.hlines(reslim, -100, 100, linestyle='dotted',
                    label='I/Imax=%3.1f%%'%(100*reslim))
            axp.vlines([-5, 5], 1e-3, 1, label='+- 5%', linestyle='dashed')
            if showNeilson:
                axp.legend(loc='lower left', prop={'size':10})
            else:
                axp.legend(loc='upper right', prop={'size':10})
            axp.semilogy();

        ax1.legend(loc='upper right', prop={'size':8})
        ax3.legend(loc='lower left', prop={'size':8})
        ax3.grid()
        ax4.set_xlim(x2.min(),x2.max())
        #ax4.set_ylim(0,0.016)
        ax4.set_yscale('log')
        ax4.set_ylim(1e-6, 1)
        #ax4.set_xlim(3.5,x2.max())
        ax4.legend(loc='lower left', prop={'size':6})
    else:
        return res

def Imu_an(mu,p):
    if 'mu0' in p.keys():
        # -- sigmoid:
        res = (1+np.exp(-(1-p['mu0'])/p['c']))/(1+np.exp(-(mu-p['mu0'])/p['c']))
        if 'e' in p.keys():
            res = res**p['e']
        mup = (mu-p['mu0'])/(1-p['mu0'])
    else:
        res = np.ones(len(mu))
        mup = mu
    c = 1.
    for k in p.keys():
        if k.startswith('f'): # 4 parameters law
            # if float(k[1:])%2==0:
            #     c -= p[k]*(1-mup**(float(k[1:])/2.))
            # else:
            #     c -= p[k]*(1-np.abs(mup)**(float(k[1:])/2.))
            c -= p[k]*(1-mu**(float(k[1:])/2.))

    if 'ap' in p.keys() and 'p' in p.keys():
        c -= p['ap']*(1-np.sign(mup)*np.abs(mup)**p['p'])

    res *= c
    return res

def V2ud_x(x):
    """
    monochromatic visibility of uniform disk
    x = np.pi*B_m*theta_rad/lambda_m
    """
    return (2*special.j1(x+1e-6*(x==0))/(x+1e-6*(x==0)))**2

def UDLD_x(x, p):
    """
    x = np.pi*B_m*theta_rad/lambda_m
    p = {'A0':..., 'Ap':..., 'p':...}
    """
    return p['A0'] - (p['Ap']*x)**[p['p']]

# ----------------------
def maxVar(I,r):
    """
    find value of r corresponding to the maximum value of dI/dr
    """
    g = np.abs(np.gradient(I)/np.gradient(r))
    i_0 = np.argmax(g)
    # -- find maximum by polynomial interpolation
    i = np.array([i_0-2,i_0-1,i_0,i_0+1,i_0+1])
    c = np.polyfit(r[i], g[i],2)
    return -c[1]/(2*c[0])

def v2ud_fit(bl, param):
    """
    *monochromatic* visibility of uniform disk, unless 'R' is defined
    bl = Baseline / wavelength in m/um
    """
    x = np.pi*param['diam']*np.pi/(180*3600.*1000)*bl
    if not 'R' in param.keys():
        return (2*special.j1(x+1e-6*(x==0))/(x+1e-6*(x==0)))**2
    else:
        _V2 = np.zeros(len(x))
        nB = 5
        for z in np.linspace(-0.5, 0.5, nB):
            _V2 += 1./nB*(2*special.j1(x*(1+z/param['R'])+1e-6*(x==0))/(x*(1+z/param['R'])+1e-6*(x==0)))**2
        return _V2

def v2_interp(bl, param):
    """
    v2 function for fit using SATLAS spherical atmospheric models.

    bl = Baseline / wavelength in m/um
    """
    global _mdata
    x = np.pi*param['diam']*np.pi/(180*3600.*1000)*bl
    if 'R' in param.keys():
        #-- bandwitdth smearing:
        _V2 = np.zeros(len(x))
        nB = 5
        for z in np.linspace(-.5, .5, nB):
            _V2 += np.interp(x*(1+z/param['R']), _mdata['x'], _mdata['V2_'+param['band']])/nB
    else:
        _V2 = np.interp(x, _mdata['x'], _mdata['V2_'+param['band']])
    return _V2

def fitDiamModel(oifits, model=None, figure=0, bootstrapping=False, firstGuess=3.0,
                compareC4 = True):
    """
    fit V2 in OIFITS data file ("oifits") using a V2 profile computed from an SATLAS models.

    Models can be found at: ftp://cdsarc.u-strasbg.fr/J/A+A/554/A98/spheric
    and should downloaded locally. "model" should be the address of a .dat model

    firstGuess is the expected diameter in mas
    """
    global _mdata, rossTable, C4

    #-- Read OIFITS file ---------------------
    # -- effective wavelengths
    wl = {0.44:'B', 0.55:'V', 0.71:'R', 0.97:'I', 1.62:'H', 2.22:'K'}
    _wl = np.array(wl.keys())
    f = pyfits.open(oifits) # -- assumes single target, single instrument!
    oidata = {'B/l':[], 'v2':[], 'ev2':[], 'station':[]}
    for h in f[1:]:
        if h.header['EXTNAME']=='OI_VIS2':
            b = np.sqrt(h.data['UCOORD']**2+h.data['VCOORD']**2)
            Bl = b[:,None]/f['OI_WAVELENGTH'].data['EFF_WAVE'][None,:]
            oidata['B/l'].extend(list(Bl.flatten()))
            oidata['v2'].extend(list(h.data['VIS2DATA'].flatten()))
            oidata['ev2'].extend(list(h.data['VIS2ERR'].flatten()))
            # -- stations
            sta = [int('%d%d'%tuple(s)) for s in h.data['STA_INDEX']]
            sta = np.array(sta)[:,None] + 0*f['OI_WAVELENGTH'].data['EFF_WAVE'][None,:]
            oidata['station'].extend(list(sta.flatten()))
            avgWl = f['OI_WAVELENGTH'].data['EFF_WAVE'].mean()*1e6
            oidata['R'] = f['OI_WAVELENGTH'].data['EFF_WAVE'].mean()/\
                            f['OI_WAVELENGTH'].data['EFF_BAND'].mean()
    for k in oidata.keys():
        oidata[k] = np.array(oidata[k])
    f.close()
    # -- done reading the OIFITS file -----------------

    # -- determining band to consider in SATLAS model:
    band = wl[_wl[np.abs(_wl-avgWl).argmin()]]
    print 'selecting %s band according to OIFITS wavelength table'%band
    # -- load SATLAS model in _mdata ------
    if model is None:
        model = './DATA/LD_NEILSON/GIANTS/spheric/ld_satlas_surface.2t5800g150m75.dat'
    # -- read file:
    f = open(model)
    cols = ['mu','B','V','R','I','H','K']
    _mdata = {c:[] for c in cols}
    for l in f.readlines():
        for k,c in enumerate(cols):
            _mdata[c].append(float(l.split()[k]))
    for c in cols:
        _mdata[c] = np.array(_mdata[c])
    f.close()
    _mdata['Teff'] = int(model.split('surface.2t')[1].split('g')[0])
    _mdata['logg'] = float(model.split('surface.2t')[1].split('g')[1].split('m')[0])/100.
    _mdata['mass'] = float(model.split('surface.2t')[1].split('g')[1].split('m')[1].split('.')[0])/10.
    # -- compute visibility to be interpolated:
    xmax = np.pi*oidata['B/l'].max()*(firstGuess*np.pi/(180*3600*1000.))
    xmin = np.pi*oidata['B/l'].min()*(firstGuess*np.pi/(180*3600*1000.))
    x = np.linspace(xmin/3., 3.*xmax, 1000) # extended spatial frequency range
    r = np.sqrt(1-_mdata['mu']**2)
    I = _mdata[band]
    # -- find Rosseland diam from files sent by Hilding:
    i0 = np.argmin(((rossTable['Teff']-_mdata['Teff'])/1000.)**2 +
                      (rossTable['logg']-_mdata['logg'])**2+
                      (rossTable['mass']-_mdata['mass'])**2)
    r0 = rossTable['Ross/Outer'][i0]
    print 'Ross. Table=%4.3f; max var=%4.3f'%(rossTable['Ross/Outer'][i0],
                                             maxVar(I, r))
    r /= r0
    _mdata['r'] = r
    V2 = np.trapz(special.jv(0,x[None,:]*r[:,None])*I[:,None]*r[:,None],
                              r[:,None], axis=0)**2
    V2_0 = np.trapz(special.jv(0,0*r)*(I*r), r)**2
    V2 /= V2_0
    _mdata['V2_'+band] = V2
    _mdata['x'] = x

    #-- Done loading the model ------

    # -- fitting LD and UD models:
    fit = dpfit.leastsqFit(v2_interp, oidata['B/l'],
                         {'diam': firstGuess, 'band':band, 'R':oidata['R']},
                         oidata['v2'], oidata['ev2'], verbose=0,
                         fitOnly=['diam'])
    # print 'wl smeared SATLAS: diam = %6.4f +- %6.4f mas (chi2=%4.3f)'%(fit['best']['diam'],
    #                                                        fit['uncer']['diam'],
    #                                                        fit['chi2'])

    fitUD = dpfit.leastsqFit(v2ud_fit, oidata['B/l'],
                         {'diam': firstGuess, 'R':oidata['R']}, oidata['v2'],
                         oidata['ev2'], fitOnly=['diam'], verbose=0)
    # print 'wl smeared UD    : diam = %6.4f +- %6.4f mas (chi2=%4.3f)'%(fitUD['best']['diam'],
    #                                                          fitUD['uncer']['diam'],
    #                                                          fitUD['chi2'])

    if compareC4:
        # -- J/A+A/529/A75/tableeq5
        c4_model = 'PHOENIX' # 'ATLAS' or 'PHOENIX'
        c4_band = band+'_'+c4_model
        d = [(k[0]-_mdata['Teff'])**2/1e6 +
             (k[1]-_mdata['logg'])**2 +
             (k[2])**2 + (k[3]-2.)**2 for k in C4[c4_band].keys()]
        k = C4[c4_band].keys()[np.argmin(d)]
        print k,
        p = {'diam': firstGuess, 'R':oidata['R']}
        p.update({'a'+str(i+1):C4[c4_band][k][i] for i in range(4)})
        print p
        fitC4 = dpfit.leastsqFit(v2Claret4, oidata['B/l'],p,
                             oidata['v2'], oidata['ev2'], fitOnly=['diam'], verbose=0)

    if oidata['B/l'].max()*1e-6*fit['best']['diam']>300:
        fitAlpha = dpfit.leastsqFit(v2Alpha, oidata['B/l'],
                         {'diam': fit['best']['diam'], 'alpha':0.0, 'R':oidata['R']},
                         oidata['v2'], oidata['ev2'], verbose=0, doNotFit=['R'])
    else:
        print '!!!', oidata['B/l'].max()*1e-6*fit['best']['diam']
        fitAlpha = None

    # -- bootstrapping on baselines:
    if bootstrapping:
        fits = []
        stations = list(set(oidata['station']))
        print stations
        for i,s in enumerate(itertools.combinations_with_replacement(stations, len(stations))):
            datap = {k:[] for k in oidata.keys()}
            for ij in s:
                for k in datap.keys():
                    datap[k].extend(list(oidata[k][np.where(oidata['station']==ij)]))
            for k in datap.keys():
                datap[k] = np.array(datap[k])
            fits.append(dpfit.leastsqFit(v2ud_fit, datap['B/l'],
                             {'diam': 2.5}, datap['v2'], datap['ev2'], verbose=0))
            fits[-1]['B/l'] = datap['B/l']

    if not figure is None:
        plt.close(figure)
        plt.figure(figure, figsize=(11,7))
        plt.suptitle(os.path.basename(oifits)+
                     '\nSATLAS [%s-band] Teff=%4.0fK logg=%4.2f M=%3.1fMsol'%(
                     band, _mdata['Teff'], _mdata['logg'], _mdata['mass']), fontsize=16)

        ax2 = plt.subplot(121)
        ax2.plot(_mdata['r']*fit['best']['diam']/2., _mdata[band], color='r',
                 label=r'SATLAS: $\theta=$ %5.3f $\pm$ %5.3f mas, $\chi^2$=%4.2f'%(fit['best']['diam'],
                                                           fit['uncer']['diam'],
                                                           fit['chi2']))
        # ax2.plot(_mdata['r']*fitUD['best']['diam']/2., _mdata['r']<=1, color='b',
        #          label=r'UD: $\theta=$ %5.3f $\pm$ %5.3f mas, $\chi^2$=%4.2f'%(fitUD['best']['diam'],
        #                                                    fitUD['uncer']['diam'],
        #                                                    fitUD['chi2']))
        ax2.set_xlabel(r'r (mas)')
        ax2.set_ylabel('I/Imax')
        mu = np.linspace(0,1,500)
        if compareC4:
            r = fitC4['best']['diam']/2.*np.sqrt(1-mu**2)
            I = 1.
            for i in [1,2,3,4]:
                I -= fitC4['best']['a'+str(i)]*(1-mu**(i/2.))
            ax2.plot(r, I, '-g', label=r'C4 %s: $\theta=$ %5.3f $\pm$ %5.3f, $\chi^2$=%4.2f'%(
                                                      c4_model,
                                                      fitC4['best']['diam'],
                                                      fitC4['uncer']['diam'],
                                                      fitC4['chi2']))

        ax3 = plt.subplot(222)
        ax3.errorbar(oidata['B/l'], oidata['v2'], yerr=oidata['ev2'], fmt='.k',
                    alpha=0.2)
        Bl = np.linspace(0.9*oidata['B/l'].min(), 1.1*oidata['B/l'].max(), 300)
        ax3.plot(Bl, v2_interp(Bl, fit['best']), '-r', label='SATLAS')
        #ax3.plot(Bl, v2ud_fit(Bl, fitUD['best']), '-b', label='UD')
        if compareC4:
            ax3.plot(Bl, v2Claret4(Bl, fitC4['best']), '-g', label='Claret 4')

        if not fitAlpha is None:
            Bl = np.linspace(0.9*oidata['B/l'].min(),
                             1.1*oidata['B/l'].max(), 100)
            ax3.plot(Bl, v2Alpha(Bl, fitAlpha['best']), '-y',
                    label='alpha=%5.3f'%fitAlpha['best']['alpha'])
            r = fitAlpha['best']['diam']/2.*np.sqrt(1-mu**2)
            ax2.plot(r, mu**fitAlpha['best']['alpha'],'-y',
                    label=r'$\alpha=%5.3f\pm%5.3f$: $\theta= %5.3f\pm%5.3f$, $\chi^2$=%4.2f'%(fitAlpha['best']['alpha'],
                                                                      fitAlpha['uncer']['alpha'],
                                                                      fitAlpha['best']['diam'],
                                                                      fitAlpha['uncer']['diam'],
                                                                      fitAlpha['chi2']
                                                                      ))
        ax2.legend(loc='lower left', fontsize=8)

        ax3.legend(loc='lower left', fontsize=8)
        ax3.set_ylabel('V$^2$')
        ax3.set_yscale('log')
        ax4 = plt.subplot(224, sharex=ax3)
        ax4.errorbar(oidata['B/l'], (oidata['v2']-fit['model'])/oidata['ev2'],
                    yerr=oidata['ev2'], fmt='.r', alpha=0.5)
        #ax4.errorbar(oidata['B/l'], (oidata['v2']-fitUD['model'])/oidata['ev2'],
        #            yerr=oidata['ev2'], fmt='.b', alpha=0.5)
        if compareC4:
            ax4.errorbar(oidata['B/l'], (oidata['v2']-fitC4['model'])/oidata['ev2'],
                        yerr=oidata['ev2'], fmt='.g', alpha=0.5)

        if not fitAlpha is None:
            ax4.errorbar(oidata['B/l'], (oidata['v2']-fitAlpha['model'])/oidata['ev2'],
                        yerr=oidata['ev2'], fmt='.y', alpha=0.5)

        ax4.grid()
        ax4.set_xlabel('B/$\lambda$')
        ax4.set_ylabel('residuals ($\sigma$)')
        ax4.set_ylim(-max(-plt.ylim()[0], plt.ylim()[1]), max(-plt.ylim()[0], plt.ylim()[1]))
        if bootstrapping:
            plt.figure(figure+1)
            plt.clf()
            ax0 = plt.subplot(311)
            diams = np.array([f['best']['diam'] for f in fits])
            errs = np.array([f['uncer']['diam'] for f in fits])
            plt.hist([f['best']['diam'] for f in fits], bins=50)
            plt.subplot(312, sharex=ax0)
            plt.errorbar( diams, 100*errs/diams, fmt='.k', alpha=0.5,
                         xerr = errs)
            plt.ylabel('err/diam (%)')
            plt.subplot(313, sharex=ax0)

            plt.xlabel('UD diam (mas)')
            plt.ylabel('B/$\lambda$ ')
            for f in fits:
                for j,p in enumerate([0,10,20,30,40]):
                    plt.plot([f['best']['diam'], f['best']['diam']],
                             [np.percentile(f['B/l'], p),
                              np.percentile(f['B/l'], 100-p)], 'k-', alpha=0.1,
                              linewidth=1+2*j)

    return

def _Vld(base, diam, wavel, alpha=0.36):
    """
    Hestroffer LD (power law)
    """
    nu = alpha /2. + 1.
    diam *= np.pi/(180*3600.*1000)
    x = -1.*(np.pi*diam*base/wavel)**2/4.
    V_ = 0
    for k_ in range(100):
        V_ += scipy.special.gamma(nu + 1.)/\
              scipy.special.gamma(nu + 1. + k_)/\
              scipy.special.gamma(k_ + 1.) *x**k_
    return V_

def v2Alpha(bw, param):
    """
    param = {'diam':, 'alpha':}
    """
    if 'R' in param.keys():
        # -- bandwidth smearing:
        res = 0.
        for z in np.linspace(-.4,.4,5):
            res += _Vld(bw, param['diam'], (1.+z/param['R']),
                        alpha=param['alpha'])**2/5.
        return res
    else:
        return _Vld(bw, param['diam'], 1., alpha=param['alpha'])**2

def v2Claret4(bw, param):
    """
    param = {'diam':, 'a1': 'a2':, 'a3':, 'a4':}
    """
    if 'R' in param.keys():
        res = 0.
        d = {k:param[k] for k in param.keys() if k!='R'}
        for z in np.linspace(-.4,.4,5):
            res += v2Claret4(bw*(1+z/param['R']),d)/5.
        return res
    res = _Vld(bw, param['diam'], 1., alpha=0)
    res *= 1-np.sum([param['a'+str(i)] for i in [1,2,3,4]])
    for i in [1,2,3,4]:
        res += param['a'+str(i)]*_Vld(bw, param['diam'], 1., alpha=i/2.)
    return res**2


def V2ld_B74(baseline, params):
    """
    squared vibility for linear LD, from Brown 1974
    params = {'diam' in mas,
              'wavel' in um,
              'linLD'}
    """
    alpha, beta = 1-params['linLD'], params['linLD']
    x = np.pi*baseline*params['diam']*np.pi/(180*3600*1000.)/params['wavel']*1e6
    V2 = alpha*special.j1(x)/x
    V2 += beta*np.sqrt(np.pi/2)*special.jv(1.5, x)/x**(1.5)
    V2 = V2**2
    V2 /= (alpha/2.+beta/3.)**2
    return V2

def _udld_B74(linLD, wavel=0.8):
    """
    linLD ~ 0.5 for R band, Teff=5500K, logg=1 (van Hamme 1993)
    """
    B = np.linspace(20,30) # NPOI, Armstrong+ 2001
    p = {'diam':1.5, 'wavel':0.8, 'linLD':linLD} # delta cep
    # plt.figure(1)
    # plt.clf()
    # plt.subplot(211)
    # plt.plot(B, V2ld_B74(B, p))
    # plt.ylabel('V2')
    ud = {'diam':p['diam']}
    fit = dpfit.leastsqFit(v2ud_fit, B/p['wavel']*1e6, ud, V2ld_B74(B, p))
    plt.plot(B, fit['model'], color='r', linestyle='--')
    print 'UD/LD= %4.3f'%(fit['best']['diam']/p['diam'])

    # -- formula by Claret & Bloemen
    print 'Claret & Bloemen formula:'
    print 'UD/LD= %4.3f'%(((1-linLD/3.)/(1-7*linLD/15.))**(-0.5))

    return
