"""
use VO tables of photometric filters. Put filters XML files in
'_dir'. All proper filters will be automatically loaded and made
available to all functions in the module.

ref: http://svo.cab.inta-csic.es/theory/fps3/index.php?mode=browse

http://svo2.cab.inta-csic.es/theory/fps3/fps.php?ID=Generic/Bessell.U

"""
import os, sys
import numpy as np
import urllib2

_dir = '../SPIPS/DATA/FILTERS' # will automatically loads all filters from there
if not os.path.exists(_dir):
    _dir = os.path.join(filter(lambda d: d.endswith('/SPIPS'), sys.path)[0], 'DATA/FILTERS/')


if not os.path.isdir(_dir):
    import sys
    rotastarDir =filter(lambda x: 'SPIPS' in x, sys.path)
    if len(rotastarDir)==1:
        _dir = os.path.join(rotastarDir[0], _dir)
        print '\033[43m', _dir, '\033[0m'

# ------------

def downloadFilters(directory=_dir):
    """
    Attempt to make an automatic dowload a set of commonly used filters from the Spanish
    Virtual Observatory
    """
    files = {}
    # build tree of files
    files['Generic']=[]
    for f in ['U','B','V','R','I']:
        files['Generic'].append('Bessell.'+f)
    for f in ['U','B','V','R','I','J','M']:
        files['Generic'].append('Johnson.'+f)
    for f in ['u','v','b','y']:
        files['Generic'].append('Stromgren.'+f)
    files['OSN']=[]
    for f in ['I', 'R']:
        files['OSN'].append('Circ.Cousins_'+f)
    files['IRAS']=[]
    for f in ['12','25','60','100']:
        files['IRAS'].append('IRAS.'+f+'mu')
    files['2MASS']=[]
    for f in ['J','H','Ks']:
        files['2MASS'].append('2MASS.'+f)
    files['SLOAN']=[]
    for f in ['u','g','r','i','z']:
        files['SLOAN'].append('SDSS.'+f)
    files['CTIO']=[]
    for f in ['J','H','K']:
        files['CTIO'].append('ANDICAM.'+f)
    # load the files and save them
    for k in files.keys():
        for f in files[k]:
            if not os.path.exists(os.path.join(directory, f+'.xml')):
                url = 'http://svo2.cab.inta-csic.es/theory/fps3/fps.php?ID='+k+'/'+f
                print url
                l = urllib2.urlopen(url)
                f = open(os.path.join(directory, f+'.xml'), 'w')
                f.write(l.read())
                f.close()
            else:
                pass
                #print k+'/'+f+'.xml already exists'
    return

def wavelRange(filtname):
    global _data
    return _data[filtname]['profile'][0].min(),\
           _data[filtname]['profile'][0].max()

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

def _atmoTrans(wl):
    """
    wl in um
    """
    global _atmoData
    try:
        n = len(_atmoData)
    except:
        _ad = '../SPIPS/DATA/ATMO/'
        if not os.path.exists(_ad):
            _ad = os.path.join(filter(lambda d: d.endswith('/SPIPS'), sys.path)[0], 'DATA/ATMO/')

        f = open(os.path.join(_ad, 'transmission_300_5000_pwv10.txt'))
        _atmoData = []
        for l in f.readlines():
            _atmoData.append([float(l.split()[0])/1000., float(l.split()[1])])
        f.close()
        _atmoData = np.array(_atmoData)
    expo = 1.5 #  <1. to crudly decrease water vapor, 0 to eliminate it
    return np.interp(wl, _atmoData[:,0], _atmoData[:,1]**expo)

def _ccd(wl):
    # -- http://www.faculty.virginia.edu/rwoclass/astr511/im/ESO-ccdQEs.gif
    # -- "EEV Average"
    _wl = [.3, .325, .35, .375, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95, 1., 1.1]
    _T =  [0., .45, .55, .7,  .8,  .9,  .92, .9, .9, .85, .8, .7,  .57, .45, .3, .15, .5, 0]
    return np.interp(wl, _wl, _T)

def Transmission(filtname, withAtmo='auto'):
    global _data
    if withAtmo == 'auto':
        # -- facilities in space or which already took into account atmo
        space = ['IRAS', 'Spitzer', 'Hipparcos', '2MASS', 'DENIS', 'TYCHO',
                'TESS', 'GAIA', 'MSX', 'AKARI', 'DIRBE', 'HST']
        withAtmo = not any([s.lower() in filtname.lower() for s in space])

    if withAtmo:
        return lambda x: np.interp(x,
                               _data[filtname]['profile'][0],
                               _data[filtname]['profile'][1],
                               left=0, right=0)*_atmoTrans(x)
    else:
        return lambda x: np.interp(x,
                               _data[filtname]['profile'][0],
                               _data[filtname]['profile'][1],
                               left=0, right=0)

def Tag(l, tag):
    return l.split(tag+'="')[1].split('"')[0]

def LoadFilters(directory=_dir):
    """
    quick and dirty reader for .xml files

    for more download VOTable files in directory '_dir' from
    http://svo.cab.inta-csic.es/theory/fps/index.php?mode=browse
    """
    res={}
    download=False
    try:
        filz = os.listdir(directory)
        filz = filter(lambda x: x.endswith('.xml'), filz)
    except:
        pass

    if download or len(filz)==0:
        if not os.path.isdir(directory):
            print '  -> creating', directory
            os.makedirs(directory)
        downloadFilters(directory) # only download the required set of filters
        filz = os.listdir(directory)
        filz = filter(lambda x: x.endswith('.xml'), filz)
    filz.sort()
    for fil in filz:
        f = open(os.path.join(directory, fil))
        lines = f.readlines()
        tmp = {}
        prof = []
        #print '-'*80
        for l in lines:
            if '<PARAM' in l and not 'ProfileReference' in l:
                #print Tag(l, 'name')
                try:
                    tmp[Tag(l, 'name')] = float(Tag(l,'value'))
                except:
                    tmp[Tag(l, 'name')] = Tag(l,'value')
            if l.strip().startswith('<TD>'):
                prof.append(float(l.split('>')[1].split('<')[0]))
        # -- compatibility:
        if not 'ID' in tmp.keys() and 'filterID' in tmp.keys():
            tmp['ID'] = tmp['filterID']
        prof[1] = 0 # force first value in profile to 0
        prof[-1] = 0 # force last value of filter to be 0
        # -- use um for the wavelength, instead of Angstrom
        tmp['profile']=(np.array(prof[::2])*1e-4,np.array(prof[1::2]))
        # -- build a more common name that the 'ID': 'band_system'
        band = tmp['ID'].split('.')[1]

        if tmp['ID'].split('/')[0]=='Generic' or \
               tmp['ID'].split('/')[0]=='Paranal':
            typ = tmp['ID'].split('/')[1].split('.')[0]
        else:
            typ = tmp['ID'].split('/')[0]

        # -- name of filter == name of file
        k = fil.split('.')[:-1] # remove .xml
        if 'Generic' in k:
            k.remove('Generic')
        if len(k)>2 and k[-2]==k[-3]:
            k.pop(-2)
        k = k[-1]+'_'+'_'.join(k[:-1])
        res[k] = tmp
    #__WalravenFilters()
    return res

def writeFilter(filt):
    """
    filt is the dictionarry read from an xml file (or built manually)

    assumes profile's wavelength are in um, will be converted in Angstrom
    """
    f = open('filterTemplate.xml')
    filename = filt['ID'].replace(' ', '').replace('/','_')+'.xml'
    g = open(filt['ID'].replace(' ', '').replace('/','_')+'.xml', 'w')
    for l in f.readlines():
        for k in filt.keys():
            if '__'+k in l:
                #print '>>> found', k
                if k!= 'profile':
                    l = l.replace('__'+k, str(filt[k]))
                else:
                    s = ["<TR>\n <TD>%f</TD>\n<TD>%f</TD>\n</TR>\n"%(10000*filt['profile'][0][i], filt['profile'][1][i])
                        for i in range(len(filt['profile'][0]))]
                    s = ''.join(s)
                    l = l.replace('__'+k, s)
                #print l
        g.write(l)
    f.close()
    g.close()
    return

def _DIRBE(flambda=None):
    """
    import atlas9
    # == crude Vega Model:
    diam = 3.1
    c = np.pi*(diam/2.0)**2 # angular surface, in mas**2
    c *= (np.pi/(180.0*3600*1000))**2 # conversion in rad**2
    flambda = lambda x: c*atlas9.flambda(x, 9800., 3.0)*10e-9

    """
    data = {'wl':[]}
    bands = [1,2,3,4,5,6,7,8,9,10]
    for b in bands:
        data[b] = []
    f = open(os.path.join(_dir, 'dirbe.txt'))
    for l in f.readlines():
        if not l.strip().startswith('#'):
            tmp = np.float_(l.split())
            data['wl'].append(tmp[0]*1e4) # um -> A
            for b in bands:
                data[b].append(tmp[b])
    f.close()

    for b in bands:
        F = {'Description': 'DIRBE '+str(b),
            'ID':'Generic/DIRBE.'+str(b),
            'FWHM': 110,
            'PhotSystem': 'Vega',
            'WavelengthPeak': -1,
            'WavelengthMin': -1,
            'WavelengthMax': -1,
            'WavelengthMean': -1,
            'WavelengthEff': -1,
            'WidthEff': -1,
            }
        _wl = np.array(data['wl'])
        _f = np.array(data[b])
        _i = np.arange(len(_f))
        i_min = min(_i[_f>0])-2
        i_max = max(_i[_f>0])+2
        profile = np.transpose(np.array([_wl[i_min:i_max], _f[i_min:i_max]]))
        F['WavelengthMean'] = np.sum(profile[:,1]*profile[:,0])/np.sum(profile[:,1])
        F['WavelengthEff'] = np.sum(profile[:,1]*profile[:,0])/np.sum(profile[:,1])
        F['WavelengthMin'] = np.min(profile[:,0])
        F['WavelengthMax'] = np.max(profile[:,0])
        F['WavelengthPeak'] = profile[np.argmax(profile[:,1]),0]
        #print F
        F['profile'] = (profile[:,0]*1e-4, profile[:,1])
        if flambda is None:
            F['ZeroPoint'] = 1.0 # fake
        else:
            tmp = np.sum(flambda(profile[:,0]*1e-4)*profile[:,1])/np.sum(profile[:,1])
            F['ZeroPoint'] = convert_Wm2um_to_Jy(tmp, F['WavelengthEff']*1e-4)
            print tmp, 'W/m2/um', F['ZeroPoint'], 'Jy', F['WavelengthEff']*1e-4
            writeFilter(F)

    return data

def _OGLE():
    f = open(os.path.join(_dir, 'QEOGLE.txt'))
    QE = []
    for l in f.readlines():
        QE.append([float(l.split(',')[0]), float(l.split(',')[1])])
    f.close
    QE = np.array(QE)

    V = {'Description': 'OGLE V',
        'ID':'OGLE/SPIPS.V',
        'FWHM': 110,
        'PhotSystem': 'Vega',
        'WavelengthPeak': -1,
        'WavelengthMin': -1,
        'WavelengthMax': -1,
        'WavelengthMean': -1,
        'WavelengthEff': -1,
        'WidthEff': -1,
         }
    f = open(os.path.join(_dir, 'VOGLE.txt'))
    profile = []
    for l in f.readlines():
        profile.append([float(l.split(',')[0]), float(l.split(',')[1])])
    f.close
    profile = np.array(profile)
    profile[0,1] = 0 # 0 at the edge
    profile[-1,1] = 0 # 0 at the edge
    V['WavelengthMean'] = np.sum(profile[:,1]*profile[:,0])/np.sum(profile[:,1])
    V['WavelengthEff'] = np.sum(profile[:,1]*profile[:,0])/np.sum(profile[:,1])
    V['WavelengthMin'] = np.min(profile[:,0])
    V['WavelengthMax'] = np.max(profile[:,0])
    V['WavelengthPeak'] = profile[np.argmax(profile[:,1]),0]
    V['profile'] = (profile[:,0]/10000, profile[:,1]*np.interp(profile[:,0],
                                                        QE[:,0], QE[:,1]))
    V['ZeroPoint'] = convert_Wm2um_to_Jy(3.85263580174e-08, V['WavelengthEff']/1e4)
    writeFilter(V)

    I = {'Description': 'OGLE I',
        'ID':'OGLE/SPIPS.I',
        'FWHM': 110,
        'PhotSystem': 'Vega',
        'WavelengthPeak': -1,
        'WavelengthMin': -1,
        'WavelengthMax': -1,
        'WavelengthMean': -1,
        'WavelengthEff': -1,
        'WidthEff': -1,
         }
    f = open(os.path.join(_dir, 'IOGLE.txt'))
    profile = []
    for l in f.readlines():
        profile.append([float(l.split(',')[0]), float(l.split(',')[1])])
    f.close
    profile = np.array(profile)
    profile[0,1] = 0 # 0 at the edge
    profile[-1,1] = 0 # 0 at the edge
    I['WavelengthMean'] = np.sum(profile[:,1]*profile[:,0])/np.sum(profile[:,1])
    I['WavelengthEff'] = np.sum(profile[:,1]*profile[:,0])/np.sum(profile[:,1])
    I['WavelengthMin'] = np.min(profile[:,0])
    I['WavelengthMax'] = np.max(profile[:,0])
    I['WavelengthPeak'] = profile[np.argmax(profile[:,1]),0]
    I['profile'] = (profile[:,0]/10000, profile[:,1]*np.interp(profile[:,0],
                                                        QE[:,0], QE[:,1]))
    I['ZeroPoint'] = convert_Wm2um_to_Jy(1.14521005341e-08 , I['WavelengthEff']/1e4)
    writeFilter(I)
    return

def _OGLE_MG():
    """
    calibrated from Martin Groenewegen, email from Alex 2017/
    """
    zp_jy = {'u':2012.02405, 'b':3861.51953, 'v':3625.37329, 'i':2369.19482}
    for band in ['u','b','v','i']:
        T = {'Description': 'OGLE %s calibrated by M. Groenewegen'%band.upper(),
            'ID':'OGLE/MG.'+band.upper(),
            'FWHM': 110,
            'PhotSystem': 'Vega',
            'WavelengthPeak': -1,
            'WavelengthMin': -1,
            'WavelengthMax': -1,
            'WavelengthMean': -1,
            'WavelengthEff': -1,
            'WidthEff': -1,
             }
        f = open(os.path.join(_dir, band+'_ogle.dat'))
        profile = []
        for l in f.readlines():
            #print '"'+l+'"'
            if len(l.strip())>5:
                profile.append([float(l.split()[0]), float(l.split()[1])])
        f.close()
        profile = np.array(profile)
        profile[0,1] = 0 # 0 at the edge
        profile[-1,1] = 0 # 0 at the edge
        T['WavelengthMean'] = np.sum(profile[:,1]*profile[:,0])/np.sum(profile[:,1])
        T['WavelengthEff'] = np.sum(profile[:,1]*profile[:,0])/np.sum(profile[:,1])
        T['WavelengthMin'] = np.min(profile[:,0])
        T['WavelengthMax'] = np.max(profile[:,0])
        T['WavelengthPeak'] = profile[np.argmax(profile[:,1]),0]
        T['profile'] = (profile[:,0]/10000, profile[:,1])
        T['ZeroPoint'] = zp_jy[band]
        writeFilter(T)
    pass

def _HipparcosBessell2000():
    """
    http://ulisse.pd.astro.it/Astro/ADPS/ADPS2/FileHtml/index_f135.html
    """
    H = {'Description': 'Hp Hipparcos Bessell 2000 version',
    'ID':'B2000/Hipparcos.Hp',
    'FWHM': 134.0,
    'PhotSystem': 'Hipparcos',
    'WavelengthPeak': 4365.0,
    'WavelengthMin': 3400.0,
    'WavelengthMax': 8800.0,
    'WavelengthMean': 4208.0,
    'WavelengthEff': 4208.0,
    'WidthEff': -1,
     }
    profile = [(3400,  0.000),
            (3500,  0.015),
            (3600,  0.032),
            (3800,  0.103),
            (3900,  0.155),
            (4000,  0.227),
            (4100,  0.300),
            (4200,  0.400),
            (4300,  0.530),
            (4400,  0.700),
            (4500,  0.845),
            (4600,  0.928),
            (4700,  0.970),
            (4800,  1.000),
            (4900,  0.997),
            (5000,  0.988),
            (5100,  0.973),
            (5200,  0.956),
            (5300,  0.935),
            (5400,  0.911),
            (5500,  0.887),
            (5600,  0.858),
            (5700,  0.825),
            (5800,  0.789),
            (5900,  0.752),
            (6000,  0.714),
            (6100,  0.675),
            (6200,  0.637),
            (6300,  0.599),
            (6400,  0.562),
            (6500,  0.525),
            (6600,  0.487),
            (6700,  0.449),
            (6800,  0.411),
            (6900,  0.372),
            (7000,  0.335),
            (7100,  0.299),
            (7200,  0.264),
            (7300,  0.232),
            (7400,  0.203),
            (7500,  0.176),
            (8000,  0.074),
            (8500,  0.020),
            (8800,  0.000),]
    H['profile'] = (np.array(profile)[:,0]/10000, np.array(profile)[:,1])
    # -- fitted to delta cep
    H['ZeroPoint'] = convert_Wm2um_to_Jy(3.752e-8, H['WavelengthEff']/1e4)
    writeFilter(H)
    return

def _WalravenFilters():
    """
    see http://ulisse.pd.astro.it/Astro/ADPS/Systems/Sys_021/index_021.html
    """
    # -- unit conversion: http://www.stsci.edu/hst/nicmos/tools/conversion_form.html
    # -- erg/cm2/s/A -> W/m2/um
    c = 10.

    W = {'Description': '-2.5 Walraven filter W',
    'ID':'Walraven.W',
    'FWHM': 134.0,
    'PhotSystem': 'Walraven',
    'WavelengthPeak': 3270.0,
    'WavelengthMin': 3110.0,
    'WavelengthMax': 3410.0,
    'WavelengthMean': 3254.0,
    'WavelengthEff': 3254.0,
    'WidthEff': -1,
     }
    profile = [[3100, 0.000], [3110, 0.000], [3120, 0.025], [3130, 0.061],
               [3140, 0.130], [3150, 0.215], [3160, 0.300], [3170, 0.388],
               [3180, 0.475], [3190, 0.565], [3200, 0.652], [3210, 0.736],
               [3220, 0.817], [3230, 0.893], [3240, 0.956], [3250, 0.997],
               [3260, 0.995], [3270, 0.953], [3280, 0.889], [3290, 0.814],
               [3300, 0.725], [3310, 0.628], [3320, 0.534], [3330, 0.444],
               [3340, 0.359], [3350, 0.285], [3360, 0.211], [3370, 0.144],
               [3380, 0.080], [3390, 0.028], [3400, 0.005], [3410, 0.000]]
    W['profile'] = (np.array(profile)[:,0]/10000, np.array(profile)[:,1])
    # -- http://ulisse.pd.astro.it/Astro/ADPS/Systems/Sys_021/fig_021_gif.html
    W['ZeroPoint'] = convert_Wm2um_to_Jy(2.12e-11*c, W['WavelengthEff']/1e4)
    writeFilter(W)

    U = {'Description': '-2.5 Walraven filter U',
    'ID':'Walraven.U',
    'FWHM': 260.,
    'PhotSystem': 'Walraven',
    'WavelengthPeak': 3620.0,
    'WavelengthMin': 3380.0,
    'WavelengthMax': 3920.0,
    'WavelengthMean': 3633.0,
    'WavelengthEff': 3633.0,
    'WidthEff': -1,
     }
    profile = [[3380, 0.000],[3390, 0.004],[3400, 0.010],[3410, 0.020],
               [3420, 0.033],[3430, 0.058],[3440, 0.097],[3450, 0.142],
               [3460, 0.192],[3470, 0.245],[3480, 0.300],[3490, 0.356],
               [3500, 0.415],[3510, 0.475],[3520, 0.535],[3530, 0.596],
               [3540, 0.657],[3550, 0.721],[3560, 0.784],[3570, 0.846],
               [3580, 0.901],[3590, 0.948],[3600, 0.982],[3610, 0.995],
               [3620, 1.000],[3630, 0.993],[3640, 0.978],[3650, 0.956],
               [3660, 0.929],[3670, 0.898],[3680, 0.865],[3690, 0.827],
               [3700, 0.787],[3710, 0.742],[3720, 0.691],[3730, 0.636],
               [3740, 0.575],[3750, 0.512],[3760, 0.444],[3770, 0.373],
               [3780, 0.306],[3790, 0.249],[3800, 0.200],[3810, 0.159],
               [3820, 0.127],[3830, 0.100],[3840, 0.076],[3850, 0.058],
               [3860, 0.043],[3870, 0.031],[3880, 0.023],[3890, 0.015],
               [3900, 0.007],[3910, 0.002],[3920, 0.000],]

    U['profile'] = (np.array(profile)[:,0]/10000, np.array(profile)[:,1])
    # -- http://ulisse.pd.astro.it/Astro/ADPS/Systems/Sys_021/fig_021_gif.html
    U['ZeroPoint'] = convert_Wm2um_to_Jy(1.61e-11*c, U['WavelengthEff']/1e4)
    writeFilter(U)

    L = {'Description': '-2.5 Walraven filter L',
    'ID':'Walraven.L',
    'FWHM': 140.,
    'PhotSystem': 'Walraven',
    'WavelengthPeak': 3900.0,
    'WavelengthMin': 3590.0,
    'WavelengthMax': 4105.0,
    'WavelengthMean': 3838.0,
    'WavelengthEff': 3838.0,
    'WidthEff': -1,
     }
    profile = [[3590, 0.000],[3600, 0.003],[3610, 0.006],[3620, 0.011],
               [3630, 0.020],[3640, 0.035],[3650, 0.064],[3660, 0.108],
               [3670, 0.157],[3680, 0.215],[3690, 0.274],[3700, 0.336],
               [3710, 0.401],[3720, 0.468],[3730, 0.538],[3740, 0.609],
               [3750, 0.684],[3760, 0.759],[3770, 0.832],[3780, 0.900],
               [3790, 0.945],[3800, 0.974],[3810, 0.990],[3820, 0.999],
               [3830, 0.998],[3840, 0.985],[3850, 0.961],[3860, 0.929],
               [3870, 0.891],[3880, 0.855],[3890, 0.815],[3900, 0.775],
               [3910, 0.727],[3920, 0.676],[3930, 0.621],[3940, 0.562],
               [3950, 0.496],[3960, 0.425],[3970, 0.356],[3980, 0.299],
               [3990, 0.246],[4000, 0.198],[4015, 0.138],[4030, 0.090],
               [4045, 0.053],[4060, 0.028],[4075, 0.012],[4090, 0.004],
               [4105, 0.000],]

    L['profile'] = (np.array(profile)[:,0]/10000, np.array(profile)[:,1])
    # -- http://ulisse.pd.astro.it/Astro/ADPS/Systems/Sys_021/fig_021_gif.html
    L['ZeroPoint'] = convert_Wm2um_to_Jy(1.52e-11*c, L['WavelengthEff']/1e4)
    writeFilter(L)

    V = {'Description': '-2.5 Walraven filter V',
    'ID':'Walraven.V',
    'FWHM': 706.,
    'PhotSystem': 'Walraven',
    'WavelengthPeak': 5400.0,
    'WavelengthMin': 4800.0,
    'WavelengthMax': 6480.0,
    'WavelengthMean': 5467.0,
    'WavelengthEff': 5467.0,
    'WidthEff': -1,}
    # -- http://ulisse.pd.astro.it/Astro/ADPS/Systems/Sys_021/fig_021_gif.html
    V['ZeroPoint'] = convert_Wm2um_to_Jy(6.73e-12*c, V['WavelengthEff']/1e4)
    profile = [[4775, 0.000],[4800, 0.001],[4825, 0.009],[4850, 0.024],
               [4875, 0.045],[4900, 0.076],[4925, 0.115],[4950, 0.161],
               [4975, 0.209],[5000, 0.264],[5025, 0.320],[5050, 0.380],
               [5075, 0.441],[5100, 0.504],[5125, 0.568],[5150, 0.634],
               [5175, 0.699],[5200, 0.759],[5225, 0.811],[5250, 0.860],
               [5275, 0.902],[5300, 0.938],[5325, 0.967],[5350, 0.986],
               [5375, 0.998],[5400, 0.999],[5425, 0.992],[5450, 0.975],
               [5475, 0.955],[5500, 0.930],[5525, 0.904],[5550, 0.876],
               [5575, 0.847],[5600, 0.812],[5625, 0.779],[5650, 0.746],
               [5675, 0.709],[5700, 0.671],[5725, 0.633],[5750, 0.594],
               [5775, 0.554],[5800, 0.514],[5840, 0.451],[5880, 0.388],
               [5920, 0.321],[5960, 0.258],[6000, 0.201],[6040, 0.152],
               [6080, 0.113],[6120, 0.084],[6160, 0.058],[6200, 0.039],
               [6240, 0.029],[6280, 0.022],[6320, 0.016],[6360, 0.012],
               [6400, 0.008],[6440, 0.004],[6480, 0.000],]
    V['profile'] = (np.array(profile)[:,0]/10000, np.array(profile)[:,1])
    writeFilter(V)

    B = {'Description': '-2.5 Walraven filter B',
        'ID':'Walraven.B',
        'FWHM': 420.,
        'PhotSystem': 'Walraven',
        'WavelengthPeak': .0,
        'WavelengthMin': 3870,
        'WavelengthMax': 4825,
        'WavelengthMean': 4325.0,
        'WavelengthEff': 4325.0,
        'WidthEff': -1,}
    # -- http://ulisse.pd.astro.it/Astro/ADPS/Systems/Sys_021/fig_021_gif.html
    B['ZeroPoint'] = convert_Wm2um_to_Jy(1.23e-11*c, B['WavelengthEff']/1e4)
    profile = [[3870,  0.000],[3880, 0.001],[3890, 0.004],[3900, 0.008],
               [3910, 0.014],[3920, 0.023],[3930, 0.035],[3940, 0.051],
               [3950, 0.071],[3960, 0.092],[3970, 0.112],[3980, 0.136],
               [3990, 0.161],[4000, 0.186],[4015, 0.228],[4030, 0.273],
               [4045, 0.320],[4060, 0.368],[4075, 0.419],[4090, 0.471],
               [4105, 0.523],[4120, 0.575],[4135, 0.627],[4150, 0.682],
               [4165, 0.733],[4180, 0.781],[4195, 0.824],[4210, 0.863],
               [4225, 0.898],[4240, 0.929],[4255, 0.954],[4270, 0.976],
               [4285, 0.990],[4300, 0.998],[4320, 0.997],[4340, 0.985],
               [4360, 0.960],[4380, 0.929],[4400, 0.893],[4420, 0.850],
               [4440, 0.800],[4460, 0.744],[4480, 0.683],[4500, 0.621],
               [4520, 0.562],[4540, 0.504],[4560, 0.445],[4580, 0.390],
               [4600, 0.335],[4620, 0.282],[4640, 0.230],[4660, 0.181],
               [4680, 0.138],[4700, 0.101],[4725, 0.063],[4750, 0.034],
               [4775, 0.015],[4800, 0.003],[4825 , 0.000]]
    B['profile'] = (np.array(profile)[:,0]/10000, np.array(profile)[:,1])
    writeFilter(B)
    return

def _TESS():
    """
    VERY APPROXIMATIVE!
    """
    T = {'Description': 'TESS space mission',
    'ID':'TESS.T',
    'FWHM': 4000.0,
    'PhotSystem': 'TESS',
    'WavelengthPeak': 8800,
    'WavelengthMin': 5800.,
    'WavelengthMax': 11050,
    'WavelengthMean': 8000,
    'WavelengthEff': 8000,
    'WidthEff': -1,
     }
    # http://astronomicaltelescopes.spiedigitallibrary.org/data/Journals/JATIAG/931021/JATIS_1_1_014003_f001.png
    profile = [[580, 0], [600, 0.9], [860, 1.0], [900, .95],
                [920, 0.9], [930, 0.85], [940, 0.8], [960, 0.7],
                [980, 0.6], [1000, 0.45], [1030, 0.2], [1060, 0.1],
                [1100, 0.08], [1130, 0]]

    T['profile'] = (np.array(profile)[:,0]/1000., np.array(profile)[:,1])
    # -- from rotastar.VegaFlux('T_TESS.T')
    T['ZeroPoint'] = convert_Wm2um_to_Jy(1.31837883589e-08, T['WavelengthEff']/1e4)
    writeFilter(T)

def effWavelength_um(filtname):
    global _data
    if isinstance(filtname, str):
        return _data[filtname.strip()]['WavelengthEff']*1e-4
        #return _data[filtname.strip()]['WavelengthMean']*1e-4
    elif isinstance(filtname, list):
        return [effWavelength_um(f) for f in filtname]
    elif isinstance(filtname, np.ndarray):
        return np.array([effWavelength_um(f) for f in filtname])
    else:
        return None

def zeroPoint_Wm2um(filtname):
    global _data
    kz = filtname.strip()
    wl0 = effWavelength_um(filtname)
    return convert_Jy_to_Wm2um(_data[kz]['ZeroPoint'],wl0)

def convert_Jy_to_Wm2um(f_jy, wl_um):
    if isinstance(wl_um, str):
        wl_um = effWavelength_um(wl_um)
    return f_jy/(wl_um**2/2.9979e-12)

def convert_Wm2um_to_Jy(f, wl_um):
    if isinstance(wl_um, str):
        wl_um = effWavelength_um(wl_um)
    return f*(wl_um**2/2.9979e-12)

def convert_mag_to_Wm2um(mag, filtname):
    zp = zeroPoint_Wm2um(filtname)
    return zp*10.0**(-mag/2.5)

def convert_Wm2um_to_mag(flux, filtname):
    zp = zeroPoint_Wm2um(filtname)
    return -2.5*np.log10(flux/zp)

def convert_mag_to_Jy(mag, filtname):
    return convert_Wm2um_to_Jy(convert_mag_to_Wm2um(mag, filtname),
                    effWavelength_um(filtname))

def convert_Jy_to_mag(f, filtname):
    return convert_Wm2um_to_mag(convert_Jy_to_Wm2um(f, filtname), filtname)

_data = LoadFilters()

print ' | available filters from "photfilt2":'

print '   ',
kz = _data.keys()
kz.sort()
for k in kz:
    print '   ', k, '('+_data[k]['ID']+')', _data[k]['Description']
print ' | for more download VOTable files (XML) in', _dir, 'from'
print ' | http://svo.cab.inta-csic.es/theory/fps3/index.php?mode=browse'
print ''
