import netCDF4 as nc
import numpy as np
from spacepy import pycdf
from scipy.io import readsav
import datetime

day = np.timedelta64(1, 'D')

class MavenData(object):

    def __init__(self, trange=None, instruments=None,
                data_dir='/Volumes/triton/Data/', verbose=True):

        yr, mnth, day = trange[0].split('-')
        self.year = yr
        self.month = mnth
        self.day = day
        self.load_failed = False
        try:
            self.t0 = np.datetime64(trange[0])
        except(ValueError):
            self.load_failed = True
        self.instruments = instruments
        self.data_dir = data_dir
        self.verbose = verbose

        self.data = {}

        if trange is not None and instruments is not None:
            self.get_data()

    def __getitem__(self, key):
        val = self.data.__getitem__(key)
        return val

    def get_data(self):
        if self.load_failed: return
        
        for instr in self.instruments:
            if self.verbose: print 'Loading data from {0}'.format(instr)

            fname = self._get_fname(instr)
            if self.verbose: print 'File: {0}'.format(fname)

            if instr in ['swia',  'swea', 'sta']:
                data = self._get_CDF_data(fname)
            elif instr in ['mag']:
                data = self._get_sav_data(fname)
            else:
                print 'Instrument not supported: {0}'.format(instr)
                raise(RuntimeError)

            if self.load_failed: return

            if 'time_unix' in data.keys():
                if self.verbose: print 'Adjusting datetime'
                self._adjust_time(data)
            else:
                self._create_time(data)
            self.data[instr] = data

    def _get_fname(self, instr):
        data_dir = self.data_dir 
        mav_base = 'maven/data/sci/'

        yr = self.year
        mnth = self.month
        day = self.day
        date_dir = '{0}/{1}/'.format(yr, mnth)

        if instr == 'swea':
            sci_dir = 'swe/l2/'
            sci_name = 'mvn_swe_l2_svyspec_'+ yr+mnth+day+'_v03_r01.cdf'
        if instr == 'swia':
            sci_dir = 'swi/l2/'
            sci_name = 'mvn_swi_l2_onboardsvyspec_'+ yr+mnth+day + '_v01_r01.cdf'
        if instr == 'sta':
            sci_dir = 'sta/l2/'
            sci_name = 'mvn_sta_l2_c6-32e64m_'+yr+mnth+day+ '_v01_r10.cdf'
        if instr == 'mag':
            sci_dir = 'mag/l2/sav/1sec/'
            sci_name = 'mvn_mag_l2_pl_1sec_'+yr+mnth+day+'.sav'

        fname = data_dir + mav_base+ sci_dir+ date_dir+sci_name
        return fname


    def _get_CDF_data(self, fname):
        try:
            with pycdf.CDF(fname) as cdffile:
                    data = cdffile.copy()
            return data
        except:
            print 'Failed to load: ', fname
            self.load_failed = True

    def _get_sav_data(self, fname):
        data = readsav(fname, python_dict=True)

        for k, v in data.items():
            print k, v.shape, v
        return data

    def _adjust_time(self, data):
        import pandas as pd

        unix_time = data['time_unix']
        ms_unix = np.array(unix_time*1000, dtype='int64')
        dtime = np.timedelta64(1, 'ms')*ms_unix+np.datetime64('1970-01-01')
        data['time'] = dtime.astype(datetime.datetime)

    def _create_time(self, data):
        t0 = self.t0
        npts = data['data'].shape[0]
        dt = np.timedelta64(24*3600*1000, 'ms')/npts
        print dt

        time = t0+dt*np.arange(npts)
        data['time'] = time.astype(datetime.datetime)

