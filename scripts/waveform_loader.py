import os
import pickle
import yaml
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

class loader:
    
    def __init__(self,  config, gain=3, peaking_time=1.2, sampling_rate=1.0,):
        self.load_flag          = False
        self.filename           = None

        self.df                 = None
        self.dT                 = 1.0/sampling_rate # unit: us
        self.n_samples          = 0
        self.times              = np.arange(0, self.n_samples*self.dT, self.dT)
        self.window_length      = self.n_samples * self.dT

        self.config             = None
        self.configfile_or_dict = config        
        self.chmap              = None
        self.load_config(config)
        self.gain               = gain
        self.peaking_time       = peaking_time
        self.sampling_rate      = sampling_rate


    def _set_waveform_filename(self, file):
        self.filename = file
    
    def load_waveform(self):
        try:
            self.df = pickle.load(open(self.filename, "rb"))[0]
            self.load_flag = True
            print('Done loading')
        except Exception as e:
            print(f'Error occurs {e}')

        self.nevents_total = len(self.df.index)
        self.n_samples = len(self.df.iloc[0]['Data'][0])
        self.times          = np.arange(0, self.n_samples*self.dT, self.dT)
        self.window_length  = self.n_samples * self.dT

        
    def load_config(self, config):
        if (type(config) == str):
            if not os.path.exists(config):
                print(f'{config} file does not exist!') 
                return
            with open(config, 'r') as stream:
                try:
                    self.config = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)
                    
        else:
            self.config = config
        
        if os.path.isfile(self.config['chmap']) == False:
            print(f"Can not find the channel map file: {self.config['channel_map']}")
            self.chmap = None
            return
        
        with open(self.config['chmap'], 'r') as stream:
            try:
                self.chmap = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc) 
    
    def get_channel_type(self, ch):
        if(self.chmap is None):
            print("Channel map didn't properly load")
            return None
        local_ch = ch % 64 #the channel number on the asic level.
        asic = math.floor(ch/64) # the asic ID that this ch corresponds to.
        
        if(asic in self.chmap):
            if(local_ch in self.chmap[asic]["xstrips"]):
                return 'x'
            elif(local_ch in self.chmap[asic]["ystrips"]):
                return 'y'
            else:
                return 'dummy'
            
        else:
            print("Asic {:d} not found in the configuration file channel map".format(asic))
            return None

    def get_channel_pos(self, ch):
        if(self.chmap is None):
            print("Channel map didn't properly load")
            return None
        local_ch = ch % 64 #the channel number on the asic level.
        asic = math.floor(ch/64) # the asic ID that this ch corresponds to.
        tile_pos = self.chmap[asic]["tile_pos"]
        pitch = self.chmap[asic]["strip_pitch"] #in mm

        if(asic in self.chmap):
            if(local_ch in self.chmap[asic]["xstrips"]):
                local_pos = float(self.chmap[asic]["xstrips"][local_ch])
                return (tile_pos[0], tile_pos[1] + np.sign(local_pos)*(np.abs(local_pos) - 0.5)*pitch)
            elif(local_ch in self.chmap[asic]["ystrips"]):
                local_pos = float(self.chmap[asic]["ystrips"][local_ch])
                return (tile_pos[0] + np.sign(local_pos)*(np.abs(local_pos) - 0.5)*pitch, tile_pos[1])
            else:
                return tile_pos #this is a dummy capacitor
        else:
            print("Asic {:d} not found in the configuration file channel map".format(asic))
            return None


    def create_df_from_event(self, evno):
        evdf = pd.DataFrame()
        ev_dict = {}
        ev_dict["Channel"] = []
        ev_dict["Data"] = []
        ev_dict["ChannelType"] = []
        ev_dict["ChannelPos"] = []
        
        ev = self.df.iloc[evno]
        for i, ch in enumerate(ev["Channels"]):
            ev_dict["Channel"].append( ch )
            ev_dict['Data'].append(ev['Data'][i])
            ev_dict["ChannelPos"].append( self.get_channel_pos(ch) )
            ev_dict['ChannelType'].append( self.get_channel_type(ch) )
            
        evdf = pd.DataFrame.from_dict(ev_dict)
        return evdf

    def is_channel_strip(self, ch):
        result = self.get_channel_type(ch)
        if result == "dummy":
            return False
        else:
            return True
        
    def baseline_onechannel_oneevent(self, evno, ch):
        wf = self.df['Data'].iloc[evno][ch]
        conts, edges = np.histogram(wf, bins=50)
        cents = np.array([(edges[i]+edges[i+1])/2 for i in range(len(edges)-1)])
        maxbinid = np.argmax(conts)
        maxbin_cent = cents[maxbinid-1:maxbinid+2]
        maxbin_cont = conts[maxbinid-1:maxbinid+2]
        weighted_cent = np.sum(maxbin_cent*maxbin_cont) / np.sum(maxbin_cont)
        return weighted_cent
        

    def baseline_subtract(self):
        looper = tqdm(self.df.iterrows(), total=len(self.df.index))
        for i, row in looper:
            waves = row["Data"]
            for ch in range(len(waves)):
                base = self.baseline_onechannel_oneevent(i, ch)
                row["Data"][ch] = np.array(waves[ch]) - base
        

            
    def plot_waves(self, evno, chs_to_plot=[], window=[], title='', adc_shift=20):
        if evno < 0:
            evno = 0
        if evno > self.nevents_total:
            print(f"Event {evno} is out of range of the dataframe.")
            return

        ev = self.df.iloc[evno]
        chs = ev["Channels"]
        if chs_to_plot == []:
            chs_to_plot = chs
        waves = ev["Data"]
        
        chs, waves = (list(t) for t in zip(*sorted(zip(chs, waves))))

        nch = len(chs)
        nsamp = len(waves[0])
        times = np.arange(0, nsamp*self.dT, self.dT)

        fig, ax = plt.subplots(figsize=(10, 8))
        for i in range(nch):
            if i in chs_to_plot:
                ax.plot(times, waves[i] + adc_shift*i, label=f'{chs[i]}')
        if window:
            ax.set_xlim(window[0], window[1])
        ax.set_xlabel('Time [us]', fontsize=14)
        ax.set_ylabel("Shifted ADC counts", fontsize=14)
        ax.tick_params(labelsize=12)
        plt.tight_layout()
        plt.show()
        return fig
    
    
    def plot_strip_waveforms_separated(self, evno, fmt=None, fig=None, ax=None, show=True, sep=None):
        if evno < 0:
            evno = 0
        if evno > self.nevents_total:
            print(f"Event {evno} is out of range of the dataframe.")
            return
        ev = self.create_df_from_event(evno)
        nsamp = len(ev['Data'].iloc[0])
        times = np.arange(0, nsamp*self.dT, self.dT)
        
        xstrip_mask = (ev['ChannelType'] == 'x')
        ystrip_mask = (ev['ChannelType'] == 'y')
        xdf = ev[xstrip_mask]
        ydf = ev[ystrip_mask]

        xstrip_img = np.zeros((len(xdf.index), len(times)))
        ystrip_img = np.zeros((len(ydf.index), len(times)))
        xpos, ypos, xchs, ychs = [], [], [], []

        for i, row in xdf.iterrows():
            xpos.append(row['ChannelPos'][1])
            xchs.append(row['Channel'])
        # sort the lists by position
        xpos, xchs = (list(t) for t in zip(*sorted(zip(xpos, xchs))))
        for i, row in xdf.iterrows():
            wave = row['Data']
            ch_idx = xchs.index(row['Channel'])
            for j in range(len(wave)):
                xstrip_img[ch_idx][j] = wave[j] * self.config['mv_per_adc']

        for i, row in ydf.iterrows():
            ypos.append(row['ChannelPos'][0])
            ychs.append(row['Channel'])
        # sort the lists by position
        ypos, ychs = (list(t) for t in zip(*sorted(zip(ypos, ychs))))
        for i, row in ydf.iterrows():
            wave = row['Data']
            ch_idx = ychs.index(row['Channel'])
            for j in range(len(wave)):
                ystrip_img[ch_idx][j] = wave[j] * self.config['mv_per_adc']
        
        
        if sep == None:
            mv_shift = 50 # unit ADC
        else:
            mv_shift = sep
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 16), nrows=2)
        
        curshift = 0.
        if fmt is None:
            fmt = '-'
        for i in range(len(xstrip_img)):
            if (xchs[i] in self.config['dead_channels'] or xchs[i] == self.config['key_channel']):
                ax[0].plot(times, np.array(xstrip_img[i] + curshift), 'k', label=str(xpos[i]))
            else:
                ax[0].plot(times, np.array(xstrip_img[i] + curshift), fmt, label=str(xpos[i]))
            curshift += mv_shift

        curshift = 0
        for i in range(len(ystrip_img)):
            if (ychs[i] in self.config['dead_channels'] or ychs[i] == self.config['key_channel']):
                ax[1].plot(times, np.array(ystrip_img[i] + curshift), 'k', label=str(ypos[i]))
            else:
                ax[1].plot(times, np.array(ystrip_img[i] + curshift), fmt, label=str(ypos[i]))
            curshift += mv_shift

        ax[0].set_xlabel('time [us]', fontsize=13)
        ax[0].set_title(f'X-strips, event {evno}', fontsize=13)
        ax[0].set_ylabel('shfited mV', fontsize=13)
        ax[1].set_xlabel('time [us]', fontsize=13)
        ax[1].set_title(f'Y-strips, event {evno}', fontsize=13)
        ax[1].set_ylabel('shfited mV', fontsize=13)

        if show:
            plt.show()
        return fig, ax