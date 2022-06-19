"""
Author: Tingqian Li

Based on:
Author: Sanna Wager
Created on: 9/18/19

This script provides an implementation of pitch shifting using the "time-domain pitch synchronous
overlap and add (TD-PSOLA)" algorithm. The original PSOLA algorithm was introduced in [1].

Description
The main script td_psola.py takes raw audio as input and applies steps similar to those described in [2].
First, it locates the time-domain peaks using auto-correlation. It then shifts windows centered at the
peaks closer or further apart in time to change the periodicity of the signal, which shifts the pitch
without affecting the formant. It applies linear cross-fading as introduced in [3] and implemented in
[4], the algorithm used for Audacity's simple pitch shifter.

Notes:
- Some parameters in the program related to frequency are hardcoded for singing voice. They can be
    adjusted for other usages.
- The program is designed to process sounds whose pitch does not vary too much, as this could result
    in glitches in peak detection (e.g., octave errors). Processing audio in short segment (e.g.,
    notes or words) is recommended. Another option would be to use a more robust peak detection
    algorithm, for example, pYIN [5]
- Small pitch shifts (e.g., up to 700 cents) should not produce many artifacts. Sound quality degrades
    if the shift is too large.
- The signal is expected to be voiced. Unexpected results may occur in the case of unvoiced signals

References:
Overlap and add algorithm exercise from UIUC
[1] F. Charpentier and M. Stella. "Diphone synthesis using an overlap-add technique for speech waveforms
    concatenation." In Int. Conf. Acoustics, Speech, and Signal Processing (ICASSP). Vol. 11. IEEE, 1986.
[2] https://courses.engr.illinois.edu/ece420/lab5/lab/#overlap-add-algorithm
[3] https://www.surina.net/article/time-and-pitch-scaling.html
[4] https://gitlab.com/soundtouch
[5] https://code.soundsoftware.ac.uk/projects/pyin
"""

import pygame
import sys, time
import numpy as np
from numpy.fft import fft, ifft, fftfreq
import scipy, scipy.io, scipy.io.wavfile, scipy.signal, scipy.interpolate
import math

# input is not always in unit of period
#
# |xxxxxxxx|xxxcccccc|ccccccccc|rrrrrrrrr|rrrr
# | period0   period1   period2  period3
# |        |         |         |         +---- in-complete pitch mark
# |        |         |         +----pitch mark 2
# |        |         +--------------pitch mark 1
# |        +------------------------pitch mark 0 (partially from last feed)
# +---------------------------------this is last valid pitch mark of last feed (has been used)
#
# x: remainder sample from last feed
# r: remainder sample of this feed
# c: sample of current feed
#
# output is always in unit of period
# all pitch marks are used to generate as much output periods as possible
# each pitch mark contributes 2 adjacent output periods
# each output period need contributions at least from 2 pitch marks to be qualified to output
# in case of pitch shift up by a lot, each output period will have contributions from more pitch marks
# thus last output period of current feed will need signals from next feed to be able to output
#
# each pitch mark has time stamp value, we use it's index since sample rate fs is fixed.
# output period can be propotional to input period or in fixed value, thus all output pitch marks
# can be determined, and output pitch mark will increase by that period, then it's being mapped
# back to input time axis to find the nearest input pitch mark to sample, if that input pitch mark
# is out of current avaliavle input pitch mark list, we stop output there and output the part that
# no future signal window can added/contribute to it (the left edge of last window).
#
# |xxxxxxxxx|cccccccccc|cccccc....|rrrrrrrrrr|
# |         |          |        | |
# |         |          |        | +--- output pitch mark N (last one from current feed)
# |         |          |        +----- ....
# |         |          +-------------- output pitch mark 1
# |         +------------------------- output pitch mark 0
# +----------------------------------- last pitch mark from previous feed
#
# x: unfinished output periods from previous feed (may be futher modified by OLA)
# r: unfinished output periods from current feed
#
class PSOLA:
    def __init__(self, fs, min_hz=75, max_hz=950) -> None:
        self.fs = fs
        self.min_period = self.fs // max_hz
        self.max_period = self.fs // min_hz
        self.pitch = 0          # pitch up(>0) and down(<0)
        self.formant = 1        # formant stretching
        self.time_scale = 1     # time stretching
        self.fix_pitch = 0      # fixed pitch

        self.remainder = np.zeros([self.max_period])    # remainder of last feed's input signal
        self.y_unfinished = np.zeros([self.max_period])  # last output period from previous feed
        self.last_pitch_mark_ypos = len(self.y_unfinished)
        self.last_pitch_mark_xpos = self.last_pitch_mark_ypos
        self.debug_feed = False
        self.debug_state = False
        self.debug_i = []
        self.debug_o = []
        self.magnitude = 0

    # analyze periods: using FFT to calculate circular auto-correlation
    def get_period(self, x):
        assert(len(x) >= self.max_period)
        fourier = fft(x)
        fourier[0] = 0  # remove DC component
        autoc = ifft(fourier * np.conj(fourier)).real
        return self.min_period + np.argmax(autoc[self.min_period: self.max_period])

    def debug_toggle(self):
        self.debug_state = not self.debug_state
        if self.debug_state:
            self.debug_i = []
            self.debug_o = []
        else:
            scipy.io.wavfile.write("debug_i.wav", self.fs, self.debug_i.astype(np.int16))
            scipy.io.wavfile.write("debug_o.wav", self.fs, self.debug_o.astype(np.int16))

    def feed(self, signal):
        # concatenate input with remainder from last round of feed
        x = np.concatenate((self.remainder, signal))
        sequence = len(x)

        x_period = self.get_period(x)

        # calculate the expected length of output
        f_ratio = 2 ** (self.pitch / 12)
        y_period = x_period / f_ratio

        if self.fix_pitch:
            y_period = ((self.min_period + self.max_period)//2 - self.pitch * (self.max_period - self.min_period)/24)

        pitch_mark_list = np.array([p for p in range(0, sequence, x_period)])
        params = []
        tx = self.last_pitch_mark_xpos
        ty = self.last_pitch_mark_ypos
        y_period_inx = y_period/self.time_scale
        while True:
            ty += y_period
            tx += y_period_inx
            # find the corresponding input pitch mark
            i = np.argmin(np.abs(tx - pitch_mark_list))
            if(i == 0):
                print("WANNING: i=0 is not right!")
                i = 1
            if (i >= len(pitch_mark_list) - 1):
                tx -= y_period_inx
                break

            params.append((pitch_mark_list[i],
                             pitch_mark_list[i] - pitch_mark_list[i-1],
                             pitch_mark_list[i+1] - pitch_mark_list[i],
                             round(ty)))

        # allocate output buffer and copy period from last feed into it
        y = np.zeros([params[-1][3] + params[-1][2]])
        y[0:len(self.y_unfinished)] = self.y_unfinished

        # OLA
        for idx,P0,P1,odx in params:
            debug = 0
            if odx-P0 < 0:
                # this is normal since left period P0 of first pitch mark 
                # is actually determined in previous feed
                P0 = odx
            if odx+P1 > len(y):
                print("WANNING 1:", odx+P1, len(y))
                debug = -9000
                P1 = len(y)-odx

            # OLA
            window = list(np.linspace(0, 1, P0 + 1)[1:]) + list(np.linspace(1, 0, P1 + 1)[1:])

            sw = window * x[idx - P0: idx + P1]

            '''
            self.magnitude = np.amax(x[idx - P0: idx + P1])
            
            wave_replace = np.ones([P1+P0])
            wave_replace[:P0//2] *= -self.magnitude
            wave_replace[P0//2:P0] *= self.magnitude
            wave_replace[P0:P0+P1//2] *= -self.magnitude
            wave_replace[P0+P1//2:] *= self.magnitude
            sw = window * wave_replace
            '''
            # formant scaling is done on windowed pitch
            ix = np.arange(-P0,P1)
            fun = scipy.interpolate.interp1d(ix, sw, kind="linear", bounds_error=False, fill_value=0)

            y[odx-P0: odx+P1] += fun(ix*self.formant)
            if self.debug_state:
                y[odx] += debug
            next_ypos = odx-P0

        next_xpos = (math.floor(sequence/x_period) - 1) * x_period
        self.last_pitch_mark_xpos = tx - next_xpos
        self.last_pitch_mark_ypos = params[-1][3] - next_ypos
        self.y_unfinished = y[next_ypos:]
        self.remainder = x[next_xpos:]

        if self.debug_state:
            self.debug_i = np.concatenate((self.debug_i, signal))
            self.debug_o = np.concatenate((self.debug_o, y[0:next_ypos]))

        return y[0:next_ypos]

class pygamePSOLAPlayer:
    def __init__(self, signal, fs, analysis_win_ms = 40, bypass = False):
        self.signal = signal
        self.channel = None
        self.sequence = int(analysis_win_ms / 1000 * fs)  # analysis sequence length in samples
        self.i0 = 0
        self.psola = PSOLA(fs)
        self.amplitude = 1
        self.last_enqueue_time = time.time()
        self.time_thr = analysis_win_ms/1000/2
        self.bypass = bypass

    def process(self):
        # immediately call channel.get_queue() after channel.queue(sound)
        # would return false negative, WA here is to take some delay
        if (time.time() - self.last_enqueue_time < self.time_thr):
            return

        if (not self.channel) or (not self.channel.get_queue()):
            if self.i0 + self.sequence > len(signal):
                x = self.signal[self.i0:-1]
                self.i0 = self.sequence - len(x)
                x = np.concatenate((x, self.signal[0:self.i0]))
            else:
                x = self.signal[self.i0:self.i0 + self.sequence]
                self.i0 += self.sequence
            
            if (self.bypass):
                new_signal = (x*self.amplitude).astype(np.int16)
            else:
                new_signal = (self.psola.feed(x)*self.amplitude).astype(np.int16)
            # channels failed to change to mono on my system, so make a stereo sound
            play_sig = np.empty(len(new_signal)*2, dtype=new_signal.dtype)
            play_sig[0::2] = new_signal
            play_sig[1::2] = new_signal
            sound = pygame.mixer.Sound(play_sig)
            self.time_thr = len(new_signal)/self.psola.fs/2
            assert(sound)
            if (not self.channel):
                self.channel = sound.play(0)
            else:
                self.channel.queue(sound)
            self.last_enqueue_time = time.time()


if __name__=="__main__":

    fname = "female_scale"
    if len(sys.argv) >= 2:
        fname = sys.argv[1]
        assert(fname.endswith(".wav"))
        fname = fname[0:-4]

    # Load audio
    fs, signal = scipy.io.wavfile.read("{}.wav".format(fname))
    if len(signal.shape) > 1:
        signal = signal[:,0]

    print("{}.wav  sample_rate={}".format(fname, fs))

    pygame.mixer.pre_init(fs, -16, 2, 1024)
    pygame.init()
    print(pygame.mixer.get_init())

    win = pygame.display.set_mode((600,270))
    font = pygame.font.SysFont(None, 48)

    def update_param(selecti = -1):
        global imgs, psola
        labels = []
        labels.append('pitch={:.2f}'.format(psola.psola.pitch))
        labels.append('formant={:.1f}'.format(psola.psola.formant))
        labels.append('time_scale={:.2f}'.format(psola.psola.time_scale))
        labels.append('flags={}{}  '.format(
                    "D" if psola.psola.debug_state else "_",
                    "F" if psola.psola.fix_pitch else "_"))
        imgs = []
        for i, text in enumerate(labels):
            imgs.append((
                (20, 20 + 40*i),
                font.render(text, True, (255, 55, 20) if (i == selecti) else (250,250,0) ,(0,0,0))))

    orignal = pygamePSOLAPlayer(signal, fs, bypass=True)
    psola = pygamePSOLAPlayer(signal, fs)
    psola2 = pygamePSOLAPlayer(signal, fs)
    #psola.psola.pitch = 4
    #psola2.psola.pitch = 7
    mouse_mode = 0
    imgs = []
    update_param()

    def get_img_id_under_mouse():
        global imgs
        x, y = pygame.mouse.get_pos()
        for k, img in enumerate(imgs):
            img_y,img_h = img[0][1],img[1].get_height()
            if y >= img_y and y < img_y + img_h:
                return k
        return -1

    while True:
        #orignal.process()
        psola.process()
        #psola2.process()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_d:
                    psola.psola.debug_toggle(); update_param()
                if event.key == pygame.K_f:
                    psola.psola.fix_pitch = not psola.psola.fix_pitch; update_param()
            elif event.type == pygame.MOUSEMOTION:
                update_param(get_img_id_under_mouse())
            elif event.type == pygame.MOUSEWHEEL:
                wheel_up = event.y > 0
                k = get_img_id_under_mouse()
                if (k == 0):
                    psola.psola.pitch = min(max(psola.psola.pitch + (1 if wheel_up else -1), -24), 24)
                    update_param(k)
                if (k == 1):
                    psola.psola.formant = min(max(psola.psola.formant + (0.1 if wheel_up else -0.1), 0.5), 2)
                    update_param(k)
                if (k == 2):
                    psola.psola.time_scale = min(max(psola.psola.time_scale + (0.1 if wheel_up else -0.1), 0.5), 8)
                    update_param(k)

        win.fill((0, 0, 0))
        for img in imgs:
            win.blit(img[1], img[0], special_flags=pygame.BLEND_RGBA_ADD)
        pygame.display.update()

    pygame.quit()

