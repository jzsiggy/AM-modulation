import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fftpack import fft, fftshift
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import soundfile as sf

class AM_mod:
    def __init__(self, file_name):
        self.fs = 44100
        self.file_name = file_name

    def LPF(self, signal, cutoff_hz, fs):
        from scipy import signal as sg
        # https://scipy.github.io/old-wiki/pages/Cookbook/FIRFilter.html
        nyq_rate = fs/2
        width = 5.0/nyq_rate
        ripple_db = 120.0 #dB
        N , beta = sg.kaiserord(ripple_db, width)
        taps = sg.firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))
        return( sg.lfilter(taps, 1.0, signal))

    def calcFFT(self, signal, fs):
        # https://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html
        N  = len(signal)
        T  = 1/fs
        xf = np.linspace(-1.0/(2.0*T), 1.0/(2.0*T), N)
        yf = fft(signal)
        return(xf, fftshift(yf))

    def generateCos(self, freq, time, fs):
        n = time*fs
        x = np.linspace(0.0, time, n)
        s = np.cos(freq*x*2*np.pi)
        return (x, s)

    def parseAudio(self):
        audio, sr = sf.read(self.file_name)
        self.audio = audio[:,1]

    def normalizeAudio(self):
        self.normalized_audio = self.audio / abs(self.audio.min())

    def filterAudio(self, freq):
        self.filtered_audio = self.LPF(self.normalized_audio, freq, self.fs)

    def getCarrier(self, freq):
        T = 5
        X,Y = self.generateCos(freq,T,self.fs)
        self.carrier = Y[0:len(self.audio)]

    def modulate(self):
        self.modulated_signal = self.filtered_audio * self.carrier

    def demodulate(self):
        self.demodulated_signal = self.modulated_signal * self.carrier

    def play(self, signal):
        sd.play(signal, samplerate=48000)
        sd.wait()

    def plot(self, title, signal, xlim_signal, xlim_fourrier):
        plt.figure()
        plt.plot(signal)
        plt.title("Sinal {}".format(title))
        plt.xlim(0, xlim_signal)
        plt.show()

        plt.figure()
        X, Y = self.calcFFT(signal, self.fs)
        plt.plot(X,np.abs(Y))
        plt.xlim(0, xlim_fourrier)
        plt.title("{} -- Fourier".format(title))
        plt.show()

# sd.default.device = 0
# sd.default.channels = 1

entrega = AM_mod('CamFis.wav')
entrega.parseAudio()
entrega.normalizeAudio()
entrega.play(entrega.audio)
entrega.filterAudio(4000)
# entrega.play(entrega.filtered_audio)
entrega.getCarrier(14000)
entrega.modulate()
# entrega.play(entrega.modulated_signal)
entrega.plot(title='original', signal=entrega.audio, xlim_signal=150000, xlim_fourrier=9000)
entrega.plot(title='normalizado', signal=entrega.normalized_audio, xlim_signal=150000, xlim_fourrier=9000)
entrega.plot(title='filtrado', signal=entrega.filtered_audio, xlim_signal=200000, xlim_fourrier=9000)
entrega.plot(title='modulado', signal=entrega.modulated_signal, xlim_signal=200000, xlim_fourrier=20000)
entrega.demodulate()
entrega.plot(title='Demodulado', signal=entrega.demodulated_signal, xlim_signal=200000, xlim_fourrier=30000)
# entrega.play(entrega.demodulated_signal)
