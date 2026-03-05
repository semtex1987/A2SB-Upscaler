import numpy as np
import unittest
import sys
from unittest.mock import MagicMock

# Mock dependencies that might be missing
mock_gradio = MagicMock()
mock_matplotlib = MagicMock()
mock_librosa = MagicMock()
mock_pydub = MagicMock()

sys.modules['gradio'] = mock_gradio
sys.modules['matplotlib'] = mock_matplotlib
sys.modules['matplotlib.pyplot'] = mock_matplotlib.pyplot
sys.modules['librosa'] = mock_librosa
sys.modules['librosa.display'] = mock_librosa.display
sys.modules['pydub'] = mock_pydub

# Import the actual function from app
try:
    from app import butter_lowpass_filter
    from scipy.signal import butter, sosfilt
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

@unittest.skipUnless(HAS_SCIPY, "scipy and numpy are required for this test")
class TestStereoProcessing(unittest.TestCase):
    def setUp(self):
        self.fs = 44100
        self.cutoff = 4000
        self.duration = 0.1
        self.t = np.linspace(0, self.duration, int(self.fs * self.duration), endpoint=False)

        # Signal with 1kHz (pass) and 10kHz (stop) components
        sig_low = np.sin(2 * np.pi * 1000 * self.t)
        sig_high = np.sin(2 * np.pi * 10000 * self.t)
        self.sig = sig_low + sig_high

        # Stereo: different signals on channels to better see the bug
        # Channel 0: low + high
        # Channel 1: low only
        self.data_stereo = np.column_stack((self.sig, sig_low))

    def test_fixed_implementation_stereo(self):
        # Using the actual function from app.py
        filtered_stereo = butter_lowpass_filter(self.data_stereo, self.cutoff, self.fs)

        # Filtered mono for reference
        filtered_mono_ch0 = butter_lowpass_filter(self.data_stereo[:, 0], self.cutoff, self.fs)
        filtered_mono_ch1 = butter_lowpass_filter(self.data_stereo[:, 1], self.cutoff, self.fs)

        # In fixed implementation, each channel should match its mono-filtered counterpart
        # because it filters along axis 0 (time).
        self.assertTrue(np.allclose(filtered_stereo[:, 0], filtered_mono_ch0),
                        "Channel 0 filtering incorrect: should filter along time axis")
        self.assertTrue(np.allclose(filtered_stereo[:, 1], filtered_mono_ch1),
                        "Channel 1 filtering incorrect: should filter along time axis")

        # Also verify that it actually did some filtering (high freq attenuated)
        # 10kHz should be significantly attenuated by 4kHz lowpass
        self.assertLess(np.max(np.abs(filtered_mono_ch0)), np.max(np.abs(self.sig)))

if __name__ == '__main__':
    unittest.main()
