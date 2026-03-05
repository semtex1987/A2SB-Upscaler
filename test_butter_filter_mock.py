import unittest
from unittest.mock import patch, MagicMock
import sys

# Mock all dependencies of app.py
mock_numpy = MagicMock()
mock_scipy = MagicMock()
mock_gradio = MagicMock()
mock_matplotlib = MagicMock()
mock_librosa = MagicMock()
mock_pydub = MagicMock()

sys.modules['numpy'] = mock_numpy
sys.modules['scipy'] = mock_scipy
sys.modules['scipy.signal'] = mock_scipy.signal
sys.modules['gradio'] = mock_gradio
sys.modules['matplotlib'] = mock_matplotlib
sys.modules['matplotlib.pyplot'] = mock_matplotlib.pyplot
sys.modules['librosa'] = mock_librosa
sys.modules['librosa.display'] = mock_librosa.display
sys.modules['pydub'] = mock_pydub

# Now import app
import app

class TestButterLowpassFilterMock(unittest.TestCase):
    def test_butter_lowpass_filter_calls_sosfilt_with_correct_axis(self):
        # Setup
        app.butter = MagicMock(return_value='dummy_sos')
        app.sosfilt = MagicMock(return_value='filtered_data')

        # We need to simulate data. Since numpy is mocked, we can just use a mock object
        # or a MagicMock that behaves like a numpy array if needed.
        # But app.py just passes 'data' to sosfilt.
        data = MagicMock()
        cutoff = 4000
        fs = 44100

        # Call
        result = app.butter_lowpass_filter(data, cutoff, fs)

        # Verify
        self.assertEqual(result, 'filtered_data')
        app.sosfilt.assert_called_once()
        args, kwargs = app.sosfilt.call_args

        # Check if axis=0 is passed
        self.assertEqual(kwargs.get('axis'), 0, "sosfilt must be called with axis=0 to handle stereo data correctly")

if __name__ == '__main__':
    unittest.main()
