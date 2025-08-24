import io

import torchaudio


def bytes_to_tensor(audio_bytes: bytes, target_sample_rate: int = 16_000):
    """Converts audio bytes to a waveform tensor and sample rate using torchaudio.
    Args:
        audio_bytes (bytes): The audio data in bytes format.

    Returns:
        waveform (torch.Tensor): The audio waveform tensor.
        sample_rate (int): The sample rate of the audio.
    """

    # Wrap bytes in a buffer
    audio_buffer = io.BytesIO(audio_bytes)

    # Load with torchaudio
    waveform, sample_rate = torchaudio.load(audio_buffer)

    # Resample if needed
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    # waveform: torch.Tensor [channels, time]
    # sample_rate: int
    return waveform.reshape(-1), target_sample_rate
