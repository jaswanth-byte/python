import librosa
import numpy as np
import torch
import torch.nn as nn

def extract_features(audio_path: str, sr: int = 16000) -> np.ndarray:
    """Extract Mel Spectrogram features from respiratory audio recordings."""
    y, sr = librosa.load(audio_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec

class RespiratoryClassifier(nn.Module):
    """Simple CNN model for classifying normal vs. adventitious respiratory sounds."""
    def _init_(self, num_classes: int = 4):  # Normal, Crackles, Wheezes, Both
        super()._init_()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_block(x)
        features = torch.flatten(features, 1)
        return self.fc(features)

if _name_ == "_main_":
    model = RespiratoryClassifier(num_classes=4)
    # Batch of 2 spectrogram inputs (Batch, Channels, Height, Width)
    dummy_input = torch.randn(2, 1, 128, 128)
    predictions = model(dummy_input)
    print("Output logits shape:", predictions.shape) 
