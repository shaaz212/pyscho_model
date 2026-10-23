"""The small face model (student): MobileNetV3-small that takes a 96x96 grey face crop.

The first layer is folded to 1 input channel: for a grey picture repeated 3 times,
summing the 3 channel weights gives exactly the same result, with less work.
Output order is the teacher's: sad, disgust, angry, neutral, fear, surprise, happy.
"""
import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

EMOTIONS = ["sad", "disgust", "angry", "neutral", "fear", "surprise", "happy"]
FER_TO_TEACHER = [2, 1, 4, 6, 0, 5, 3]  # FER-2013 label number -> index in EMOTIONS
SIZE = 96


def build_student(pretrained=True):
    net = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
    first = net.features[0][0]
    folded = nn.Conv2d(1, first.out_channels, first.kernel_size, first.stride, first.padding, bias=False)
    folded.weight.data = first.weight.data.sum(dim=1, keepdim=True)
    net.features[0][0] = folded
    net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, len(EMOTIONS))
    return net


def prepare(images_uint8):
    """(N,48,48) uint8 grey -> (N,1,96,96) float in -1..1, the student's input."""
    x = torch.as_tensor(images_uint8, dtype=torch.float32)[:, None] / 255.0
    x = nn.functional.interpolate(x, size=SIZE, mode="bilinear")
    return (x - 0.5) / 0.5
