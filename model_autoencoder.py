import torch
from torch.nn import Module, BatchNorm2d, Conv2d, ConvTranspose2d


class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        result = args[0] * torch.sigmoid(args[0])
        ctx.save_for_backward(args[0])
        return result

    @staticmethod
    def backward(ctx, *grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Encoder(Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.bn1 = BatchNorm2d(3)
        self.conv1 = Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2d(16)
        self.conv2 = Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn3 = BatchNorm2d(32)
        self.conv3 = Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn4 = BatchNorm2d(64)

        self.swish = Swish.apply
        self.shape_before_flattening = None

    def forward(self, x):
        x = self.bn1(x)

        x = self.conv1(x)
        x = self.bn2(x)
        x = self.swish(x)

        x = self.conv2(x)
        x = self.bn3(x)
        x = self.swish(x)

        x = self.conv3(x)
        x = self.bn4(x)

        self.shape_before_flattening = x.shape[1:]
        return x


class Decoder(Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.bn1 = BatchNorm2d(64)
        self.deconv1 = ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2d(64)
        self.deconv2 = ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = BatchNorm2d(32)
        self.deconv3 = ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = BatchNorm2d(16)

        self.swish = Swish.apply
        self.conv1 = Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.bn1(x)

        x = self.deconv1(x)
        x = self.bn2(x)
        x = self.swish(x)

        x = self.deconv2(x)
        x = self.bn3(x)
        x = self.swish(x)

        x = self.deconv3(x)
        x = self.bn4(x)
        x = self.swish(x)

        x = self.conv1(x)
        x = torch.sigmoid(x)
        return x


class Autoencoder(Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)


class AutoencoderModel:
    def __init__(self, is_custom_pretrained: bool = True):
        self.autoencoder = Autoencoder()
        if is_custom_pretrained:
            self.autoencoder.load_state_dict(torch.load('D:/noise/model_train/autoencoder_30_130424.pth'))
        self.autoencoder.to('cpu').eval()
