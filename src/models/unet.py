from torch import nn

import torch

# encoding block
class encoding_block(nn.Module):
    """
    Convolutional batch norm block with relu activation (main block used in the encoding steps)
    """
    def __init__(self, in_size, out_size, kernel_size=3, padding=0, stride=1, dilation=1, batch_norm=True, dropout=False):
        super().__init__()


        if batch_norm:

            # reflection padding for same size output as input (reflection padding has shown better results than zero padding)
            layers = [nn.ReflectionPad2d(padding=(kernel_size -1)//2),
                      nn.Conv2d(in_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
                      nn.BatchNorm2d(out_size),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                      nn.Conv2d(out_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
                      nn.BatchNorm2d(out_size),
                      nn.ReLU(inplace=True)]
        else:
            layers = [nn.ReflectionPad2d(padding=(kernel_size - 1)//2),
                      nn.Conv2d(in_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                      nn.Conv2d(out_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
                      nn.ReLU(inplace=True)]

        if dropout:
            layers.append(nn.Dropout())

        self.encoding_block = nn.Sequential(*layers)

    def forward(self, input):

        output = self.encoding_block(input)

        return output


# decoding block
class decoding_block(nn.Module):
    def __init__(self, in_size, out_size, batch_norm=False):
        super().__init__()

        self.conv = encoding_block(in_size, out_size, batch_norm=batch_norm)
        self.up = nn.Sequential(#nn.ReflectionPad2d(padding=1),
                                nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2))

    def forward(self, input1, input2):

        output2 = self.up(input2)

        output1 = nn.functional.upsample(input1, output2.size()[2:], mode='bilinear')

        return self.conv(torch.cat([output1, output2], 1))


class UNet(nn.Module):
    """
    Main UNet architecture
    """
    def __init__(self, num_classes=1):
        super().__init__()

        # encoding
        self.conv1 = encoding_block(3, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = encoding_block(64, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = encoding_block(128, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = encoding_block(256, 512)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        # center
        self.center = encoding_block(512, 1024)

        # decoding
        self.decode4 = decoding_block(1024, 512)
        self.decode3 = decoding_block(512, 256)
        self.decode2 = decoding_block(256, 128)
        self.decode1 = decoding_block(128, 64)

        # final
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, input):

        # encoding
        conv1 = self.conv1(input)
        maxpool1 = self.maxpool1(conv1)
        # print("maxpool1: {}".format(maxpool1.size()))

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        # print("maxpool2: {}".format(maxpool2.size()))


        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        # print("maxpool3: {}".format(maxpool3.size()))


        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        # print("maxpool4: {}".format(maxpool4.size()))

        # center
        center = self.center(maxpool4)
        # print("center: {}".format(center.size()))

        # decoding
        decode4 = self.decode4(conv4, center)
        # print("decode4: {}".format(decode4.size()))

        decode3 = self.decode3(conv3, decode4)
        # print("decode3: {}".format(decode3.size()))

        decode2 = self.decode2(conv2, decode3)
        # print("decode2: {}".format(decode2.size()))

        decode1 = self.decode1(conv1, decode2)
        # print("decode1: {}".format(decode1.size()))


        # final
        final = nn.functional.upsample(self.final(decode1), input.size()[2:], mode='bilinear')
        # print("final: {}".format(final.size()))

        return nn.functional.sigmoid(final) # sigmoid to make it easier to threshold later
