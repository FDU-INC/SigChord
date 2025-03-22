import unittest
import torch
import numpy as np

from ..torch import conv_encode, interleave, deinterleave, viterbi_decode

class TestTorch(unittest.TestCase):
    def test_conv_encode1(self):
        input = torch.tensor([[1,0,1,1,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0]])
        output = torch.tensor([[1,1,0,1,0,0,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,1,1,1,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]])

        encoded = conv_encode(input)
        self.assertTrue(torch.all(encoded == output), f"Expected {output}\nBut got {encoded}")

    def test_conv_encode2(self):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "7"
        input = torch.tensor([[1,0,1,1,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0]])
        input = input.cuda()
        output = torch.tensor([[1,1,0,1,0,0,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,1,1,1,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]])
        output = output.cuda()

        encoded = conv_encode(input)
        self.assertTrue(torch.all(encoded == output), f"Expected {output}\nBut got {encoded}")

    def test_interleave1(self):
        from .. import head
        input = torch.tensor([[1,1,0,1,0,0,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,1,1,1,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]])
        gt = head.interleave(input.numpy().flatten(), 1)
        output = interleave(input, 1).flatten().numpy()
        self.assertTrue(np.all(output == gt), f"Expected {gt}\nBut got {output}")

    def test_interleave2(self):
        from .. import head
        input = torch.tensor([[1,1,0,1,0,0,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,1,1,1,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]])
        gt = head.interleave(input.numpy().flatten(), 1)
        output = interleave(input.cuda(), 1).flatten().cpu().numpy()
        self.assertTrue(np.all(output == gt), f"Expected {gt}\nBut got {output}")

    def test_lsig_encoder(self):
        def lsig_encoder(input):
            input = conv_encode(input)
            output = interleave(input, 1)
            return output

        input = torch.tensor([[0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        expected = torch.tensor([[0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 
                         1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 
                         0, 1, 0, 0, 0, 0, 1, 1, 0]])
        output = lsig_encoder(input)
        self.assertTrue(torch.all(output == expected), f"Expected {expected}\nBut got {output}")

    def test_lsig_deinterleave1(self):
        from .. import head
        input = torch.tensor([[1,1,0,1,0,0,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,1,1,1,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]])
        interleaved_input = head.interleave(input.numpy().flatten(), 1)
        output = deinterleave(torch.from_numpy(interleaved_input).unsqueeze(0), 1)
        self.assertTrue(torch.all(output == input), f"Expected {input}\nBut got {output}")

    def test_lsig_deinterleave2(self):
        from .. import head
        input = torch.tensor([[1,1,0,1,0,0,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,1,1,1,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]]).cuda()
        interleaved_input = head.interleave(input.cpu().numpy().flatten(), 1)
        output = deinterleave(torch.from_numpy(interleaved_input).unsqueeze(0).cuda(), 1)
        self.assertTrue(torch.all(output == input), f"Expected {input}\nBut got {output}")

    def test_lsig_decoder1(self):
        def lsig_decoder(input):
            input = deinterleave(input, 1)
            output = viterbi_decode(input)
            return output

        input = torch.tensor([[0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 
                         1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 
                         0, 1, 0, 0, 0, 0, 1, 1, 0]])
        expected = torch.tensor([[0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        output = lsig_decoder(input)
        self.assertTrue(torch.all(output == expected), f"Expected {expected}\nBut got {output}")

    def test_lsig_decoder2(self):
        def lsig_decoder(input):
            input = deinterleave(input, 1)
            output = viterbi_decode(input)
            return output

        input = torch.tensor([[0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 
                         1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 
                         0, 1, 0, 0, 0, 0, 1, 1, 0]]).cuda()
        expected = torch.tensor([[0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).cuda()
        output = lsig_decoder(input)
        self.assertTrue(torch.all(output == expected), f"Expected {expected}\nBut got {output}")