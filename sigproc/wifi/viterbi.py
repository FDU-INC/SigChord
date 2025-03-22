import numpy as np
import commpy.channelcoding.convcode as convcode

__trellis = convcode.Trellis(
    np.array([6]), np.array([[0o133, 0o171]]), polynomial_format="Matlab"
)


def encode(input: np.ndarray):
    """
    use polynomial g0 = 0o133, g1 = 0o171 and rate 1/2 to convolutionally
    encode bits

    # parameters:
    - `input`: bits with 6 tail 0 bits

    # return:
    1/2 convolutional encoded bits
    """
    return convcode.conv_encode(input[:-6], __trellis)


def decode(input: np.ndarray):
    """
    use polynomial g0 = 0o133, g1 = 0o171 and rate 1/2 to convolutionally
    decode bits

    # parameters:
    - `input`: 1/2 rate convolutionally encoded bits

    # return:
    decoded bits
    """
    return convcode.viterbi_decode(input, __trellis)

def my_encode(input: np.ndarray):
    # generator, g0 = 0o133, g1 = 0o171
    # k = 7
    conv_a = np.array([1,0,1,1,0,1,1])
    conv_b = np.array([1,1,1,1,0,0,1])
    padded_bits = np.concatenate((np.zeros(6, dtype=int), input))
    a_out = np.convolve(conv_a, padded_bits, "valid") % 2
    b_out = np.convolve(conv_b, padded_bits, "valid") % 2
    res = np.vstack((a_out, b_out)).T.flatten()
    return res

def __encode_decode_test():
    input = np.array([1,0,1,1,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0])
    gt_encoded = np.array([1,1,0,1,0,0,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,1,1,1,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0])
    trellis = convcode.Trellis(np.array([6]), np.array([[0o133, 0o171]]), polynomial_format="Matlab")
    lib_encoded_bits = convcode.conv_encode(input[:-6], trellis)

    decoded_bits = convcode.viterbi_decode(lib_encoded_bits, trellis)
    assert np.all(decoded_bits == input)

    lib_encoded_bits = encode(input)

    decoded_bits = decode(lib_encoded_bits)
    assert np.all(decoded_bits == input)

    encoded_bits = my_encode(input)
    assert np.all(encoded_bits == gt_encoded)
    print("test convolutional encoding pass")


if __name__ == "__main__":
    __encode_decode_test()
