import argparse
import torch
from bitpack.pytorch_interface import load_quantized_state_dict

parser = argparse.ArgumentParser(description='To unpack models that are packed by BitPack')
parser.add_argument('--device',
                    type=int,
                    default=-1,
                    help='index of target device, -1 means cpu')
parser.add_argument('--input-packed-file',
                    type=str,
                    default=None,
                    help='path to packed file')
parser.add_argument('--original-int-file',
                    type=str,
                    default=None,
                    help='original quantized file in integer format, this is for correctness check')
args = parser.parse_args()

if args.device == -1:
    target_device = torch.device('cpu')
else:
    target_device = torch.device('cuda:' + str(args.device))

unpacked_state_dict = load_quantized_state_dict(args.input_packed_file, target_device)

if args.original_int_file:
    original_state_dict = torch.load(args.original_int_file)['weight_integer']

    for k in original_state_dict.keys():
        if not torch.all(unpacked_state_dict[k].type_as(original_state_dict[k])==original_state_dict[k]):
            print("Error Detected between Unpacked Tensor and Original Tensor with Key Value: ", k)
            print("Unpacked Tensor: ", unpacked_state_dict[k])
            print("Original Tensor: ", original_state_dict[k])
            break
        else:
            print("Correctly Match: ", k)
