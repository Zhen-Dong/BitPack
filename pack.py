import argparse
import torch
from bitpack.pytorch_interface import save_quantized_state_dict

parser = argparse.ArgumentParser(description='BitPack to efficiently save mixed-precision models')
parser.add_argument('--input-int-file',
                    type=str,
                    default=None,
                    help='path to the quantized model with integer format')
parser.add_argument('--packed-output-path',
                    type=str,
                    default='./packed_quantized_checkpoint.pth.tar',
                    help='path to output the packed checkpoint')
parser.add_argument('--force-pack-fp',
                    action='store_true',
                    help='if the input is in floating-point form'
                         'whether to force the input tensor to int8 and then pack it')
args = parser.parse_args()

state_dict = torch.load(args.input_int_file)
weight_integer = state_dict['weight_integer']

# If the checkpoint contains integer values stored in floating point format,
# force_pack_fp can convert it to integer tensor and then pack accordingly.
# Here we use int32 to represent temporary results, in order to prevent potential overflow.
if args.force_pack_fp:
    weight_integer = {k : weight_integer[k].type(torch.int32) for k in weight_integer.keys()}

save_quantized_state_dict(weight_integer, args.packed_output_path)
