import os
import argparse
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('-m',
                    '--model-path',
                    type=str,
                    required=True,
                    help='Path to pytorch model')
parser.add_argument('-h',
                    '--height',
                    type=int,
                    required=False,
                    default=32,
                    help='Input height Default is 32')
parser.add_argument('-w',
                    '--width',
                    type=int,
                    required=False,
                    default=32,
                    help='Input width Default is 32.')

FLAGS = parser.parse_args()

if os.path.isfile(FLAGS.model_path):
    model = torch.load(FLAGS.model_path).to(device)
    model.eval()
else:
    raise RuntimeError(f"PyTorch Model not found at {FLAGS.model_path}")
print(f"Setting input shape to (3, {FLAGS.h}, {FLAGS.w})")
dummy_input = torch.random.randn((1, 3, FLAGS.h, FLAGS.w)).to(device)
jit_model = torch.jit.trace(model, dummy_input)
print("Saved jit trace model in current working directory")
jit_model.save('model.pt')