# Standalone regression test for Mojo NVIDIA Float64 sqrt compilation.
#
# Reproduce with:
#   mojo build repro_nvidia_f64_sqrt.mojo \
#     --target-accelerator sm_90 -o /tmp/repro_nvidia_f64_sqrt
#
# Expected output on current nightlies:
#   sqrt(4.0) = 2.0

from layout import TensorLayout, TileTensor, row_major
from std.gpu.host import DeviceContext
from std.math import sqrt


def sqrt_f64_kernel[OutputLayout: TensorLayout](
    output: TileTensor[DType.float64, OutputLayout, MutAnyOrigin],
):
    comptime assert output.flat_rank == 1
    output[0] = sqrt(Float64(4.0))


def main() raises:
    var context = DeviceContext()
    var output_buffer = context.enqueue_create_buffer[DType.float64](1)
    var output_layout = row_major(1)
    var output = TileTensor(output_buffer, output_layout)
    comptime kernel = sqrt_f64_kernel[type_of(output_layout)]
    context.enqueue_function[kernel](
        output,
        grid_dim=1,
        block_dim=1,
    )
    context.synchronize()

    with output_buffer.map_to_host() as mapped_output:
        print("sqrt(4.0) =", mapped_output[0])
