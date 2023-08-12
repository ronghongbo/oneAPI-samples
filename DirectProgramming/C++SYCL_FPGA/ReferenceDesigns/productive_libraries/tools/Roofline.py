###############################################################################
# Copyright 2021 Intel Corporation
#
# Licensed under the BSD-2-Clause Plus Patent License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSDplusPatent
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
#
# SPDX-License-Identifier: BSD-2-Clause-Patent
###############################################################################

# Usage: python Roofline.py is_double_precision, kernel_variation, size, hardware, number_ops, exec_time, number_bytes, fmax
# e.g.   python Roofline.py 0, "sgemm", "large", "a10", number_ops, exec_time, number_byte, 250
# Here is_double_precision means if DSPs are used to compute double precision results; if not, they are used for computing single-precision results.

import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Dictionary from an FPGA model to its #DSPs, and memory bandwidth in GB/s. These parameters below are from DevCloud configurations. Please modify them according to the FPGAs you are using.
hardware_params = {
    "A10" : [1518, 33], # A10 1150
    "S10" : [5760, 75]  # S10 2800
}

def roofline(is_double_precision, kernel_variation, size, hardware, number_ops, exec_time, number_bytes, fmax):
        plt.figure()
        
        plt.title(kernel_variation + " on " + hardware, ", " + size + " array") 
        plt.xlabel("FLOP/B") 
        plt.ylabel("GFLOPS") 
        
        [DSPs, mem_bandwidth] = hardware_params[hardware]

        # Single precision: 1 MAD is done by 1 DSP.
        # Double precision: 1 MAD is done by 4 DSPs (according to synthesis results in OpenCL. TODO: verify if this holds in SYCL)
        # So FLOPS per DSP: 2/1 for single precision, 2/4 for double precision
        double compute_roof = (is_double_precision == 1 ? 0.5 * DSPs * fmax : 2 * DSPs * fmax); 

        y0=compute_roof*0.001
        x0=y0/mem_bandwidth

        x = np.arange(0,5*x0) 
        y = mem_bandwidth*x
        plt.plot(x,y,ls="--",c="cornflowerblue") 
        plt.axhline(y=y0,ls="--",c="orange")

        x1=number_ops/number_bytes
        y1=number_ops/exec_time
        plt.scatter([x1],[y1],s=300,marker="^")

        font={'weight':'normal',
              'color':'black',
              'size':8
        }
        plt.text(2*x0,1.1*y0,"y(GFLOPS) = %g"%(y0),fontdict=font)
        plt.text(2*x0,1.9*y0,"y(GFLOPS) = %g(GB/S) * x(FLOP/B)"%(mem_bandwidth),fontdict=font)
        plt.text(1.1*x1,y1,"(%g,%g)"%(x1,y1),fontdict=font)

        plt.xlim((0, max(5*x0, 1.5*x1)))
        plt.ylim((0, 2*y0))
        plt.grid(alpha=0.4)

        plt.show(block=False)
        plt.savefig('roofline.png')

if __name__=="__main__":
    roofline(is_double_precision=sys.argv[1], kernel_variation=sys.argv[2], size=sys.argv[3], hardware=sys.argv[4], number_ops=float(argv[5]), exec_time=float(argv[6]), number_bytes=float(argv[7]), fmax=float(argv[8]))
