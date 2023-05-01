#TODO: auto retrive libHalide.a and include files from a website
cp /home/u89062/a10-multi-ure-groups/Halide/lib/libHalide.a Halide/lib
cp /home/u89062/a10-multi-ure-groups/Halide/include/Halide.h Halide/include
cp /home/u89062/a10-multi-ure-groups/Halide/include/HalideBuffer.h Halide/include
cp /home/u89062/a10-multi-ure-groups/Halide/include/HalideRuntime.h Halide/include

cd Halide/lib
split -b 100M libHalide.a libHalide.part.
