Pre-requisites
---
lua 5.1
torch 7
itorch 2.1

the only exotic luarocks package is csv for reading the file names

Running
---
Running the gtsrb.ipynb notebook with itorch should do it all to measuring error. It should download the data, generate the t7 files (options for YUV and Y only by setting the variable color). There are a few different configurations I looked at they them and the results are sketched in the configure.lua. The interesting ones are config=3 and config=6. The results should be around 97 with these configs and color=false. 

Note that since I didn't have an IDE I played around directly in the notebook so there are some duplicate calls to the scripts there shouldn't be any problem.

The approach is pretty straightforward and as close as I could make it to #27 from the paper. This means 2 conv layers with abs(tanh(x)) non-linearities and both additive and divisive normalization. Then the MLP with tanh non-linearities. The parameters are very similar to the ones in the paper. The "matching" mode is 1-NN on the last layer (layer 18) or the vectorized response of the last convolutional layer (after the non-linearities). The result for matching is around 86%.

Added the "multi-scale" architecture (conf=5) via Replicate, SplitTable, ParallelTable then JoinTable. The parameter tuning is still ongoing but recognition results are already in the 90%.

I started looking a bit at the skip layer architecture or the distance thing you use. I'll try to have a look when I have some time again.

