torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(123)
-- the only initial experiments were done with gray images 

if color then
  cdims = 3
else
  cdims = 1  
end

-- modifs no ReLU in the FC layers and normalization of the subtractive kernel

kernel_subtractive = image.gaussian(5);
kernel_subtractive = torch.div(kernel_subtractive,torch.sum(kernel_subtractive))

--kernel_divisive = image.gaussian(7);
-- 5 seems best for now
-- 7 seems a bit less
kernel_divisive = torch.Tensor(5,5):fill(1);
-- 3,3 is worse
-- 5,5 is best for now
-- 7,7 ok (comparable to 7,7
  
model = nil
if config == 1 then
-- configuration is the most vanilla with spatial contrastive normalization and the non-linearities saturated tanh
  print('Configuration 1')
  
  nfeats = {100,200,100}
  filtsize = 5
  nclasses = 43
  model = nn.Sequential()
  
  -- first block : convolutional
  model:add(nn.SpatialContrastiveNormalization(cdims,image.gaussian(7)))
  model:add(nn.SpatialConvolutionMM(cdims,nfeats[1],filtsize,filtsize,1,1))
  model:add(nn.Tanh())
  model:add(nn.ReLU())
  -- model:add(nn.SpatialContrastiveNormalization())
  model:add(nn.SpatialMaxPooling(2,2,2,2))
  
  -- second block : convolutional
  model:add(nn.SpatialConvolutionMM(nfeats[1],nfeats[2],filtsize,filtsize,1,1))
  model:add(nn.Tanh())
  model:add(nn.ReLU())
  -- model:add(nn.SpatialContrastiveNormalization())
  model:add(nn.SpatialMaxPooling(2,2,2,2))
  
  -- third block : standard 2-layer neural network
  model:add(nn.Reshape(nfeats[2]*filtsize*filtsize))
  model:add(nn.Linear(nfeats[2]*filtsize*filtsize,nfeats[3]))
  model:add(nn.Tanh())
  model:add(nn.ReLU())
  model:add(nn.Linear(nfeats[3], nclasses))
  model:add(nn.Tanh())
  model:add(nn.ReLU())
  model:add(nn.Reshape(nclasses))
  model:add(nn.LogSoftMax())
  
  criterion = nn.ClassNLLCriterion()
  learningRate = .1
  maxIt = 5;
elseif config == 2 then
-- configuration adds some subtracitve and divisive normalizations at the bottom
--FIXME this is not working yet
  print('Configuration 2')
  
  nfeats = {100,200,200}
  filtsize = 5
  nclasses = 43
  model = nn.Sequential()
  
  -- first block : convolutional
  model:add(nn.SpatialContrastiveNormalization(cdims,image.gaussian(5)))
  model:add(nn.SpatialConvolutionMM(cdims,nfeats[1],filtsize,filtsize,1,1))
  model:add(nn.Tanh())
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2,2,2,2))
  
  -- second block : convolutional
  model:add(nn.SpatialConvolutionMM(nfeats[1],nfeats[2],filtsize,filtsize,1,1))
  model:add(nn.Tanh())
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2,2,2,2))
  
  -- third block : standard 2-layer neural network
  model:add(nn.Reshape(nfeats[2]*filtsize*filtsize))
  model:add(nn.Linear(nfeats[2]*filtsize*filtsize,nfeats[3]))
  model:add(nn.Tanh())
  model:add(nn.ReLU())
  model:add(nn.Linear(nfeats[3], nclasses))
  model:add(nn.Tanh())
  model:add(nn.ReLU())
  model:add(nn.Reshape(nclasses))
  model:add(nn.LogSoftMax())
  
  criterion = nn.ClassNLLCriterion()
  learningRate = .001
  maxIt = 5;
elseif config == 3 then
-- configuration uses saturated tanh + subtractive and divisive normalizations in all conv layers
-- learning rate .01, iterations 5, FC 100 dims -- test 84% (learning rate seems optimal) 
-- learning rate .01, 15 iterations, FC 200 dims-- test 71%
-- learning rate .01, 5 iterations, FC 100 dims (gaussian sdn)-- test 71%
-- learning rate .01, 5 iterations, FC 200 dims (gaussian sdn)-- test 76.8%
-- learning rate .01, iterations 15, FC 100 dims -- test 86% (learning rate seems close to optimal) 
-- learning rate .009, iterations 5, FC 100 dims -- test 88%
-- learning rate .011, iterations 5, FC 100 dims -- test 86%
-- learning rate .008, iterations 5, FC 100 dims -- test 85%
-- learning rate .01, iterations 15, FC  50 dims -- test 48% color 
-- normalized subtractive kernel and removed ReLU from MLP
-- lr = .01, 15 iterations, FC100, test 95% Black and white (BW)
  print('Configuration 3')
   
  nfeats = {100,200,100}
  filtsize = 5
  nclasses = 43
  model = nn.Sequential()
  
  -- first block : convolutional
  model:add(nn.SpatialContrastiveNormalization(cdims,image.gaussian(5)))
  model:add(nn.SpatialConvolutionMM(cdims,nfeats[1],filtsize,filtsize,1,1))
  model:add(nn.Tanh())
  model:add(nn.ReLU())
  model:add(nn.SpatialSubtractiveNormalization(nfeats[1],kernel_subtractive))
  model:add(nn.SpatialDivisiveNormalization(nfeats[1],kernel_divisive))
  model:add(nn.SpatialMaxPooling(2,2,2,2))
  
  -- second block : convolutional
  model:add(nn.SpatialConvolutionMM(nfeats[1],nfeats[2],filtsize,filtsize,1,1))
  model:add(nn.Tanh())
  model:add(nn.ReLU())
  model:add(nn.SpatialSubtractiveNormalization(nfeats[2],kernel_subtractive))
  model:add(nn.SpatialDivisiveNormalization(nfeats[2],kernel_divisive))
  model:add(nn.SpatialMaxPooling(2,2,2,2))
  
  -- third block : standard 2-layer neural network
  model:add(nn.Reshape(nfeats[2]*filtsize*filtsize))
  model:add(nn.Linear(nfeats[2]*filtsize*filtsize,nfeats[3]))
  model:add(nn.Tanh())
  --model:add(nn.ReLU())
  model:add(nn.Linear(nfeats[3], nclasses))
  model:add(nn.Tanh())
  --model:add(nn.ReLU())
  model:add(nn.Reshape(nclasses))
  --model:add(nn.LogSoftMax())
  
  criterion = nn.CrossEntropyCriterion()
  --criterion = nn.ClassNLLCriterion()
  learningRate = .009
  maxIt = 15;
elseif config == 6 then
-- configuration is 3 with 40+64 in the conv layers
-- 200 FC dims -- test accuracy 86%
--  50 FC dims -- test accuracy 78%
-- 100 FC dims -- test accuracy 86%
-- 100 FC dims color -- test accuracy 78%
-- normalized subtractive kernel and removed ReLU from MLP
-- lr = .01, 5 iterations, FC100, test 96% Black and white (BW)

  print('Configuration 6')
   
  nfeats = {40,64,100}
  filtsize = 5
  nclasses = 43
  model = nn.Sequential()
  
  -- first block : convolutional
  model:add(nn.SpatialContrastiveNormalization(cdims,image.gaussian(5)))
  model:add(nn.SpatialConvolutionMM(cdims,nfeats[1],filtsize,filtsize,1,1))
  model:add(nn.Tanh())
  model:add(nn.ReLU())
  model:add(nn.SpatialSubtractiveNormalization(nfeats[1],kernel_subtractive))
  model:add(nn.SpatialDivisiveNormalization(nfeats[1],kernel_divisive))
  model:add(nn.SpatialMaxPooling(2,2,2,2))
  
  -- second block : convolutional
  model:add(nn.SpatialConvolutionMM(nfeats[1],nfeats[2],filtsize,filtsize,1,1))
  model:add(nn.Tanh())
  model:add(nn.ReLU())
  model:add(nn.SpatialSubtractiveNormalization(nfeats[2],kernel_subtractive))
  model:add(nn.SpatialDivisiveNormalization(nfeats[2],kernel_divisive))
  model:add(nn.SpatialMaxPooling(2,2,2,2))
  
  -- third block : standard 2-layer neural network
  model:add(nn.Reshape(nfeats[2]*filtsize*filtsize))
  model:add(nn.Linear(nfeats[2]*filtsize*filtsize,nfeats[3]))
  model:add(nn.Tanh())
  --model:add(nn.ReLU())
  model:add(nn.Linear(nfeats[3], nclasses))
  model:add(nn.Tanh())
  --model:add(nn.ReLU())
  model:add(nn.Reshape(nclasses))
  --model:add(nn.LogSoftMax())
  criterion = nn.CrossEntropyCriterion()
  --criterion = nn.ClassNLLCriterion()
  learningRate = .009
  maxIt = 15;
elseif config == 4 then
-- has two hidden layers slightly larger. the bottom is the same as config 3
-- with FC 200dim lr = .01, maxIt=5, test accuracy is 56%
-- with FC 100dim lr = .01, maxIt=5, test accuracy is 63%
-- with FC 100dim lr = .001, maxIt=5, test accuracy is 72%
-- with FC 200dim lr = .001, maxIt=5, test accuracy is 71%
-- with FC 200dim lr = .001, maxIt=5, gaussian spatialdivisivenormalization test accuracy is 77%
-- with FC 50dim lr = .001, maxIt=5, gaussian spatialdivisivenormalization test accuracy is 81%
-- with FC 100 test acc 71
  print('Configuration 4')
  
  nfeats = {100,200,200,200}
  filtsize = 5
  nclasses = 43
  model = nn.Sequential()
  
  -- first block : convolutional
  model:add(nn.SpatialContrastiveNormalization(cdims,image.gaussian(5)))
  model:add(nn.SpatialConvolutionMM(cdims,nfeats[1],filtsize,filtsize,1,1))
  model:add(nn.Tanh())
  model:add(nn.ReLU())
  model:add(nn.SpatialSubtractiveNormalization(nfeats[1],kernel_subtractive))
  model:add(nn.SpatialDivisiveNormalization(nfeats[1],kernel_divisive))  
  model:add(nn.SpatialMaxPooling(2,2,2,2))
  
  -- second block : convolutional
  model:add(nn.SpatialConvolutionMM(nfeats[1],nfeats[2],filtsize,filtsize,1,1))
  model:add(nn.Tanh())
  model:add(nn.ReLU())
  model:add(nn.SpatialSubtractiveNormalization(nfeats[2],kernel_subtractive))
  model:add(nn.SpatialDivisiveNormalization(nfeats[2],kernel_divisive))  
  model:add(nn.SpatialMaxPooling(2,2,2,2))
  
  -- third block : standard 2-layer neural network
  model:add(nn.Reshape(nfeats[2]*filtsize*filtsize))
  model:add(nn.Linear(nfeats[2]*filtsize*filtsize,nfeats[3]))
  model:add(nn.Tanh())
  model:add(nn.ReLU())
  model:add(nn.Linear(nfeats[3], nfeats[4]))
  model:add(nn.Tanh())
  model:add(nn.ReLU())
  model:add(nn.Linear(nfeats[4], nclasses))
  model:add(nn.Tanh())
  model:add(nn.ReLU())
  model:add(nn.Reshape(nclasses))
  model:add(nn.LogSoftMax())
  
  criterion = nn.ClassNLLCriterion()
  learningRate = .001
  maxIt = 5;
elseif config == 5 then
-- configuration uses saturated tanh + subtractive and divisive normalizations in all conv layers
-- learning rate .01, iterations 5 -- test 84% 
  require 'nngraph'
  
  print('Configuration 5')
   
  nfeats = {100,200,100}
  filtsize = 5
  nclasses = 43
  model = nn.Sequential()
  
  -- first block : convolutional
  model:add(nn.SpatialContrastiveNormalization(cdims,image.gaussian(5)))
  model:add(nn.SpatialConvolutionMM(cdims,nfeats[1],filtsize,filtsize,1,1))
  model:add(nn.Tanh())
  model:add(nn.ReLU())
  model:add(nn.SpatialSubtractiveNormalization(nfeats[1],kernel_subtractive))
  model:add(nn.SpatialDivisiveNormalization(nfeats[1],kernel_divisive))
  moduleIn1 = nn.SpatialMaxPooling(2,2,2,2)
  model:add(moduleIn1)
  
  -- second block : convolutional
  model:add(nn.SpatialConvolutionMM(nfeats[1],nfeats[2],filtsize,filtsize,1,1))
  model:add(nn.Tanh())
  model:add(nn.ReLU())
  model:add(nn.SpatialSubtractiveNormalization(nfeats[2],kernel_subtractive))
  model:add(nn.SpatialDivisiveNormalization(nfeats[2],kernel_divisive))
  moduleIn2 = nn.SpatialMaxPooling(2,2,2,2)
  
  -- third block : standard 2-layer neural network
  moduleOut = nn.Reshape(nfeats[2]*filtsize*filtsize+nfeats[1]*14*14);
  model:add(nn.gModule({moduleIn1,moduleIn2},{moduleOut}))
  model:add(nn.Linear(nfeats[2]*filtsize*filtsize,nfeats[3]))
  model:add(nn.Tanh())
  model:add(nn.ReLU())
  model:add(nn.Linear(nfeats[3], nclasses))
  model:add(nn.Tanh())
  model:add(nn.ReLU())
  model:add(nn.Reshape(nclasses))
  model:add(nn.LogSoftMax())
  
  criterion = nn.ClassNLLCriterion()
  learningRate = .01
  maxIt = 5;
elseif config == 9 then
  print('Configuration 9')
  model = nn.Sequential()

  model:add(nn.SpatialContrastiveNormalization(cdims,image.gaussian(5)))
  
elseif config == 10 then
-- alexnet'ish
  print('Configuration 10')

  model = nn.Sequential()
  -- convolution layers
  model:add(nn.SpatialConvolutionMM(3, 128, 5, 5, 1, 1))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  model:add(nn.SpatialConvolutionMM(128, 256, 5, 5, 1, 1))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  --model:add(nn.SpatialZeroPadding(1, 1, 1, 1))
  model:add(nn.SpatialConvolutionMM(256, 512, 4, 4, 1, 1))
  model:add(nn.ReLU())
  -- fully connected layers
  model:add(nn.SpatialConvolutionMM(512, 1024, 2, 2, 1, 1))
  model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))
  model:add(nn.SpatialConvolutionMM(1024, 10, 1, 1, 1, 1))
  model:add(nn.Reshape(10))
  model:add(nn.SoftMax())
  
  criterion = nn.ClassNLLCriterion()
else
  print('bad!')
end
  
if cuda == true then
  print('cuda')
  require 'cunn';
  require 'cutorch';
  trainData.data = trainData.data:cuda()
  trainData.label = trainData.label:cuda()
  testData.data = testData.data:cuda()
  testData.label = testData.label:cuda()
  criterion = criterion:cuda()
  model = model:cuda()
else
  print('float')
  require 'nn';
  require 'torch';
  trainData.data = trainData.data:float()
  trainData.label = trainData.label:float()
  testData.data = testData.data:float()
  testData.label = testData.label:float()
  criterion = criterion:float()
  model = model:float()
end

--print(model)
--print(criterion)