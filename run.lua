require 'image';
require 'xlua';
require 'optim';

config=5
color = false;
cuda = true

-- preprocess
dofile('preprocess.lua')

-- configure
dofile('configure.lua')
print(model:__tostring())

-- setup training
trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = learningRate
trainer.learningRateDecay = 1e-7
trainer.momentum = .9
trainer.maxIteration = maxIt
trainer.shuffleIndices = true
trainer.verbose = true

-- now run training
trainer:train(trainData)

-- print recognition accuracy
ra = utils.recognition_accuracy(model,testData)
print('Recognition accuracy on the test set ' .. ra .. ' % ')

-- print reconition accuracy via matching
ra = utils.matching_accuracy(model,model:size()-2,trainData,testData)
print('Matching accuracy on the test set ' .. ra .. ' % ')

--
ra = utils.matching_accuracy(model,model:size()-5,trainData,testData)
print('Matching accuracy on the test set with last conv layer ' .. ra .. ' % ')
