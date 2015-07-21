local utils = {}

-- normalize the data by whitening
function utils.whiten(trainData, testData)
  mean = {} -- store the mean, to normalize the test set in the future
  stdv  = {} -- store the standard-deviation for the future

  for i = 1,trainData.data:size(2) do
    mean[i] = trainData.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainData.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = trainData.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainData.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
  end
  
  for i = 1,trainData.data:size(2) do
    testData.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    testData.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
  end
end

-- evaluate recognition accuracy
function utils.recognition_accuracy(model,dataset)
  local correct = 0;
  local predLabel = torch.Tensor(dataset:size()):zero()
  for i=1,dataset:size() do
      local groundtruth = dataset.label[i]
      local prediction = model:forward(dataset.data[i])
      local confidences, indices = torch.max(prediction,1)
      if groundtruth == indices[1] then
          correct = correct + 1
      end
      predLabel[i] = indices[1]
  end
  
  return 100*correct/dataset:size(), predLabel
end

-- evaluate matching accuracy
function utils.matching_accuracy(model,layer,trainSet,testSet)
  local correct = 0;
  
  model:forward(trainSet.data[1])
  feat = model.modules[layer].output;
  dims = feat:numel();
  
  -- compute features and their squared vector norms
  local features = torch.CudaTensor(trainSet:size(),feat:numel()):zero()
  local features_norm = torch.CudaTensor(trainSet:size()):zero()
  for i=1,trainSet:size() do
      model:forward(trainSet.data[i])
      features[i] = model.modules[layer].output;
      features_norm[i] = torch.norm(features[i])
      features_norm[i] = features_norm[i]*features_norm[i]
  end
  
  -- compute distances, ignore the same, take minimum and check if its label is correct 
  local pred = torch.CudaTensor(testSet:size()):zero()
  local predFeat = torch.CudaTensor(testSet:size(),feat:numel()):zero()
  for i = 1,testSet:size() do
      local groundtruth = testSet.label[i]
      model:forward(testSet.data[i])
      predFeat[i] = model.modules[layer].output;
      p = torch.mv(features,predFeat[i])
      p:add(-2,p)
      p = p+features_norm
      val,ii = torch.min(p,1);
      pred[i] = trainSet.label[{ii[1]}]
      if groundtruth == pred[i] then
          correct = correct + 1
      end
  end
  
  return 100*correct/testSet:size()
end

return utils