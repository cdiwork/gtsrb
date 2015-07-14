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

return utils