require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator
csv = require 'csv';
utils = require 'utils';

torch.setdefaulttensortype('torch.FloatTensor')

-- if we already preprocess the files just load them up
if (paths.filep('gtsrb_32x32_train.t7') and not color) or (paths.filep('gtsrb_32x32_color_train.t7') and color) then 
  print('Loading data!')
  --trainData = torch.load('data/cifar10-train.t7')
  --trainData.data = trainData.data:float()
  --trainData.label = trainData.label:float()
  --setmetatable(trainData, {__index = function(t, i) return {t.data[i], t.label[i]} end});
  --function trainData:size() return trainData.label:size(1) end
  if color then
    trainData = torch.load('gtsrb_32x32_color_train.t7')
    testData = torch.load('gtsrb_32x32_color_test.t7')
  else
    trainData = torch.load('gtsrb_32x32_train.t7')
    testData = torch.load('gtsrb_32x32_test.t7')
  end 
  
  trainData.data = trainData.data:float()
  trainData.label = trainData.label:float()
  trainData.label:add(1);
  setmetatable(trainData, {__index = function(t, i) return {t.data[i], t.label[i]} end});
  function trainData:size() return trainData.label:size(1) end
  
  testData.data = testData.data:float()
  testData.label = testData.label:float()
  testData.label:add(1);
  setmetatable(testData, {__index = function(t, i) return {t.data[i], t.label[i]} end});
  function testData:size() return testData.label:size(1) end
else
  print('Preparing data!')
  -- FIXME add 1 to the labels
  -- FIXME verify that the data initialization is correct
  -- FIXME add local contrast normalization
  -- FIXME add augmentation

  -- first download the data
  if not paths.filep('GTSRB_Final_Training_Images.zip') then
    os.execute('wget http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip')
    os.execute('unzip GTSRB_Final_Training_Images.zip')
    os.execute('wget http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip')
    os.execute('unzip GTSRB_Final_Test_Images.zip')
    os.execute('mv GT-final_test.csv GTSRB')
  end
  
  if color then 
    cdims = 3
  else
    cdims = 1
  end
    
  -- process the training data
  trainSize = 39210
  trainData = {
    data = torch.FloatTensor(trainSize,cdims,32,32):zero(),
    label = torch.FloatTensor(trainSize):zero(),
  }
  function trainData:size() return trainData.label:size(1) end
  setmetatable(trainData, {__index = function(t, i) return {t.data[i], t.label[i]} end});
  
  count = 1
  for class = 0,42 do
    -- read csv
    -- print(string.format('GTSRB/Final_Training/Images/%05d/GT-%05d.csv',class,class))
    f = csv.open(string.format("GTSRB/Final_Training/Images/%05d/GT-%05d.csv",class,class),{separator=";", header=true})
  
    for fields in f:lines() do
      if fields['Filename']=='' then break; end
      filename = string.format("GTSRB/Final_Training/Images/%05d/%s",class,fields['Filename']);
      --print(filename)
      im = image.load(filename,3,FloatTensor)
  
      -- rescale to 32x32, transform to yuv, keep only y, paste in to tensor
      ims = image.scale(im,32,32);
      yuvims = image.rgb2yuv(ims);
      
      trainData.data[{{count},{},{},{}}] = torch.reshape(yuvims[{{1,cdims},{},{}}],torch.LongStorage{cdims,32,32,1});
      trainData.label[count]=fields['ClassId'];
      count = count+1;
    end
  end
  -- save the test data
  print('Finished train data')
  
  -- process the training data
  testSize = 12630
  testData = {
    data = torch.FloatTensor(testSize,cdims,32,32):zero(),
    label = torch.FloatTensor(testSize):zero(),
    size = function() return testSize end
  }
  function testData:size() return testData.label:size(1) end
  setmetatable(testData, {__index = function(t, i) return {t.data[i], t.label[i]} end});
  
  count = 1
  f = csv.open(string.format("GTSRB/GT-final_test.csv",class,class),{separator=";", header=true})
  for fields in f:lines() do
    if fields['Filename']=='' then break; end
    filename = string.format("GTSRB/Final_Test/Images/%s",fields['Filename']);
    --print(filename)
    im = image.load(filename,3,FloatTensor)
  
    -- rescale to 32x32, transform to yuv, keep only y, paste in to tensor
    ims = image.scale(im,32,32);
    yuvims = image.rgb2yuv(ims);
        
    testData.data[{{count},{},{},{}}] = torch.reshape(yuvims[{{1,cdims},{},{}}],torch.LongStorage{cdims,32,32,1});
    testData.label[count]=fields['ClassId'];
  
    count = count+1;
  end
  
  -- whiten the data
  utils.whiten(trainData,testData);
  
  -- save the data
  if color then
    torch.save('gtsrb_32x32_color_train.t7',trainData)
    torch.save('gtsrb_32x32_color_test.t7',testData)
  else
    torch.save('gtsrb_32x32_train.t7',trainData)
    torch.save('gtsrb_32x32_test.t7',testData)
  end 
end

print('Data is ready for work!')
