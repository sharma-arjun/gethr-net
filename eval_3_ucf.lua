
--[[

This file trains a character-level multi-layer RNN on text data

Code is based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6
but modified to have multi-layer support, GPU support, as well as
many other common model/optimization bells and whistles.
The practical6 code is in turn based on 
https://github.com/wojciechz/learning_to_execute
which is turn based on other stuff in Torch, etc... (long lineage)

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'
local matio = require 'matio'
local CharSplitLMMinibatchLoader = require 'util.CharSplitLMMinibatchLoader'
local model_utils = require 'util.model_utils'
local LSTM3 = require 'model.LSTM3'

input_size_vid = 4096
input_size_aud = 4096
input_size = input_size_vid + input_size_aud
combination_size = 2048
output_size = 101
max_frames = 0

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','data/tinyshakespeare','data directory. Should contain the file input.txt with input data')
-- model params
cmd:option('-rnn_size', 128, 'size of LSTM internal state')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-model', 'lstm', 'for now only lstm is supported. keep fixed')
-- optimization
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-seq_length',50,'number of timesteps to unroll for')
cmd:option('-batch_size',50,'number of sequences to train on in parallel')
cmd:option('-max_epochs',50,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-train_frac',0.95,'fraction of data that goes into train set')
cmd:option('-val_frac',0.05,'fraction of data that goes into validation set')
            -- test_frac will be computed as (1 - train_frac - val_frac)
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',1000,'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','lstm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
-- GPU/CPU
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
opt.batch_size = 1 -- added by arjun

-- train / val / test split for data, in fractions
local test_frac = math.max(0, 1 - (opt.train_frac + opt.val_frac))
local split_sizes = {opt.train_frac, opt.val_frac, test_frac} 

-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- initialize clnn/cltorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 1 then
    local ok, cunn = pcall(require, 'clnn')
    local ok2, cutorch = pcall(require, 'cltorch')
    if not ok then print('package clnn not found!') end
    if not ok2 then print('package cltorch not found!') end
    if ok and ok2 then
        print('using OpenCL on GPU ' .. opt.gpuid .. '...')
        cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        torch.manualSeed(opt.seed)
    else
        print('If cltorch and clnn are installed, your OpenCL driver may be improperly configured.')
        print('Check your OpenCL driver installation, check output of clinfo command, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- create the data loader class
local loader = CharSplitLMMinibatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes)
local vocab_size = loader.vocab_size  -- the number of distinct characters
local vocab = loader.vocab_mapping
--print('vocab size: ' .. vocab_size)
-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

-- define the model: prototypes for one timestep, then clone them in time
local do_random_init = true
if string.len(opt.init_from) > 0 then
    print('loading an LSTM from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    protos = checkpoint.protos
    -- make sure the vocabs are the same
    local vocab_compatible = true
    for c,i in pairs(checkpoint.vocab) do 
        if not vocab[c] == i then 
            vocab_compatible = false
        end
    end
    assert(vocab_compatible, 'error, the character vocabulary for this dataset and the one in the saved checkpoint are not the same. This is trouble.')
    -- overwrite model settings based on checkpoint to ensure compatibility
    print('overwriting rnn_size=' .. checkpoint.opt.rnn_size .. ', num_layers=' .. checkpoint.opt.num_layers .. ' based on the checkpoint.')
    opt.rnn_size = checkpoint.opt.rnn_size
    opt.num_layers = checkpoint.opt.num_layers
    do_random_init = false
else
    print('creating an LSTM3 with ' .. opt.num_layers .. ' layers')
    protos = {}
    protos.rnn = LSTM3.lstm(input_size, combination_size, output_size, opt.rnn_size, opt.num_layers, opt.dropout)
    protos.criterion = nn.ClassNLLCriterion()
    --protos.criterion = nn.BCECriterion()
end

-- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
    if opt.gpuid >=0 and opt.opencl == 0 then h_init = h_init:cuda() end
    if opt.gpuid >=0 and opt.opencl == 1 then h_init = h_init:cl() end
    table.insert(init_state, h_init:clone())
    table.insert(init_state, h_init:clone())
end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 and opt.opencl == 0 then
    for k,v in pairs(protos) do v:cuda() end
end
if opt.gpuid >= 0 and opt.opencl == 1 then
    for k,v in pairs(protos) do v:cl() end
end

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn) -- check what changes are to be made to this function.

-- initialization
if do_random_init then
    params:uniform(-0.08, 0.08) -- small numbers uniform

    if opt.model == 'lstm' then
        for layer_idx = 1, opt.num_layers do
            for _,node in ipairs(protos.rnn.forwardnodes) do
                if node.data.annotations.name == "i2h_" .. layer_idx then
                    print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
                    -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
                    node.data.module.bias[{{opt.rnn_size+1, 2*opt.rnn_size}}]:fill(1.0)
                end
            end
        end
    end
end

print('number of parameters in the model: ' .. params:nElement())
-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
--for name,proto in pairs(protos) do
--    print('cloning ' .. name)
--    clones[name] = model_utils.clone_many_times(proto, 1057, not proto.parameters)
--    --clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
--end
-- load training/val/test data (written by arjun)

--h = 1
--feature_folder = 'new_audio_features_user6/win10/'
phase = 2
split_no = 2
function loaddata()

    local training_features = {}
    local training_labels = {}
    local validation_features = {}
    local validation_labels = {}
    if phase == 1 then
        --local f1 = io.open("/data/arjun/adl/activity/train/trainfiles.txt","r")
        local f1_v1 = io.open("/home/arjun/public_speaking/trainfiles0" .. tostring(split_no) .. ".txt","r")
        local f1_lb = io.open("/home/arjun/public_speaking/trainlabels0" .. tostring(split_no) .. ".txt","r")

        --local training_ids = {}
        --weight_line = f1:read("*line")
        local count = 0
        print('loading training set ...')

        while true do
            local filename = f1_v1:read("*line")
            if filename == nil then break end

            local data_m1 = matio.load('/data/arjun/ucf_features/split' .. tostring(split_no) .. '/motion/' .. filename)
            local data_m2 = matio.load('/data/arjun/ucf_features/split' .. tostring(split_no) .. '/appearance/' .. filename)

            local raw_feat_m1 = data_m1['features']
            local raw_feat_m2 = data_m2['features']
            raw_feat_m1 = raw_feat_m1:t()
            raw_feat_m2 = raw_feat_m2:t()
            local label = f1_lb:read("*number")
            --if raw_feat_m1:size(1) ~= raw_feat_m2:size(1) then print(tostring(count) .. 'WTF!!' .. tostring(raw_feat_m1:size(1)) .. ' ' .. tostring(raw_feat_m2:size(1)) ) end
            n_frames = math.min(raw_feat_m1:size(1),raw_feat_m2:size(1))/10
            local feat = torch.Tensor(n_frames, input_size_vid + input_size_aud):zero()
            for i=1,n_frames*10 do
                feat[{(i%n_frames)+1,{1,input_size_vid}}] = feat[{(i%n_frames)+1,{1,input_size_vid}}] + raw_feat_m1[{i,{}}]
                feat[{(i%n_frames)+1,{input_size_vid+1,-1}}] = feat[{(i%n_frames)+1,{input_size_vid+1,-1}}] + raw_feat_m2[{i,{}}]
            end
            feat:div(10)

            if n_frames > max_frames then max_frames = n_frames end
            local labels = torch.Tensor(n_frames,1)

            for i=1,n_frames do
                labels[i] = label
            end

            table.insert(training_features,feat)
            table.insert(training_labels,labels)
            --table.insert(training_ids,id)
            count = count + 1
            if count % 2000 == 0 then print(count) end
        end
    else
        --local f3 = io.open("/data/arjun/adl/activity/val/valfiles.txt","r")
        local f2_v1 = io.open("/home/arjun/public_speaking/testfiles0" .. tostring(split_no) .. ".txt","r")
        local f2_lb = io.open("/home/arjun/public_speaking/testlabels0" .. tostring(split_no) .. ".txt","r")
        --local validation_ids = {}

        local count = 0
        print('loading validation set ...')

        while true do
            filename = f2_v1:read("*line")
            if filename == nil then break end

            local data_m1 = matio.load('/data/arjun/ucf_features/split' .. tostring(split_no) .. '/motion/' .. filename)
            local data_m2 = matio.load('/data/arjun/ucf_features/split' .. tostring(split_no) .. '/appearance/' .. filename)

            local raw_feat_m1 = data_m1['features']
            local raw_feat_m2 = data_m2['features']
            raw_feat_m1 = raw_feat_m1:t()
            raw_feat_m2 = raw_feat_m2:t()
            local label = f2_lb:read("*number")

            n_frames = math.min(raw_feat_m1:size(1),raw_feat_m2:size(1))/10
            local val_feat = torch.Tensor(n_frames, input_size_vid + input_size_aud):zero()
            for i=1,n_frames*10 do
                val_feat[{(i%n_frames)+1,{1,input_size_vid}}] = val_feat[{(i%n_frames)+1,{1,input_size_vid}}] + raw_feat_m1[{i,{}}]
                val_feat[{(i%n_frames)+1,{input_size_vid+1,-1}}] = val_feat[{(i%n_frames)+1,{input_size_vid+1,-1}}] + raw_feat_m2[{i,{}}]
            end
            val_feat:div(10)

            if n_frames > max_frames then max_frames = n_frames end
            local val_labels = torch.Tensor(n_frames,1)

            for i=1,n_frames do
                val_labels[i] = label
            end
            table.insert(validation_features,val_feat)
            table.insert(validation_labels,val_labels)
            --table.insert(validation_ids,id)
            count = count + 1
            if count % 1000 == 0 then print(count) end
        
        end
    end
    return training_features,training_labels,validation_features,validation_labels
end
train_x, train_y, val_x, val_y = loaddata()
vid_idx = {1,1}

print('Training set size ' .. tostring(#train_x))
print('Test set size ' .. tostring(#val_x))
print('Max frames ' .. tostring(max_frames))

for name,proto in pairs(protos) do
    print('cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto, max_frames, not proto.parameters) -- check clone_many_times function too.
    --clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end

function fetch_video(split_index)
    if split_index == 1 then
        x = train_x[vid_idx[1]]
        y = train_y[vid_idx[1]]
        --id = train_ids[vid_idx[1]]
        if vid_idx[1] == #train_x then vid_idx[1] = 1 else vid_idx[1] = vid_idx[1] + 1 end
    elseif split_index == 2 then
        x = val_x[vid_idx[2]]
        y = val_y[vid_idx[2]]
        --id = val_ids[vid_idx[2]]
        if vid_idx[2] == #val_x then vid_idx[2] = 1 else vid_idx[2] = vid_idx[2] + 1 end
    end

    --return x,y,id
    return x,y
end

-- evaluate the loss over an entire split
function eval_split(split_index)
    print('evaluating loss over split index ' .. split_index)
    --local n = loader.split_sizes[split_index]
    --if max_batches ~= nil then n = math.min(max_batches, n) end

    --loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    if split_index == 1 then n = #train_x else n = #val_x end
    local acc = 0 

    local loss = 0
    local rnn_state = {[0] = init_state}

    --total_overall_predictions = {}
    --total_actual_labels = {}
    --total_ids = {}
    all_predictions = torch.Tensor(n,output_size)
    for i = 1,n do -- iterate over batches in the split
        -- fetch a batch
        local x, y = fetch_video(split_index)
        if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
            -- have to convert to float because integers can't be cuda()'d
            x = x:cuda()
            y = y:cuda()
        end
        if opt.gpuid >= 0 and opt.opencl == 1 then -- ship the input arrays to GPU
            x = x:cl()
            y = y:cl()
        end
        --table.insert(total_ids,id)
        n_frames = x:size(1)
        -- forward pass
        for t=1,n_frames do
            --print(t)
            clones.rnn[t]:evaluate() -- for dropout proper functioning
            inp = torch.CudaTensor(1,input_size)
            inp[{1,{}}] = x[{t,{}}]
            local lst = clones.rnn[t]:forward{inp, unpack(rnn_state[t-1])}
            rnn_state[t] = {}
            for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end
            prediction = lst[#lst]

            if t == n_frames then
                all_predictions[{i,{}}] = prediction:double()  
            --    max_pred = -1000
            --    max_class = -1
            --    for cl=1,output_size do 
            --        if prediction[1][cl] > max_pred then 
            --            max_class = cl
            --            max_pred = prediction[1][cl]
            --        end
            --    end
            --    loss = loss + clones.criterion[t]:forward(prediction, y[{t,{}}])
            end
        end
        -- carry over lstm state
        --rnn_state[0] = rnn_state[#rnn_state]
        rnn_state = {[0] = init_state}
        --overall_prediction = all_predictions[n_frames]
        --if max_class == y[{1,1}] then
        --    acc = acc + 1
        --end
        
        --overall_prediction = 0
        --for j=1,n_frames do
        --    if all_predictions[j] == 1 then
        --        overall_prediction = 1
        --        break
        --    end
        --end
        --print(i .. '/' .. n .. '...')
        --print(overall_prediction)
        --print(y[{1,1}])
        --table.insert(total_overall_predictions,overall_prediction)
        --table.insert(total_actual_labels,y[{1,1}])
        --if overall_prediction == y[{1,1}] then acc = acc + 1 end
        --if y[{1,1}] == 1 then pos = pos + 1 else neg = neg + 1 end
        --if overall_prediction == 1 and y[{1,1}] == 1 then tpos = tpos + 1 end
        --if overall_prediction == 0 and y[{1,1}] == 0 then tneg = tneg + 1 end
    end

    --loss = loss / n
    --acc = acc / n
    
    --unique = -1
    --current_label = 0
    --current_pred = 0
    --current_id = -1

    --for i=1,n do
    --  id = total_ids[i]
    --  current_label = total_actual_labels[i]
    --  current_pred = total_overall_predictions[i]

    --  if id ~= current_id then
    --    unique = unique + 1
    --    if current_id ~= -1 then
    --      if pos_pred >= neg_pred then overall_pred = 1 else overall_pred = 0 end
    --      if overall_pred == overall_label then acc = acc + 1 end
    --      if overall_label == 1 then pos = pos + 1 else neg = neg + 1 end
    --      if overall_pred == 1 and overall_label == 1 then tpos = tpos + 1 end
    --      if overall_pred == 0 and overall_label == 0 then tneg = tneg + 1 end
    --    end
    --    overall_label = current_label
    --    overall_pred = 0
    --    current_id = id
    --    pos_pred = 0 -- comment this out for even one positive
    --    neg_pred = 0 -- comment this out for even one positive
    --  end
      
      ----if current_pred == 1 then overall_pred = 1 end
      --if current_pred == 1 then pos_pred = pos_pred + 1 else neg_pred = neg_pred + 1 end -- comment this out for even one positive
    --end

    --unique = unique + 1
    --if pos_pred >= neg_pred then overall_pred = 1 else overall_pred = 0 end
    --if overall_pred == overall_label then acc = acc + 1 end
    --if overall_label == 1 then pos = pos + 1 else neg = neg + 1 end
    --if overall_pred == 1 and overall_label == 1 then tpos = tpos + 1 end
    --if overall_pred == 0 and overall_label == 0 then tneg = tneg + 1 end
        
    --acc = acc/unique
    --acc = acc/n
    --print(string.format("Validation accuracy %f",acc))
    return all_predictions
end

local all_predictoins = eval_split(phase) -- 2 = validation
matio.save('proposed5_490_490_490_768_1_1_1.mat',{t1=all_predictions})
