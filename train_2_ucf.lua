
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
local LSTM2 = require 'model.LSTM2'

input_size_vid = 4096
input_size_aud = 4096
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
cmd:option('-rnn_size_vid', 128, 'size of LSTM internal state for video')
cmd:option('-rnn_size_aud', 128, 'size of LSTM internal state for audio')
cmd:option('-rnn_size_comb', 128, 'size of LSTM internal state for combination')
cmd:option('-num_layers_vid', 2, 'number of layers in the LSTM for video')
cmd:option('-num_layers_aud', 2, 'number of layers in the LSTM for audio')
cmd:option('-num_layers_comb', 2, 'number of layers in the LSTM for combination')
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
    print('creating an LSTM2 with ' .. opt.num_layers_vid .. ' layers for video, ' .. opt.num_layers_aud .. ' layers for audio and ' .. opt.num_layers_comb .. ' layers for combination')
    protos = {}
    protos.rnn = LSTM2.lstm(input_size_vid, input_size_aud, output_size, opt.rnn_size_vid, opt.rnn_size_aud, opt.rnn_size_comb, opt.num_layers_vid, opt.num_layers_aud, opt.num_layers_comb, opt.dropout)
    protos.criterion = nn.ClassNLLCriterion()
    --protos.criterion = nn.BCECriterion()
end

-- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers_vid do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size_vid)
    if opt.gpuid >=0 and opt.opencl == 0 then h_init = h_init:cuda() end
    if opt.gpuid >=0 and opt.opencl == 1 then h_init = h_init:cl() end
    table.insert(init_state, h_init:clone())
    table.insert(init_state, h_init:clone())
end
for L=1,opt.num_layers_aud do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size_aud)
    if opt.gpuid >=0 and opt.opencl == 0 then h_init = h_init:cuda() end
    if opt.gpuid >=0 and opt.opencl == 1 then h_init = h_init:cl() end
    table.insert(init_state, h_init:clone())
    table.insert(init_state, h_init:clone())
end
for L=1,opt.num_layers_comb do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size_comb)
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
        for layer_idx = 1, opt.num_layers_vid + opt.num_layers_aud + opt.num_layers_comb do
            for _,node in ipairs(protos.rnn.forwardnodes) do
                if node.data.annotations.name == "i2h_v_" .. layer_idx then
                    print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
                    -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
                    node.data.module.bias[{{opt.rnn_size_vid+1, 2*opt.rnn_size_vid}}]:fill(1.0)
                end
                if node.data.annotations.name == "i2h_a_" .. layer_idx then
                    print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
                    -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
                    node.data.module.bias[{{opt.rnn_size_aud+1, 2*opt.rnn_size_aud}}]:fill(1.0)
                end
                if node.data.annotations.name == "i2h_c_" .. layer_idx then
                    print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
                    -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
                    node.data.module.bias[{{opt.rnn_size_comb+1, 2*opt.rnn_size_comb}}]:fill(1.0)
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

split_no = 3

function loaddata()

    --local f1 = io.open("/data/arjun/adl/activity/train/trainfiles.txt","r")
    local f1_v1 = io.open("/home/arjun/public_speaking/trainfiles0" .. tostring(split_no) .. ".txt","r")
    local f1_lb = io.open("/home/arjun/public_speaking/trainlabels0" .. tostring(split_no) .. ".txt","r")

    local training_features = {}
    local training_labels = {}
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

    --local f3 = io.open("/data/arjun/adl/activity/val/valfiles.txt","r")
    local f2_v1 = io.open("/home/arjun/public_speaking/testfiles0" .. tostring(split_no) .. ".txt","r")
    local f2_lb = io.open("/home/arjun/public_speaking/testlabels0" .. tostring(split_no) .. ".txt","r")
    local validation_features = {}
    local validation_labels = {}
    --local validation_ids = {}

    count = 0
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
        --if raw_feat_m1:size(1) ~= raw_feat_m2:size(1) then print('WTF!!') end
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
        --all_predictions = torch.IntTensor(n_frames)
        -- forward pass
        for t=1,n_frames do
            --print(t)
            clones.rnn[t]:evaluate() -- for dropout proper functioning
            inp = torch.CudaTensor(1,input_size_vid + input_size_aud)
            inp[{1,{}}] = x[{t,{}}]
            local lst = clones.rnn[t]:forward{inp[{{},{1,input_size_vid}}], inp[{{},{input_size_vid+1,-1}}], unpack(rnn_state[t-1])}
            rnn_state[t] = {}
            for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end
            prediction = lst[#lst]
            
            if t == n_frames then
 
                max_pred = -1000
                max_class = -1
                for cl=1,output_size do 
                    if prediction[1][cl] > max_pred then 
                        max_class = cl
                        max_pred = prediction[1][cl]
                    end
                end
                loss = loss + clones.criterion[t]:forward(prediction, y[{t,{}}])
            end
        end
        -- carry over lstm state
        --rnn_state[0] = rnn_state[#rnn_state]
        rnn_state = {[0] = init_state}
        --overall_prediction = all_predictions[n_frames]
        if max_class == y[{1,1}] then
            acc = acc + 1
        end
        
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

    loss = loss / n
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
    acc = acc/n
    print(string.format("Validation accuracy %f",acc))
    return loss,acc
end


-- do fwd/bwd and return loss, grad_params
local init_state_global = clone_list(init_state)
my_batch_size = 10
accumulated_grad_params = torch.CudaTensor(grad_params:size())

function feval(p)

    accumulated_grad_params:zero()
    --grad_params:zero()
    local loss = 0

    for n_runs=1,my_batch_size do
    if p ~= params then
        params:copy(p)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y = fetch_video(1)
    --print(x[{1,{}}])
    --print(y[{1,{}}])
    
    if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:cuda()
        y = y:cuda()
    end
    if opt.gpuid >= 0 and opt.opencl == 1 then -- ship the input arrays to GPU
        x = x:cl()
        y = y:cl()
    end
    ------------------- forward pass -------------------
    local rnn_state = {[0] = init_state_global}
    local predictions = {}           -- softmax outputs
    local loss_ex = 0 -- loss for this examples. Accumulate over all time steps and then divide by n_frames. Add this to loss for my batch
    n_frames = x:size(1)
    for t=1,n_frames do
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        inp = torch.CudaTensor(1,input_size_vid + input_size_aud):zero()
        inp[{1,{}}] = x[{t,{}}]
        --print(t)
        --print(inp)
        local lst = clones.rnn[t]:forward{inp[{{},{1,input_size_vid}}], inp[{{},{input_size_vid+1,-1}}], unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
        predictions[t] = lst[#lst] -- last element is the prediction
        if t == n_frames then
            loss_ex = loss_ex + clones.criterion[t]:forward(predictions[t], y[{t,{}}])
        end
    end
    --loss_ex = loss_ex / n_frames
    loss = loss + loss_ex

    prediction = predictions[n_frames]

    --max = -999
    --maxid = -1
    --for j=1,18 do
    --    if prediction[1][j] > max then
    --        max = prediction[1][j]
    --        maxid = j
    --    end
    --end
    --print(maxid)
    --print(y[{1,1}])

    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[n_frames] = clone_list(init_state, true)} -- true also zeros the clones
    for t=n_frames,n_frames,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[{t,{}}])
        table.insert(drnn_state[t], doutput_t)
        inp = torch.CudaTensor(1,input_size_vid + input_size_aud):zero()
        inp[{1,{}}] = x[{t,{}}]
        local dlst = clones.rnn[t]:backward({inp[{{},{1,input_size_vid}}], inp[{{},{input_size_vid+1,-1}}], unpack(rnn_state[t-1])}, drnn_state[t])
        drnn_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k > 2 then -- k == 1 and 2 is gradient on x_vid and x_aud, which we dont need
                -- note we do k-2 because first and second items are dembeddings, and then follow the 
                -- derivatives of the state, starting at index 3. I know...
                drnn_state[t-1][k-2] = v
            end
        end
    end
    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    --init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    --print(grad_params[1])
    accumulated_grad_params = accumulated_grad_params + grad_params
    --print(accumulated_grad_params[1])
    end --- end of for loop over batch
    loss = loss/my_batch_size
    grad_params:copy(accumulated_grad_params)
    --print(grad_params[1])
    grad_params = grad_params:div(my_batch_size)
    return loss, grad_params
end

-- start optimization here
train_losses = {}
val_losses = {}
val_accs = {}
--print('Training set size ' .. tostring(#train_x))
--print('Test set size ' .. tostring(#val_x))
--print('Max frames ' .. tostring(max_frames))

local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
--local optim_state = {learningRate = opt.learning_rate, momentum = 0.9, weightDecay = 0.005}
local iterations = opt.max_epochs * #train_x/my_batch_size
local iterations_per_epoch = #train_x/my_batch_size -- added to take care of my new batch size in feval
local loss0 = nil
local last_epoch = 0
for i = 1, iterations do
    local epoch = i / (#train_x/my_batch_size)

    local timer = torch.Timer()
    local _, loss = optim.rmsprop(feval, params, optim_state)
    --local _, loss = optim.nag(feval, params, optim_state)
    local time = timer:time().real

    local train_loss = loss[1] -- the loss is inside a list, pop it
    train_losses[i] = train_loss

    -- exponential learning rate decay
    if math.floor(epoch) > math.floor(last_epoch) and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

    -- every now and then or on last iteration
    if i % opt.eval_val_every == 0 or i == iterations then
        -- evaluate loss on validation data
        local val_loss, val_acc = eval_split(2) -- 2 = validation
        val_losses[i] = val_loss
        val_accs[i] = val_acc

        local savefile = string.format('%s/lm_%s_%d_%d_%d_%d_%d_%d_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, opt.rnn_size_vid, opt.rnn_size_aud, opt.rnn_size_comb, opt.num_layers_vid, opt.num_layers_aud, opt.num_layers_comb, epoch, val_acc)
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.val_acc = val_loss
        checkpoint.val_accs = val_accs
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.vocab = loader.vocab_mapping
        torch.save(savefile, checkpoint)
    end

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.2fs, learning_rate = %.5f", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time, optim_state.learningRate))
    end
   
    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break -- halt
    end
    if loss0 == nil then loss0 = loss[1] end
    if loss[1] > loss0 * 50 then -- originally 3
        print('loss is exploding, aborting.')
        break -- halt
    end
    last_epoch = epoch
end
