
local LSTM2_all = {}
function LSTM2_all.lstm(input_size_vid1, input_size_vid2, input_size_aud, output_size, rnn_size_vid1, rnn_size_vid2, rnn_size_aud, rnn_size_comb, n_vid1, n_vid2, n_aud, n_comb, dropout)
  dropout = dropout or 0

  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x_vid1
  table.insert(inputs, nn.Identity()()) -- x_vid2
  table.insert(inputs, nn.Identity()()) -- x_aud
  for L=1,n_vid1 do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end
  for L=1,n_vid2 do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  --table.insert(inputs, nn.Identity()()) -- x_aud
  for L=1,n_aud do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  for L=1,n_comb do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  -- LSTM for video1
  local x_vid1, input_size_L_vid1
  local outputs = {}
  for L = 1,n_vid1 do
    local prev_h_vid1 = inputs[(L+1)*2 + 1]
    local prev_c_vid1 = inputs[(L+1)*2]
    if L == 1 then
      x_vid1 = inputs[1]
      input_size_L_vid1 = input_size_vid1
    else
      x_vid1 = outputs[(L-1)*2]
      if dropout > 0 then x_vid1 = nn.Dropout(dropout)(x_vid1) end
      input_size_L_vid1 = rnn_size_vid1
    end

    local i2h_vid1 = nn.Linear(input_size_L_vid1, 4 * rnn_size_vid1)(x_vid1):annotate{name='i2h_v1_'..L}
    local h2h_vid1 = nn.Linear(rnn_size_vid1, 4 * rnn_size_vid1)(prev_h_vid1):annotate{name='h2h_v1_'..L}
    local all_input_sums_vid1 = nn.CAddTable()({i2h_vid1, h2h_vid1})
    
    local reshaped_vid1 = nn.Reshape(4, rnn_size_vid1)(all_input_sums_vid1)
    local n1_v1, n2_v1, n3_v1, n4_v1 = nn.SplitTable(2)(reshaped_vid1):split(4)
    -- decode the gates
    local in_gate_vid1 = nn.Sigmoid()(n1_v1)
    local forget_gate_vid1 = nn.Sigmoid()(n2_v1)
    local out_gate_vid1 = nn.Sigmoid()(n3_v1)
    -- decode the write inputs
    local in_transform_vid1 = nn.Tanh()(n4_v1)

    local next_c_vid1 = nn.CAddTable()({nn.CMulTable()({forget_gate_vid1, prev_c_vid1}), nn.CMulTable()({in_gate_vid1, in_transform_vid1})})

    local next_h_vid1 = nn.CMulTable()({out_gate_vid1, nn.Tanh()(next_c_vid1)})

    table.insert(outputs, next_c_vid1)
    table.insert(outputs, next_h_vid1)

  end

  -- LSTM for video2
  local x_vid2, input_size_L_vid2
  for L = 1,n_vid2 do
    local prev_h_vid2 = inputs[2 * n_vid1 + 3 + 2 * L ]
    local prev_c_vid2 = inputs[2 * n_vid1 + 3 + 2 * L - 1]
    if L == 1 then
      x_vid2 = inputs[2]
      input_size_L_vid2 = input_size_vid2
    else
      x_vid2 = outputs[2*n_vid1 + (L-1)*2]
      if dropout > 0 then x_vid2 = nn.Dropout(dropout)(x_vid2) end
      input_size_L_vid2 = rnn_size_vid2
    end

    local i2h_vid2 = nn.Linear(input_size_L_vid2, 4 * rnn_size_vid2)(x_vid2):annotate{name='i2h_v2_'..L}
    local h2h_vid2 = nn.Linear(rnn_size_vid2, 4 * rnn_size_vid2)(prev_h_vid2):annotate{name='h2h_v2_'..L}
    local all_input_sums_vid2 = nn.CAddTable()({i2h_vid2, h2h_vid2})
    
    local reshaped_vid2 = nn.Reshape(4, rnn_size_vid2)(all_input_sums_vid2)
    local n1_v2, n2_v2, n3_v2, n4_v2 = nn.SplitTable(2)(reshaped_vid2):split(4)
    -- decode the gates
    local in_gate_vid2 = nn.Sigmoid()(n1_v2)
    local forget_gate_vid2 = nn.Sigmoid()(n2_v2)
    local out_gate_vid2 = nn.Sigmoid()(n3_v2)
    -- decode the write inputs
    local in_transform_vid2 = nn.Tanh()(n4_v2)

    local next_c_vid2 = nn.CAddTable()({nn.CMulTable()({forget_gate_vid2, prev_c_vid2}), nn.CMulTable()({in_gate_vid2, in_transform_vid2})})

    local next_h_vid2 = nn.CMulTable()({out_gate_vid2, nn.Tanh()(next_c_vid2)})

    table.insert(outputs, next_c_vid2)
    table.insert(outputs, next_h_vid2)

  end
  -- LSTM for audio
  local x_aud, input_size_L_aud
  for L = 1,n_aud do
    local prev_h_aud = inputs[2 * (n_vid1 + n_vid2) + 3 + L * 2]
    local prev_c_aud = inputs[2 * (n_vid1 + n_vid2) + 3 + L * 2 - 1]
    if L == 1 then
      x_aud = inputs[3]
      input_size_L_aud = input_size_aud
    else
      x_aud = outputs[2*(n_vid1 + n_vid2) + (L-1)*2]
      if dropout > 0 then x_aud = nn.Dropout(dropout)(x_aud) end
      input_size_L_aud = rnn_size_aud
    end

    local i2h_aud = nn.Linear(input_size_L_aud, 4 * rnn_size_aud)(x_aud):annotate{name='i2h_a_'..L}
    local h2h_aud = nn.Linear(rnn_size_aud, 4 * rnn_size_aud)(prev_h_aud):annotate{name='h2h_a_'..L}
    local all_input_sums_aud = nn.CAddTable()({i2h_aud, h2h_aud})
    
    local reshaped_aud = nn.Reshape(4, rnn_size_aud)(all_input_sums_aud)
    local n1_a, n2_a, n3_a, n4_a = nn.SplitTable(2)(reshaped_aud):split(4)
    -- decode the gates
    local in_gate_aud = nn.Sigmoid()(n1_a)
    local forget_gate_aud = nn.Sigmoid()(n2_a)
    local out_gate_aud = nn.Sigmoid()(n3_a)
    -- decode the write inputs
    local in_transform_aud = nn.Tanh()(n4_a)

    local next_c_aud = nn.CAddTable()({nn.CMulTable()({forget_gate_aud, prev_c_aud}), nn.CMulTable()({in_gate_aud, in_transform_aud})})

    local next_h_aud = nn.CMulTable()({out_gate_aud, nn.Tanh()(next_c_aud)})

    table.insert(outputs, next_c_aud)
    table.insert(outputs, next_h_aud)

  end
 
  -- LSTM for combination
  local x_comb, input_size_L_comb
  for L = 1,n_comb do
    local prev_h_comb = inputs[2 * (n_vid1 + n_vid2 + n_aud) + 3 + 2 * L]
    local prev_c_comb = inputs[2 * (n_vid1 + n_vid2 + n_aud) + 3 + 2 * L - 1]
    if L == 1 then
      x_comb = nn.JoinTable(2)({outputs[2*n_vid1], outputs[2*n_vid1 + 2*n_vid2], outputs[2*n_vid1 + 2*n_vid2 + 2*n_aud]})
      x_comb = nn.Sigmoid()(x_comb)
      input_size_L_comb = rnn_size_vid1 + rnn_size_vid2 + rnn_size_aud
    else
      x_comb = outputs[2*(n_vid1 + n_vid2 + n_aud) + (L-1)*2]
      if dropout > 0 then x_comb = nn.Dropout(dropout)(x_comb) end
      input_size_L_comb = rnn_size_comb
    end

    local i2h_comb = nn.Linear(input_size_L_comb, 4 * rnn_size_comb)(x_comb):annotate{name='i2h_c_'..L}
    local h2h_comb = nn.Linear(rnn_size_comb, 4 * rnn_size_comb)(prev_h_comb):annotate{name='h2h_c_'..L}
    local all_input_sums_comb = nn.CAddTable()({i2h_comb, h2h_comb})

    local reshaped_comb = nn.Reshape(4, rnn_size_comb)(all_input_sums_comb)
    local n1_c, n2_c, n3_c, n4_c = nn.SplitTable(2)(reshaped_comb):split(4)
    -- decode the gates
    local in_gate_comb = nn.Sigmoid()(n1_c)
    local forget_gate_comb = nn.Sigmoid()(n2_c)
    local out_gate_comb = nn.Sigmoid()(n3_c)
    -- decode the write inputs
    local in_transform_comb = nn.Tanh()(n4_c)

    local next_c_comb = nn.CAddTable()({nn.CMulTable()({forget_gate_comb, prev_c_comb}), nn.CMulTable()({in_gate_comb, in_transform_comb})})

    local next_h_comb = nn.CMulTable()({out_gate_comb, nn.Tanh()(next_c_comb)})

    table.insert(outputs, next_c_comb)
    table.insert(outputs, next_h_comb)

  end

  top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size_comb, output_size)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  --local sigmoid_output = nn.Sigmoid()(proj)
  --table.insert(outputs, sigmoid_output)

  return nn.gModule(inputs, outputs)
end

return LSTM2_all
