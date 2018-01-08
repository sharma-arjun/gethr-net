
local LSTM2 = {}
function LSTM2.lstm(input_size_vid, input_size_aud, output_size, rnn_size_vid, rnn_size_aud, rnn_size_comb, n_vid, n_aud, n_comb, dropout)
  dropout = dropout or 0

  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x_vid
  table.insert(inputs, nn.Identity()()) -- x_aud
  for L=1,n_vid do
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

  -- LSTM for video
  local x_vid, input_size_L_vid
  local outputs = {}
  for L = 1,n_vid do
    local prev_h_vid = inputs[(L+1)*2]
    local prev_c_vid = inputs[L*2 + 1]
    if L == 1 then
      x_vid = inputs[1]
      input_size_L_vid = input_size_vid
    else
      x_vid = outputs[(L-1)*2]
      if dropout > 0 then x_vid = nn.Dropout(dropout)(x_vid) end
      input_size_L_vid = rnn_size_vid
    end

    local i2h_vid = nn.Linear(input_size_L_vid, 4 * rnn_size_vid)(x_vid):annotate{name='i2h_v_'..L}
    local h2h_vid = nn.Linear(rnn_size_vid, 4 * rnn_size_vid)(prev_h_vid):annotate{name='h2h_v_'..L}
    local all_input_sums_vid = nn.CAddTable()({i2h_vid, h2h_vid})
    
    local reshaped_vid = nn.Reshape(4, rnn_size_vid)(all_input_sums_vid)
    local n1_v, n2_v, n3_v, n4_v = nn.SplitTable(2)(reshaped_vid):split(4)
    -- decode the gates
    local in_gate_vid = nn.Sigmoid()(n1_v)
    local forget_gate_vid = nn.Sigmoid()(n2_v)
    local out_gate_vid = nn.Sigmoid()(n3_v)
    -- decode the write inputs
    local in_transform_vid = nn.Tanh()(n4_v)

    local next_c_vid = nn.CAddTable()({nn.CMulTable()({forget_gate_vid, prev_c_vid}), nn.CMulTable()({in_gate_vid, in_transform_vid})})

    local next_h_vid = nn.CMulTable()({out_gate_vid, nn.Tanh()(next_c_vid)})

    table.insert(outputs, next_c_vid)
    table.insert(outputs, next_h_vid)

  end

  -- LSTM for audio
  local x_aud, input_size_L_aud
  for L = 1,n_aud do
    local prev_h_aud = inputs[2 * n_vid + 2 + 2 * L]
    local prev_c_aud = inputs[2 * n_vid + 2 +  2 * L - 1]
    if L == 1 then
      x_aud = inputs[2]
      input_size_L_aud = input_size_aud
    else
      x_aud = outputs[2*n_vid + (L-1)*2]
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
    local prev_h_comb = inputs[2 * n_vid + 2 * n_aud + 2 + 2 * L]
    local prev_c_comb = inputs[2 * n_vid + 2 * n_aud + 2 + 2 * L - 1]
    if L == 1 then
      x_comb = nn.JoinTable(2)({outputs[2*n_vid], outputs[2*n_vid + 2*n_aud]})
      input_size_L_comb = rnn_size_vid + rnn_size_aud
    else
      x_comb = outputs[2*n_vid + 2*n_aud + (L-1)*2]
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

return LSTM2
