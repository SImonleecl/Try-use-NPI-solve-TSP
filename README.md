# Try-use-NPI-solve-TSP

This is not successful test on NPI, there are mainly two reasons. One is that excution data is too large. Even though I simplified NPI structure VScode still crashed down. The other one is that I am not faimiliar with LSTM core, so I use BasicLSTMCell in this model. However, BasicLSTMCell cannot support for training if the number of output braches is more than one.
Introducion on model
1.demo.py shows the basic idea solve TSP with NPI
2.encoder.py shows the expanded function and it will generate 'encodertraining.npy' as the state data without encoding
3.eco.py will use encodertraining.npy to generate and train encoder and save all parameters, including encoder and decoder parts, to
  "my_encoder/save_net.ckpt"
4.VartoCons.py will extract data form "my_encoder/save_net.ckpt" and generate weight and bias data of encoder, which are 'ecow1.npy'.'ecow2.npy','ecob1.npy' and 'ecob2.npy'.
5. LSTMTRAIN.py(failed) will use encoder data and train. Warning that it is easily crashed down. So there is no output analysis until this part.
