# Try-use-NPI-solve-TSP

<p>This is not successful test on NPI, there are mainly two reasons. One is that excution data is too large. Even though I simplified NPI structure VScode still crashed down. The other one is that I am not faimiliar with LSTM core, so I use BasicLSTMCell in this model. However, BasicLSTMCell cannot support for training if the number of output braches is more than one.</p>
<p>Then I use sub-program of TPS problem, and it works and partially achieve the goals of NPI. This part includes training, running in different-size problems instances, LSTM error and NPI accuracy and ablation study.</p>
<h1>Introducion on model</h1>
<p>1. demo.py shows the basic idea solve TSP with NPI</p>
<p>2. encoder.py shows the expanded function and it will generate 'encodertraining.npy' as the state data without encoding</p>
<p>3. eco.py will use encodertraining.npy to generate and train encoder and save all parameters, including encoder and decoder parts, to
  "my_encoder/save_net.ckpt"</p>
<p>4. VartoCons.py will extract data form "my_encoder/save_net.ckpt" and generate weight and bias data of encoder, which are 'ecow1.npy'.'ecow2.npy','ecob1.npy' and 'ecob2.npy'.</p>
<p>5. LSTMTRAIN.py(failed) will use encoder data and train. Warning that it is easily crashed down. So there is no output analysis until this part.</p>
<p>6. data.py can show the scale of excution trace, so it can prove why LSTMTRAIN.py crashes down.</p>
<h2>Sub-program</h2>
<p>7. LSTMtu.py will generate trained NPI model saved to "my_net/save_net.skpt". The way to getting data and how to the sub-program runs is shown in get_batch(). The number of cities in training data is 2 to 5. Also it will generate a graph of tensorboard, it can show total structure.</p>
<p>8. Test.py will load LSTM structure from "my_net/save_net.skpt" and output testing error depend on different number of cities after setting. The program also save input trace and output trace as 'LSTMinput.npy' and 'LSTMoutput.npy'</p>
<p>9. NPI.py will load data form 'LSTMinput.npy' and 'LSTMoutput.npy' and will output test result of NPI. The result setting equals number of cities plus one.</p>
<p>10. ablation.py is similar to Test.py. The change is get_batch(). Data is shuffled and noise is added.</p>
<h2>Other files</h2>
<p>11. report.pdf is a report on this project</p>
<p>12. TestResult.xlsx is the data used in report.</p>
