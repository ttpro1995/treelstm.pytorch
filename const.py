attention_hid_dim = 150
mlp_com_hid_dim = 150
mlp_com_out_dim = 150
mlp_num_hid_layer = -1
# p_dropout_input = 0.5
# p_dropout_memory = 0.1
attention = False
def show_setting():
    print('Attention: ' + str(attention))
    if attention:
        print('Attention hidden dim: ' + str(attention_hid_dim))
    print('Compositional MLP: ' + str(mlp_com_hid_dim) + ' ' + str(mlp_num_hid_layer) + ' ' +  str(mlp_com_out_dim))
"""
2017-05-14 13:16:22,414 : INFO :
Namespace(at_hid_dim=100, batchsize=25, cuda=True, data='data/sst/', emblr=0.05, epochs=30, fine_grain=False, glove='data/glove/', 
input_dim=300, lr=0.05, mem_dim=150, name='full_comlstm_mid_adagad_norel_13_13_14_5', 
num_classes=3, optim='adagrad', reg=0.0001, rel_dim=0, rel_emblr=0.0, rel_glove=False, saved='saved_model/', 
seed=123, tag_dim=50, tag_emblr=0.05, tag_glove=True, wd=0.0001, word_dim=300)
2017-05-14 13:16:22,611 : INFO : ==> SST vocabulary size : 21701
"""