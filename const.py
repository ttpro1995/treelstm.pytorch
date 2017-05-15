attention_hid_dim = 150
mlp_com_hid_dim = 150
mlp_com_out_dim = 150
mlp_num_hid_layer = 1
p_dropout_input = 0.5
p_dropout_memory = 0.1
attention = False
def show_setting():
    print('Attention: ' + str(attention))
    if attention:
        print('Attention hidden dim: ' + str(attention_hid_dim))
    print('Dropout input: ' + str(p_dropout_input))
    print('Dropout memory: ' + str(p_dropout_memory))
    print('Compositional MLP: ' + str(mlp_com_hid_dim) + ' ' + str(mlp_num_hid_layer) + ' ' +  str(mlp_com_out_dim))
