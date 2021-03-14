import tensorflow as tf
import time
import numpy as np

localtime = time.asctime(time.localtime(time.time()))
print(localtime)

print(tf.__version__)

tf.app.flags.DEFINE_string("tables", "", "tables info including train/test")
tf.app.flags.DEFINE_integer('is_valid', 1,
                            '1: validation; 0: not validation')
FLAGS = tf.app.flags.FLAGS

# [train_table, test_table]
tables = FLAGS.tables.strip().split(",")

# ===================================== hyper para
BATCH_SIZE = 1024
EMB_DIM = 8
NUM_EPOCH = int(1e5)
L2_REG =  1.0
KEEP_PROB = 0.4
model_type = 'co_att_trans_final'


N_USER_ID_HASH = int(4e5)
N_USER_AGE_HASH = int(1000)
N_USER_GENDER_HASH = 10
N_USER_PROV_HASH = 300
N_USER_CITY_HASH = int(5e3)  # 828
N_USER_COUNTY_HASH = int(2e4)  # 4054
N_USER_PURCHASE_HASH = 50

N_ITEM_CATE_HASH = int(1e5)  # 26747
N_ITEM_CATE_LEVEL_HASH = 1000  # 198

# 8121032 too large, maybe need partion embedding matrix
N_ITEM_BRAND_HASH = int(2e7)  # 4e5)  # int(1e7)

N_ITEM_SCORE_HASH = 100  # edge features

user_hash_size_list = [N_USER_AGE_HASH, N_USER_GENDER_HASH, N_USER_PROV_HASH,
                       N_USER_CITY_HASH, N_USER_COUNTY_HASH, N_USER_PURCHASE_HASH]
item_hash_size_list = [N_ITEM_CATE_HASH, N_ITEM_CATE_LEVEL_HASH, N_ITEM_BRAND_HASH]

initializer = tf.contrib.layers.xavier_initializer(uniform=False)
regularizer = tf.contrib.layers.l2_regularizer(L2_REG)


def decode_node_attr(infos, hash_size_list, is_hash=False):
    # decode arbitrary num of node attr, len(infos) can be arbitrary number
    # work for both user and item
    fea_val_list = [tf.decode_csv(info,
                                  [[" "], [" "]],
                                  ":")[1]
                    for info in infos]
    if is_hash:
        fea_hash_list = [tf.string_to_hash_bucket(i, j)
                         for (i, j) in zip(fea_val_list, hash_size_list)]
        return fea_hash_list
    return fea_val_list


def decode_node_list_attr(infos, node_num, hash_size_list, is_hash=False):
    """
    decode artibrary len node_fea list, e.g., user_friend_list or user_buy_list
    node_num: num of node in list, e.g., num of user friend
    """
    infos_list = tf.decode_csv(infos,
                               [[" "]] * node_num,
                               chr(3))
    infos_fea_list = [tf.decode_csv(i,
                                    [[' ']] * len(hash_size_list),
                                    '#')
                      for i in infos_list]

    infos_fea_val_list = [decode_node_attr(node, hash_size_list,
                                           is_hash=False)
                          for node in infos_fea_list]
    # print('infos_fea_val_list' , infos_fea_val_list)
    return_list = [[] for i in range(len(hash_size_list))]

    # print(len(return_list), len(infos_fea_val_list), len(infos_fea_val_list[0]))
    for x in infos_fea_val_list:
        for idx, val in enumerate(hash_size_list):
            return_list[idx].append(x[idx])
    # print(return_list, len(return_list))

    if is_hash:
        return_hash_list = [
            tf.string_to_hash_bucket(node, hash_size)
            for node, hash_size in zip(return_list, hash_size_list)
        ]
        return return_hash_list


def input_fn_1021(table,
                  selected_cols="u_fea,v_fea,i_fea,u_friend,v_friend,u_share,v_share,u_pay,v_pay,i_buy,label",
                  shuffle=True):
    """
    selected_cols: label must be the last one
    for u, i, v
    shuffle=True for train/val
    shuffle=False for test
    """
    col_num = len(selected_cols.split(','))
    print('input_fn: {}'.format(table))
    print('select col: {}'.format(selected_cols))
    file_queue = tf.train.string_input_producer([table],
                                                num_epochs=NUM_EPOCH,
                                                shuffle=shuffle)

    reader = tf.TableRecordReader(selected_cols=selected_cols)
    keys, values = reader.read_up_to(file_queue,
                                     num_records=BATCH_SIZE)
    # , to_ndarray=False) # len(red) = num_records

    # src_user_fea, des_user_fea, src_user_items, des_user_items, _, _
    default_val = [[' ']] * col_num
    default_val[-1] = [-1.0]
    [u_fea, v_fea, i_fea, u_tao_friend, v_tao_friend, u_share, v_share, u_pay, v_pay, i_buy,
     label] = tf.decode_csv(values,
                            default_val)
    # u_fea type: id_age:15#id_gender:2#
    u_fea = tf.decode_csv(u_fea,
                          [[' ']] * 6,
                          "#")
    v_fea = tf.decode_csv(v_fea,
                          [[' ']] * 6,
                          "#")
    i_fea = tf.decode_csv(i_fea,
                          [[' ']] * 3,
                          "#")

    u_info_hash = decode_node_attr(u_fea,
                                   user_hash_size_list,
                                   is_hash=True)
    v_info_hash = decode_node_attr(v_fea,
                                   user_hash_size_list,
                                   is_hash=True)
    i_info_hash = decode_node_attr(i_fea,
                                   item_hash_size_list,
                                   is_hash=True)

    uf_info_hash = decode_node_list_attr(u_tao_friend,
                                         5,
                                         user_hash_size_list,
                                         is_hash=True)
    vf_info_hash = decode_node_list_attr(v_tao_friend,
                                         5,
                                         user_hash_size_list,
                                         is_hash=True)
    us_info_hash = decode_node_list_attr(u_share,
                                         10,
                                         user_hash_size_list,
                                         is_hash=True)
    vs_info_hash = decode_node_list_attr(v_share,
                                         10,
                                         user_hash_size_list,
                                         is_hash=True)
    up_info_hash = decode_node_list_attr(u_pay,
                                         2,
                                         user_hash_size_list,
                                         is_hash=True)
    vp_info_hash = decode_node_list_attr(v_pay,
                                         2,
                                         user_hash_size_list,
                                         is_hash=True)
    ib_info_hash = decode_node_list_attr(i_buy,
                                         50,
                                         user_hash_size_list,
                                         is_hash=True)



    return u_info_hash, v_info_hash, i_info_hash, \
           uf_info_hash, vf_info_hash, \
           us_info_hash, vs_info_hash, \
           up_info_hash, vp_info_hash, \
           ib_info_hash, \
           label


def cat_fea_emb_list(fea_list):
    return tf.concat(fea_list, axis=-1)


def multi_fea_emb_list(emb_list):
    # list of node embedding [2d, 2d, ]
    # to a 3d tensor, [None, len_list, emb_size]
    emb_list_expand = [tf.expand_dims(emb, axis=1) for emb in emb_list]
    return tf.concat(emb_list_expand, axis=1)


def avg_fea_emb_list(fea_list):
    # for both 2-D and 3-D tensor
    fea_list_expanded = [tf.expand_dims(fea, axis=-1) for fea in fea_list]
    fea_list_concat = tf.concat(fea_list_expanded, axis=-1)
    fea_list_avg = tf.reduce_mean(fea_list_concat, axis=-1)
    return fea_list_avg



def aggregator(node, neigh, type='mean'):
    if type == 'mean':
        return tf.concat([node, tf.reduce_mean(neigh, axis=1)],
                         axis=1)



def model_fn_1021(u_info_hash, v_info_hash, i_info_hash,
                  uf_info_hash, vf_info_hash,
                  us_info_hash, vs_info_hash,
                  up_info_hash, vp_info_hash,
                  ib_info_hash,
                  batch_y,
                  keep_prob,
                  model_type=None):
    # avg_trans, co_att_concat, avg_concat, co_att_trans, meirec, co_att_trans, han_trans, avg_concat_noshare, co_att_trans_noshare
    print('model: {} ........'.format(model_type))

    all_emb_mat = {}
    for idx, val in enumerate(user_hash_size_list):
        all_emb_mat['user_{}_emb_mat'.format(idx)] = tf.get_variable('user_{}_emb_mat'.format(idx),
                                                                     [val, EMB_DIM],
                                                                     initializer=initializer)
    for idx, val in enumerate(item_hash_size_list):
        all_emb_mat['item_{}_emb_mat'.format(idx)] = tf.get_variable('item_{}_emb_mat'.format(idx),
                                                                     [val, EMB_DIM],
                                                                     initializer=initializer)
    u_fea_emb_list = [
        tf.nn.embedding_lookup(
            all_emb_mat['user_{}_emb_mat'.format(i)], u_info_hash[i])
        for i in range(len(u_info_hash))
    ]
    v_fea_emb_list = [
        tf.nn.embedding_lookup(
            all_emb_mat['user_{}_emb_mat'.format(i)], v_info_hash[i])
        for i in range(len(v_info_hash))
    ]

    u_fea_final = cat_fea_emb_list(u_fea_emb_list)
    v_fea_final = cat_fea_emb_list(v_fea_emb_list)

    batch_y = tf.expand_dims(batch_y, axis=1)

    i_fea_emb_list = [
        tf.nn.embedding_lookup(
            all_emb_mat['item_{}_emb_mat'.format(i)], i_info_hash[i])
        for i in range(len(i_info_hash))
    ]
    i_fea_final = cat_fea_emb_list(i_fea_emb_list)
    print('u, v, i, shape: ', u_fea_final.shape, v_fea_final.shape, i_fea_final.shape)
    # # =========================================  u, v  friends embedding
    uf_fea_emb_list = [
        tf.nn.embedding_lookup(
            all_emb_mat['user_{}_emb_mat'.format(i)], uf_info_hash[i])
        for i in range(len(uf_info_hash))
    ]

    u_fd_emb_list = [tf.transpose(i, [1, 0, 2]) for i in uf_fea_emb_list]
    u_fd_emb_concat = cat_fea_emb_list(u_fd_emb_list)
    print('uf_fea_emb_list shape: ', uf_fea_emb_list[0].shape)
    print('u_fd_emb_concat shape: ', u_fd_emb_concat.shape)

    vf_fea_emb_list = [
        tf.nn.embedding_lookup(
            all_emb_mat['user_{}_emb_mat'.format(i)], vf_info_hash[i])
        for i in range(len(vf_info_hash))
    ]
    v_fd_emb_list = [tf.transpose(i, [1, 0, 2]) for i in vf_fea_emb_list]
    v_fd_emb_concat = cat_fea_emb_list(v_fd_emb_list)

    u_emb_via_friend = aggregator(u_fea_final, u_fd_emb_concat)
    v_emb_via_friend = aggregator(v_fea_final, v_fd_emb_concat)
    # # =========================================  u, v  share embedding
    us_fea_emb_list = [
        tf.nn.embedding_lookup(
            all_emb_mat['user_{}_emb_mat'.format(i)], us_info_hash[i])
        for i in range(len(us_info_hash))
    ]
    u_share_emb_list = [tf.transpose(i, [1, 0, 2]) for i in us_fea_emb_list]
    u_share_emb_concat = cat_fea_emb_list(u_share_emb_list)
    print('us_fea_emb_list shape: ', us_fea_emb_list[0].shape)
    print('u_share_emb_concat shape: ', u_share_emb_concat.shape)

    vs_fea_emb_list = [
        tf.nn.embedding_lookup(
            all_emb_mat['user_{}_emb_mat'.format(i)], vs_info_hash[i])
        for i in range(len(vs_info_hash))
    ]
    v_share_emb_list = [tf.transpose(i, [1, 0, 2]) for i in vs_fea_emb_list]
    v_share_emb_concat = cat_fea_emb_list(v_share_emb_list)

    u_emb_via_share = aggregator(u_fea_final, u_share_emb_concat)
    v_emb_via_share = aggregator(v_fea_final, v_share_emb_concat)
    # # =========================================  u, v  pay embedding
    up_fea_emb_list = [
        tf.nn.embedding_lookup(
            all_emb_mat['user_{}_emb_mat'.format(i)], up_info_hash[i])
        for i in range(len(up_info_hash))
    ]
    u_pay_emb_list = [tf.transpose(i, [1, 0, 2]) for i in up_fea_emb_list]
    u_pay_emb_concat = cat_fea_emb_list(u_pay_emb_list)
    print('up_fea_emb_list shape: ', up_fea_emb_list[0].shape)
    print('u_pay_emb_concat shape: ', u_pay_emb_concat.shape)

    vp_fea_emb_list = [
        tf.nn.embedding_lookup(
            all_emb_mat['user_{}_emb_mat'.format(i)], vp_info_hash[i])
        for i in range(len(vp_info_hash))
    ]
    v_pay_emb_list = [tf.transpose(i, [1, 0, 2]) for i in vp_fea_emb_list]
    v_pay_emb_concat = cat_fea_emb_list(v_pay_emb_list)

    u_emb_via_pay = aggregator(u_fea_final, u_pay_emb_concat)
    v_emb_via_pay = aggregator(v_fea_final, v_pay_emb_concat)
    # # ============================================ item buy user
    ib_fea_emb_list = [
        tf.nn.embedding_lookup(
            all_emb_mat['user_{}_emb_mat'.format(i)], ib_info_hash[i])
        for i in range(len(ib_info_hash))
    ]
    i_buy_emb_list = [tf.transpose(i, [1, 0, 2]) for i in ib_fea_emb_list]
    i_buy_emb_concat = cat_fea_emb_list(i_buy_emb_list)
    print('i_buy_emb_list shape: ', i_buy_emb_list[0].shape)
    print('i_buy_emb_concat shape: ', i_buy_emb_concat.shape)
    i_emb_via_buy = aggregator(i_fea_final, i_buy_emb_concat)


    if model_type == 'co_att_trans_final':
        # auc can achieve 0.87+, as reproduce version of co_att_trans
        # adopt it from uiv_gnn_1028.py
        final_emb_size = 128
        # user_emb_size = u_emb_via_pay.get_shape().as_list()[1]  # 6 * EMB_DIM * 2
        # item_emb_size = i_emb_via_buy.get_shape().as_list()[1]
        att_mat = tf.get_variable('user_item_att_mat',
                                  [final_emb_size, final_emb_size],
                                  initializer=initializer)
        att_vec_size = 512
        att_vec = tf.get_variable('user_item_att_vec',
                                  [att_vec_size, 1],
                                  initializer=initializer)
        print('att_vec ', att_vec.shape)
        i_multi_emb = multi_fea_emb_list([i_emb_via_buy])
        print('i_multi_emb', i_multi_emb.shape)
        i_multi_emb_proj = tf.layers.dense(i_multi_emb,
                                           final_emb_size,
                                           activation=tf.nn.elu,
                                           use_bias=True,
                                           kernel_initializer=initializer,
                                           kernel_regularizer=regularizer,
                                           name='item_proj'
                                           )
        print('i_multi_emb_proj', i_multi_emb_proj.shape)
        # first_user ---- item, u-i
        u_multi_emb = multi_fea_emb_list([u_emb_via_friend, u_emb_via_share, u_emb_via_pay])
        u_multi_emb_proj = tf.layers.dense(u_multi_emb,
                                           final_emb_size,
                                           activation=tf.nn.elu,
                                           use_bias=True,
                                           kernel_initializer=initializer,
                                           kernel_regularizer=regularizer,
                                           name='user_proj'
                                           )
        u_and_i = tf.concat([u_multi_emb_proj,
                             tf.concat([i_multi_emb_proj, i_multi_emb_proj, i_multi_emb_proj], axis=1)
                             # user
                             ],
                            axis=2)
        u_and_i = tf.layers.dense(u_and_i,
                                  att_vec_size,
                                  activation=tf.nn.elu,
                                  use_bias=True,
                                  kernel_initializer=initializer,
                                  kernel_regularizer=regularizer,
                                  name='u_and_v_proj'
                                  )
        # u_and_i = tf.nn.dropout(u_and_i, keep_prob=0.4)
        # u_att_emb_via_i, i_att_emb_via_u = co_att_process(u_multi_emb_proj, att_mat, i_multi_emb_proj)
        u_att_emb_via_i, i_att_emb_via_u, u_att_val = co_att_process_2(u_multi_emb_proj,
                                                                       att_vec,
                                                                       i_multi_emb_proj,
                                                                       u_and_i)
        print('u_att_emb_via_i {},  i_att_emb_via_u {}'.format(u_att_emb_via_i.shape, i_att_emb_via_u.shape))
        # second_user --- item, v-i
        v_multi_emb = multi_fea_emb_list([v_emb_via_friend, v_emb_via_share, v_emb_via_pay])
        v_multi_emb_proj = tf.layers.dense(v_multi_emb,
                                           final_emb_size,
                                           activation=tf.nn.elu,
                                           use_bias=True,
                                           kernel_initializer=initializer,
                                           kernel_regularizer=regularizer,
                                           name='user_proj',
                                           reuse=True
                                           )
        v_and_i = tf.concat([v_multi_emb_proj,
                             tf.concat([i_multi_emb_proj, i_multi_emb_proj, i_multi_emb_proj], axis=1)
                             # user
                             ],
                            axis=2)
        v_and_i = tf.layers.dense(v_and_i,
                                  att_vec_size,
                                  activation=tf.nn.elu,
                                  use_bias=True,
                                  kernel_initializer=initializer,
                                  kernel_regularizer=regularizer,
                                  name='u_and_v_proj',
                                  reuse=True
                                  )
        # v_and_i = tf.nn.dropout(v_and_i, keep_prob=0.4)
        # v_att_emb_via_i, i_att_emb_via_v = co_att_process(v_multi_emb_proj, att_mat, i_multi_emb_proj)
        v_att_emb_via_i, i_att_emb_via_v, v_att_val = co_att_process_2(v_multi_emb_proj,
                                                                       att_vec,
                                                                       i_multi_emb_proj,
                                                                       v_and_i)
        print('v_att_emb_via_i {}, i_att_emb_via_v {}'.format(v_att_emb_via_i.shape, i_att_emb_via_v.shape))
        h1 = tf.layers.dense(tf.abs(u_att_emb_via_i - v_att_emb_via_i + i_att_emb_via_u),
                             # tf.concat([u_age_emb, v_age_emb], axis=1),
                             # uv_rep,
                             EMB_DIM,
                             activation=tf.nn.elu,
                             use_bias=True,
                             kernel_initializer=initializer,
                             kernel_regularizer=regularizer,
                             name='h1_layer'
                             )

        h1 = tf.nn.dropout(h1, keep_prob=keep_prob)
        pred = tf.layers.dense(h1, 1,
                               activation=None,
                               use_bias=True,
                               kernel_initializer=initializer,
                               kernel_regularizer=regularizer,
                               name='pred_layer'
                               )
        # u_att_emb_via_i = tf.nn.l2_normalize(u_att_emb_via_i, axis=1)
        # v_att_emb_via_i = tf.nn.l2_normalize(v_att_emb_via_i, axis=1)
        # i_att_emb_via_u = tf.nn.l2_normalize(i_att_emb_via_u, axis=1)
        # pred = tf.reduce_sum(
        #     (u_att_emb_via_i + i_att_emb_via_u - v_att_emb_via_i) ** 2,
        #     # abs(tf.reduce_mean(u_multi_emb_proj, axis=1) - tf.reduce_mean(v_multi_emb_proj, axis=1) + tf.reduce_mean(i_multi_emb_proj, axis=1)),
        #     axis=1,
        #     keep_dims=True
        # )




    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_y,
                                                                  logits=pred))
    auc, auc_op = tf.metrics.auc(labels=batch_y,
                                 predictions=tf.nn.sigmoid(pred))
 
    return pred, loss, auc, auc_op

# ======================  train
# train_infos is a tuple (u_info_hash, v_info_hash, i_info_hash, \
#            uf_info_hash, vf_info_hash, \
#            ib_info_hash, ub_info_hash, vb_info_hash, label)
train_infos = input_fn_1021(tables[0], shuffle=True)

with tf.variable_scope('model'):
    train_pred, train_loss, train_auc, train_auc_op, train_u_att_val_op, train_v_att_val_op = model_fn_1021(
        *train_infos,
        keep_prob=KEEP_PROB,
        model_type=model_type)


train_op = tf.train.AdamOptimizer(LR).minimize(train_loss)

init_op = tf.global_variables_initializer()
local_init_op = tf.local_variables_initializer()
max_valid_auc = 0.0
print('start sess....................................')
k = 0
max_k = 7  # 1e5
att_val_list = []
with tf.Session() as sess:
    sess.run(init_op)
    sess.run(local_init_op)

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
 

    try:
        for i in range(NUM_EPOCH):
            t1 = time.time()
 
            train_loss_value, _, _, train_auc_value, train_u_att_val, train_v_att_val = sess.run(
                [train_loss, train_op, train_auc_op, train_auc, train_u_att_val_op, train_v_att_val_op])

            print(i, train_auc_value)


    except tf.errors.OutOfRangeError:
        print('done')

    finally:
        coord.request_stop()
        coord.join(threads)
