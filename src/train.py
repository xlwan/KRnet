from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import time

import sys
sys.path.append('../')
import BR_lib.BR_model as BR_model
import BR_lib.BR_data as BR_data

# choose GPU 0 or 1
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

def main(args):
    def get_data(n_train, n_dim):
        w = np.ones((n_train,1), dtype='float32')
        
        x, _ = BR_data.gen_xd_Logistic_w_2d_hole(n_dim, n_train, 2.0, 3.0, np.pi/4.0, 7.6)
        #np.savetxt('./dataset_for_training/Logistic_8d_w_2d_holes_double.dat'.format(), x)
        return x,w

    x,w = get_data(args.n_train, args.n_dim)

    data_flow = BR_data.dataflow(x, buffersize=args.n_train, batchsize=args.batch_size, y=w)
    train_dataset = data_flow.get_shuffled_batched_dataset()

    # create the model
    def create_model():
        # build up the model
        pdf_model = BR_model.IM_rNVP_KR_CDF('pdf_model_KR_CDF',
                                            args.n_dim,
                                            args.n_step,
                                            args.n_depth,
                                            n_width=args.n_width,
                                            n_bins=args.n_bins4cdf,
                                            shrink_rate=args.shrink_rate,
                                            flow_coupling=args.flow_coupling,
                                            rotation=args.rotation)
        return pdf_model

    # call model one to complete the building process
    x_init = data_flow.get_n_batch_from_shuffled_batched_dataset(1)
    pdf_model = create_model()
    pdf_model(x_init)

    def get_loss(x, w):
        pdf = pdf_model(x)
        loss = -tf.reduce_mean(pdf*w)

        reg = tf.constant(0.0)

        loss += reg

        return loss, reg

    # metrics for loss and regularization
    loss_metric = tf.keras.metrics.Mean()
    reg_metric = tf.keras.metrics.Mean()

    # stochastic gradient method ADAM
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    # check point and initialization
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=pdf_model)
    manager=tf.train.CheckpointManager(ckpt, args.ckpts_dir, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print(" ------ Initializing from scratch ------")
        m = 4 # the number minibatches used for data initialization
        x_init = data_flow.get_n_batch_from_shuffled_batched_dataset(m)
        #if args.rotation:
        #    pdf_model.WLU_data_initialization()
        pdf_model.actnorm_data_initialization()
        pdf_model(x_init)
    # summary
    summary_writer = tf.summary.create_file_writer(args.summary_dir)

    # prepare one training iteration step
    @tf.function
    def train_step(inputs, vars):
        xt, wt = inputs
        with tf.GradientTape() as tape:
            loss, reg = get_loss(xt, wt)

        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))

        return loss, reg

    # used for the computation of KL divergence
    n_valid = 16000
    #y = np.loadtxt('./dataset_for_training/Logistic_8d_w_2d_holes_valid.dat').astype(np.float32)[:n_valid,:]
    y, _ = BR_data.gen_xd_Logistic_w_2d_hole(args.n_dim, n_valid, 2.0, 3.0, np.pi/4.0, 7.6)

    loss_vs_epoch=[]
    reg_vs_epoch=[]
    KL_vs_epoch=[]

    n_epochs = args.n_epochs
    with summary_writer.as_default():
        # iterate over epochs
        iteration = 0
        for i in range(1,n_epochs+1):
            # freeze the rotation layers after a certain number of epochs
            g_vars = pdf_model.trainable_weights

            start_time = time.time()
            # iterate over the batches of the dataset
            for step, train_batch in enumerate(train_dataset):
                loss, reg = train_step((train_batch[0], train_batch[1]), g_vars)

                loss_metric(loss-reg)
                reg_metric(reg)

                iteration += 1

                # write the summary file
                #if tf.equal(optimizer.iterations % args.log_step, 0):
                #    tf.summary.scalar('loss', loss_metric.result(), step=optimizer.iterations)
    
            ln_f = pdf_model(y)
            ln_y = -y/2.0 - np.log(2.0) -2.0*np.log(1+np.exp(-y/2.0))
            ln_y = tf.reduce_sum(ln_y, axis=1, keepdims=True) - np.log(0.18360558216051442) # for 4d
            #ln_y = tf.reduce_sum(ln_y, axis=1, keepdims=True) - np.log(0.0220251667263824) # for 8d H is: -21.527786
            kl_d = -tf.reduce_mean(ln_f-ln_y)

            print('epoch %s, iteration %s, loss = %s, reg = %s,  kl_d = %s, time = %s' %
                             (i, iteration, loss_metric.result().numpy(), reg_metric.result().numpy(), 
                                 kl_d.numpy(), time.time()-start_time))

            loss_vs_epoch += [loss_metric.result().numpy()]
            reg_vs_epoch  += [reg_metric.result().numpy()]
            KL_vs_epoch += [kl_d.numpy()]

            loss_metric.reset_states()
            reg_metric.reset_states()

            # re-shuffle the dataset
            train_dataset = data_flow.update_shuffled_batched_dataset()

            ckpt.step.assign_add(1)
            if int(ckpt.step) % args.ckpt_step == 0:
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

            if i % args.n_draw_samples == 0:
                xs = pdf_model.draw_samples_from_prior(args.n_samples, args.n_dim)
                ys = pdf_model.mapping_from_prior(xs)
                np.savetxt('epoch_{}_prior.dat'.format(i), xs.numpy())
                np.savetxt('epoch_{}_sample.dat'.format(i), ys.numpy())

        c1=np.array(range(1,n_epochs+1)).reshape(-1,1)
        c2=np.array(loss_vs_epoch).reshape(-1,1)
        c3=np.array(reg_vs_epoch).reshape(-1,1)
        c4=np.array(KL_vs_epoch).reshape(-1,1)
        np.savetxt('cong_vs_epoch.dat',np.concatenate((c1, c2, c3, c4), axis=1))

if __name__ == '__main__':
    from configargparse import ArgParser
    p = ArgParser()
    # Data arguments
    p.add('--data_dir', type=str, help='Path to preprocessed data files.')

    # save parameters
    p.add('--ckpts_dir', type=str, default='./pdf_ckpt', help='Path to the check points.')
    p.add('--summary_dir', type=str, default='./pdf_summary', help='Path to the summaries.')
    p.add('--log_step', type=int, default=16, help='Record information every n optimization iterations.')
    p.add('--ckpt_step', type=int, default=50, help='Save the model every n epochs.')

    # Neural network hyperparameteris
    p.add('--n_depth', type=int, default=4, help='The number of affine coupling layers.')
    p.add('--n_width', type=int, default=24, help='The number of neurons for the hidden layers.')
    p.add('--n_step', type=int, default=1, help='The step size for dimension reduction in each squeezing layer.')
    p.add('--rotation', action='store_true', help='Specify rotation layers or not?')
    #p.set_defaults(rotation=True)
    p.add('--n_bins4cdf', type=int, default=4, help='The number of bins for uniform partition of the support of PDF.')
    p.add('--flow_coupling', type=int, default=1, help='Coupling type: 0=additive, 1=affine.')
    p.add('--shrink_rate', type=float, default=0.9, help='The shrinking rate of the width of NN.')

    #optimization hyperparams:
    p.add("--n_dim", type=int, default=4, help='The number of random dimension.')
    p.add("--n_train", type=int, default=16000, help='The number of samples in the training set.')
    p.add('--batch_size', type=int, default=4000, help='Batch size of training generator.')
    p.add("--lr", type=float, default=0.001, help='Base learning rate.')
    p.add('--n_epochs',type=int, default=8000, help='Total number of training epochs.')

    # samples:
    p.add("--n_samples", type=int, default=100000, help='Sample size for the trained model.')
    p.add("--n_draw_samples", type=int, default=1000, help='Draw samples every n epochs.')

    args = p.parse_args()
    main(args)
