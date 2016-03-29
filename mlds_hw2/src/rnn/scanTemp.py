def step(x_t, y_tm1):
    return self.oneStep(x_t, y_tm1)

y_0 = theano.shared(np.zeros(N_OUTPUT))

[y_seq],_ = theano.scan(
            step,
            sequences = x_seq,
            outputs_info = [ y_0 ],
            truncate_gradient=-1
            )

gradients = T.grad(cost,parameters)
def MyUpdate(parameters,gradients):
    mu =  np.float32(0.001)
    parameters_updates = [(p,p - mu * g) for p,g in izip(parameters,gradients) ]
    return parameters_updates

rnn_test = theano.function(
        inputs= [x_seq],
        outputs=y_seq_last
)

rnn_train = theano.function(
        inputs=[x_seq,y_hat],
        outputs=cost,
    updates=MyUpdate(parameters,gradients)
)

for i in range(10000000):
    x_seq, y_hat = gen_data()
    print "iteration:", i, "cost:",  rnn_train(x_seq,y_hat)

for i in range(10):
    x_seq, y_hat = gen_data()
    print "reference", y_hat, "RNN output:", rnn_test(x_seq)
