
def log_gradients(model, summary, step_it, grad=True):

    if grad:
        # Visualise gradients of learnable parameters
        summary.add_histogram(f'Q_grad', model.MHSA.Q.weight.grad, step_it)
        summary.add_histogram(f'K_grad', model.MHSA.K.weight.grad, step_it)
        summary.add_histogram(f'V_grad', model.MHSA.V.weight.grad, step_it)
        summary.add_histogram(f'Unary_grad', model.unary.weight.grad, step_it)

    # Visualise weights of learnable parameters
    summary.add_histogram(f'Q_weights', model.MHSA.Q.weight, step_it)
    summary.add_histogram(f'K_weights', model.MHSA.K.weight, step_it)
    summary.add_histogram(f'V_weights', model.MHSA.V.weight, step_it)
    summary.add_histogram(f'Unary_weights', model.unary.weight, step_it)

    for i in model.grads:
        u, p = i

        summary.add_histogram(f'U', u, step_it)
        summary.add_histogram(f'P', p, step_it)

    if grad:
        summary.add_scalars('Abs_Mean Grads', 
                {'Q': model.MHSA.Q.weight.grad.abs().mean(), 
                'K':  model.MHSA.K.weight.grad.abs().mean(), 
                'V':  model.MHSA.V.weight.grad.mean(),
                'Unary': model.unary.weight.grad.abs().mean()
                }, step_it)
        summary.add_scalars('Abs_Max Grads', 
                {'Q': model.MHSA.Q.weight.grad.abs().max(), 
                'K':  model.MHSA.K.weight.grad.abs().max(), 
                'V':  model.MHSA.V.weight.grad.mean(),
                'Unary': model.unary.weight.grad.abs().max()
                }, step_it)
