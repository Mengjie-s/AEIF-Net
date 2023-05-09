import torch.optim as optim


def get_optimizer(args, model):
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)

    elif args.optimizer == 'Adam':
        # optimizer = optim.Adam(model.parameters(),
        #                        lr=args.lr,
        #                        weight_decay=args.weight_decay)
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight_decay)                      

    elif args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)


    else:
        return NotImplementedError("Optimizer [%s] is not implemented", args.optimizer)

    return optimizer
