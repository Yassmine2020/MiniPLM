import os
import time
import json
import torch
import wandb

from linear_cls_model import LinearCLSModel
from dyna_alpha import LinearCLSModelDynaAlpha
# from fix_alpha import LinearModelFixAlpha
from arguments import get_args


from utils import print_args, initialize, save_rank

torch.backends.cudnn.enabled = False


def main():
    args = get_args()
    initialize(args, do_distributed=False)

    print_args(args)
    with open(os.path.join(args.save, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))
    args.time_stamp = cur_time
    
    model_cls = {
        "linear_soft_cls": LinearCLSModel,
        "linear_soft_cls_da": LinearCLSModelDynaAlpha,
        # "linear_fa": LinearModelFixAlpha
    }[args.model_type]
    
    model = model_cls(args, device, dim=args.input_dim, real_dim=args.input_real_dim)
    model.set_theta_gd()
    
    if args.load_toy_data:
        data_path = os.path.join(args.base_path, "processed_data/toy_data")
        train_x, train_y, dev_x, dev_y, test_x, test_y, theta_init = torch.load(os.path.join(data_path, "data.pt"))
    else:
        train_x, train_y = model.generate_data(
            args.train_num, args.train_noise, args.train_mu, args.train_sigma)
    
        dev_x, dev_y = model.generate_data(
            args.dev_num, args.dev_noise, args.dev_mu, args.dev_sigma)
        test_x, test_y = model.generate_data(
            args.dev_num, args.dev_noise, args.dev_mu, args.dev_sigma)
        theta_init = None

    model.set_train_data(train_x, train_y)
    model.set_dev_data(dev_x, dev_y)
    # linear_model.set_dev_data(train_x, train_y)
    model.set_test_data(test_x, test_y)
    model.set_init_theta(theta_init)

    torch.save((train_x, train_y, dev_x, dev_y, test_x, test_y, model.theta_init), os.path.join(args.save, "data.pt"))

    model.train()
    
if __name__ == "__main__":
    main()