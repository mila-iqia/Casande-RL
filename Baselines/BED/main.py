import argparse
import logging
from bayesian_experimental_design import BED, param_search
from bayesian_experimental_design_batch import BEDBatch
from cheater import Cheater


def parse_arguments():
    parser = argparse.ArgumentParser(description='Arguments for QMR model')
    # Data path arguments
    parser.add_argument('--input_dir', type=str, default='dataset')
    parser.add_argument('--output_dir', type=str, default='output')

    # Data set arguments
    parser.add_argument('--dataset_name', type=str, default='SymCAT')
    parser.add_argument('--n_diseases', type=int, default=200)

    # Common arguments
    parser.add_argument('--solver', type=str, required=True)
    parser.add_argument('--test_size', type=int, default=10000)
    parser.add_argument('--max_episode_len', type=int, default=10)
    parser.add_argument('--param_search', action='store_true')
    parser.add_argument('--no_patho_restriction', action='store_true')
    # Bayesian experimental design arguments
    parser.add_argument('--utility_func', type=str, default='KL',
                        help="Choose from (`SI` and `KL`), case insensitive.")
    parser.add_argument('--threshold', type=float, default=0.05)
    parser.add_argument('--search_depth', type=int, default=1)

    # Cheater arguments
    parser.add_argument('--cheater_method', type=str, default='inference')

    args = parser.parse_args()
    return args


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()
    args.solver = args.solver.lower()
    if args.solver == 'bed' or args.solver == 'bed_batch' or args.solver == 'cheater':
        args.single_diag_action = True
        args.mask_actions = False
        args.aux_reward = False

    if args.solver == 'bed' and args.param_search:
        param_search(args)
        return

    if args.solver == 'bed':
        solver = BED(args)
    elif args.solver == 'bed_batch':
        solver = BEDBatch(args)
    elif args.solver == 'cheater':
        solver = Cheater(args)
    else:
        raise NotImplementedError(f'No solver name {args.solver} is defined.')
    solver.run()


if __name__ == '__main__':
    main()
