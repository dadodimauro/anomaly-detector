# TODO: add parser
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        

parser = argparse.ArgumentParser(description='Multivariate Time Series Anomaly Detection')

parser.add_argument('--type',
                    '-t',
                    type=str,
                    required=False,
                    default='multivariate',
                    help="Type of analysis from ['multivariate', 'univariate']"
                   )
parser.add_argument('--single_file',
                    '-s',
                    type=str2bool,
                    required=False,
                    nargs='?',
                    const=True, 
                    default=True,
                    help="Use a single file or two separate files for train and test"
                   )
parser.add_argument('--auto',
                    '-a',
                    type=str2bool,
                    required=False,
                    nargs='?',
                    const=True, 
                    default=False,
                    help="Use an automatic method do divide data into test-train or specify manually the intervals"
                   )
parser.add_argument('--params_path',
                    '-p',
                    type=str,
                    required=False,
                    default='src/params.yaml',
                    help="Path of the params file (see 'src/params.yaml' for the structure)"
                   )
# ATTENTION: leave it always to False
parser.add_argument('--show_plots',
                    type=str2bool,
                    required=False,
                    nargs='?',
                    const=True, 
                    default=False,
                    help=""
                   )

args = parser.parse_args()