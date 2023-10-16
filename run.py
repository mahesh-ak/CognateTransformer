from src.trainer import *
import sys


if len(sys.argv) != 2 or sys.argv[1] not in ['reflex','proto-cog', 'proto-nmt']:
    print("Correct usage:")
    print("python run.py <method>")
    print("<method> = 'reflex' | 'proto-cog' | 'proto-nmt'")
else:
    method = sys.argv[1]
    if method == 'reflex':
        train_evaluate_reflex()
    elif method == 'proto-cog':
        train_eval_proto_cog()
    elif method == 'proto-nmt':
        train_eval_proto_nmt()