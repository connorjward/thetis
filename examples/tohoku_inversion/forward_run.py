from thetis import *
import time as time_mod
from model_config import *
import argparse


parser = argparse.ArgumentParser(
    description='Tohoku tsunami propagation',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('source_model',
                    help='Source model to use',
                    choices=['CG1', 'DG0'])
args = parser.parse_args()
source_model = args.source_model

solver_obj = construct_solver(output_directory='outputs_forward')
mesh2d = solver_obj.mesh2d
elev_init = initial_condition(mesh2d, source_model=source_model)
elev = Function(elev_init.function_space())
elev.project(mask(mesh2d)*elev_init)
print_output(f'Exporting to {solver_obj.options.output_directory}')
solver_obj.assign_initial_conditions(elev=elev)
tic = time_mod.perf_counter()
solver_obj.iterate()
toc = time_mod.perf_counter()
print_output(f'Total duration: {toc-tic:.2f} seconds')
