from thetis import *
import time as time_mod
from model_config import construct_solver

solver_obj = construct_solver(output_directory='outputs_forward')
elev_init_2d = solver_obj.fields.elev_init_2d

print_output('Exporting to ' + solver_obj.options.output_directory)
solver_obj.assign_initial_conditions(elev=elev_init_2d, uv=Constant((1e-5, 0)))
tic = time_mod.perf_counter()
solver_obj.iterate()
toc = time_mod.perf_counter()
print_output(f'Total duration: {toc-tic:.2f}')
