from thetis import *
from firedrake_adjoint import *
import thetis.inversion_tools as inversion_tools
from model_config import *
import argparse
import numpy


# Parse user input
parser = argparse.ArgumentParser(
    description='Tohoku tsunami source inversion problem',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('source_model',
                    help='Source model to use',
                    choices=['CG1', 'DG0'])
parser.add_argument('--no-consistency-test', action='store_true',
                    help='Skip consistency test')
parser.add_argument('--no-taylor-test', action='store_true',
                    help='Skip Taylor test')
args = parser.parse_args()
source_model = args.source_model
do_consistency_test = not args.no_consistency_test
do_taylor_test = not args.no_taylor_test

# Setup PDE
solver_obj = construct_solver(
    output_directory='outputs_elev-init-optimization',
    store_station_time_series=False,
)
mesh2d = solver_obj.mesh2d
elev_init = initial_condition(mesh2d, source_model=source_model)
elev = Function(elev_init.function_space(), name='Elevation')
elev.project(mask(mesh2d)*elev_init)
options = solver_obj.options
mesh2d = solver_obj.mesh2d
bathymetry_2d = solver_obj.fields.bathymetry_2d

# Assign initial conditions
print_output(f'Exporting to {options.output_directory}')
solver_obj.assign_initial_conditions(elev=elev)

# Choose optimisation parameters
control = 'elev_init'
control_bounds = [-numpy.inf, numpy.inf]
op = inversion_tools.OptimisationProgress(options.output_directory)
op.add_control(elev_init)
gamma_hessian_list = [Constant(0.1)]

# Define the (appropriately scaled) cost function
dt_const = Constant(solver_obj.dt)
total_time_const = Constant(options.simulation_end_time)
J_scalar = dt_const/total_time_const

# Set up observation and regularization managers
observation_data_dir = 'observations'
variable = 'elev'
station_names = list(stations.keys())
start_times = [dat['interval'][0] for sta, dat in stations.items()]
end_times = [dat['interval'][1] for sta, dat in stations.items()]
stationmanager = inversion_tools.StationObservationManager(
    mesh2d, J_scalar=J_scalar, output_directory=options.output_directory
)
stationmanager.load_observation_data(observation_data_dir, station_names, variable,
                                     start_times=start_times, end_times=end_times)
stationmanager.set_model_field(solver_obj.fields.elev_2d)
reg_manager = inversion_tools.ControlRegularizationManager(
    op.control_coeff_list, gamma_hessian_list, J_scalar=J_scalar,
)


def cost_function():
    """
    Compute square misfit between data and observations.
    """
    t = solver_obj.simulation_time

    J_misfit = stationmanager.eval_cost_function(t)
    op.J += J_misfit


def gradient_eval_callback(j, djdm, m):
    """
    Stash optimisation state.
    """
    op.set_control_state(j, djdm, m)
    op.nb_grad_evals += 1


# compute regularization term
op.J = reg_manager.eval_cost_function()

# Solver and setup reduced functional
solver_obj.iterate(export_func=cost_function)
Jhat = ReducedFunctional(op.J, op.control_list, derivative_cb_post=gradient_eval_callback)
pause_annotation()

if do_consistency_test:
    print_output('Running consistency test')
    J = Jhat(op.control_coeff_list)
    assert numpy.isclose(J, op.J)
    print_output('Consistency test passed!')

if do_taylor_test:
    func_list = []
    for f in op.control_coeff_list:
        dc = Function(f.function_space()).assign(f)
        func_list.append(dc)
    minconv = taylor_test(Jhat, op.control_coeff_list, func_list)
    assert minconv > 1.9
    print_output('Taylor test passed!')


def optimisation_callback(m):
    """
    Stash optimisation progress after successful line search.
    """
    op.update_progress()
    stationmanager.dump_time_series()


# Run inversion
opt_method = 'L-BFGS-B'
opt_verbose = -1
opt_options = {
    'maxiter': 100,
    'ftol': 1.0e-05,
    'disp': opt_verbose if mesh2d.comm.rank == 0 else -1,
}
print_output(f'Running {opt_method} optimisation')
op.reset_counters()
op.start_clock()
J = float(Jhat(op.control_coeff_list))
op.set_initial_state(J, Jhat.derivative(), op.control_coeff_list)
control_opt = minimize(
    Jhat, method=opt_method, bounds=control_bounds,
    callback=optimisation_callback, options=opt_options,
)
op.stop_clock()
control_opt.rename(op.control_coeff_list[0].name())
print_function_value_range(control_opt, prefix='Optimal')
File(f'{options.output_directory}/{op.control_coefficient.name()}_optimised.pvd').write(control_opt)
print_output(f'Total runtime: {op.toc - op.tic} seconds')
