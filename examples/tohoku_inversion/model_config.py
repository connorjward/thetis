from thetis import *
import netCDF4
import os
import scipy.interpolate as si
import utm


__all__ = ['stations', 'initial_condition', 'mask', 'construct_solver']


stations = {
    '801': {'latlon': (38.2325, 141.6856), 'interval': (0.0, 3600.0)},
    '802': {'latlon': (39.2586, 142.0969), 'interval': (0.0, 3600.0)},
    '803': {'latlon': (38.8578, 141.8944), 'interval': (0.0, 3600.0)},
    '804': {'latlon': (39.6272, 142.1867), 'interval': (0.0, 3600.0)},
    '806': {'latlon': (36.9714, 141.1856), 'interval': (0.0, 3600.0)},
    '807': {'latlon': (40.1167, 142.0667), 'interval': (0.0, 3600.0)},
    'P02': {'latlon': (38.5002, 142.5016), 'interval': (0.0, 3600.0)},
    'P06': {'latlon': (38.6340, 142.5838), 'interval': (0.0, 3600.0)},
    'KPG1': {'latlon': (41.7040, 144.4375), 'interval': (0.0, 3600.0)},
    'KPG2': {'latlon': (42.2365, 144.8485), 'interval': (0.0, 3600.0)},
    'MPG1': {'latlon': (32.3907, 134.4753), 'interval': (4800.0, 7200.0)},
    'MPG2': {'latlon': (32.6431, 134.3712), 'interval': (4800.0, 7200.0)},
    '21401': {'latlon': (42.617, 152.583), 'interval': (3000.0, 7200.0)},
    '21413': {'latlon': (30.533, 152.132), 'interval': (3000.0, 7200.0)},
    '21418': {'latlon': (38.735, 148.655), 'interval': (0.0, 3600.0)},
    '21419': {'latlon': (44.435, 155.717), 'interval': (3000.0, 7200.0)},
}


def initial_condition(mesh2d, source_model='CG1'):
    """
    Construct an initial condition :class:`Function` for the chosen
    source model.

    Choices:
      - 'CGp': Piece-wise polynomial (order p) and continuous.
      - 'DGp': Piece-wise polynomial (order p) and discontinuous.

    :arg mesh2d: the underlying mesh
    :kwarg source_model: method for approximating the tsunami source
    """
    x, y = SpatialCoordinate(mesh2d)
    x0, y0 = 700.0e+03, 4200.0e+03
    X = as_vector([x - x0, y - y0])
    w, l = 2*56.0e+03, 2*24.0e+03
    h = 8.0
    theta = 7*pi/12
    R = as_matrix([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    xx, yy = dot(R, X)
    if source_model[:2] in ('CG', 'DG'):
        family = source_model[:2]
        degree = int(source_model[2:])
        elev_init = Function(get_functionspace(mesh2d, family, degree))
        elev_init.interpolate(h*exp(-((xx/w)**2 + (yy/l)**2)))
    else:
        raise NotImplementedError  # TODO
    return elev_init


def mask(mesh2d):
    """
    Mask to apply to the initial surface so that is constrained
    to only be non-zero within a particular region.
    """
    x, y = SpatialCoordinate(mesh2d)
    x0, y0 = 700.0e+03, 4200.0e+03
    X = as_vector([x - x0, y - y0])
    # w, l = 560.0e+03, 240.0e+03
    # theta = 7*pi/12
    # R = as_matrix([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    # xx, yy = dot(R, X)
    P0_2d = get_functionspace(mesh2d, 'DG', 0)
    # return interpolate(conditional(And(And(xx > -w/2, xx < w/2), And(yy > -l/2, yy < l/2)), 1, 0), P0_2d)
    r = 200.0e+03
    xx, yy = X
    return interpolate(conditional(xx**2 + yy**2 < r**2, 1, 0), P0_2d)


def interpolate_bathymetry(bathymetry_2d, cap=30.0):
    """
    Interpolate a bathymetry field from the ETOPO1 data set.

    :arg bathymetry_2d: :class:`Function` to store the data in
    :kwarg cap: minimum value to cap the bathymetry at in the shallows
    """
    if cap <= 0.0:
        raise NotImplementedError("Wetting and drying is not enabled in this example")
    mesh = bathymetry_2d.function_space().mesh()

    # Read data from file
    cwd = os.path.dirname(__file__)
    with netCDF4.Dataset(os.path.join(cwd, 'etopo1.nc'), 'r') as nc:
        lon = nc.variables['lon'][:]
        lat = nc.variables['lat'][:]
        elev = nc.variables['Band1'][:, :]

    # Interpolate at mesh vertices
    interp = si.RectBivariateSpline(lat, lon, elev)
    for i, xy in enumerate(mesh.coordinates.dat.data_ro_with_halos):
        lat, lon = utm.to_latlon(xy[0], xy[1], 54, northern=True, strict=False)
        bathymetry_2d.dat.data_with_halos[i] -= min(interp(lat, lon), -30)


def construct_solver(store_station_time_series=True, **model_options):
    """
    Construct a *linear* shallow water equation solver for tsunami
    propagation modelling.
    """
    mesh2d = Mesh('japan_sea.msh')

    t_end = 2*3600.0
    u_mag = Constant(5.0)
    t_export = 60.0
    dt = 60.0

    # Bathymetry
    P1_2d = get_functionspace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(P1_2d, name='Bathymetry')
    interpolate_bathymetry(bathymetry_2d)

    # Create solver
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    options = solver_obj.options
    options.use_nonlinear_equations = False
    options.element_family = 'dg-dg'
    options.simulation_export_time = t_export
    options.simulation_end_time = t_end
    options.horizontal_velocity_scale = u_mag
    options.swe_timestepper_type = 'CrankNicolson'
    if not hasattr(options.swe_timestepper_options, 'use_automatic_timestep'):
        options.timestep = dt
    options.swe_timestepper_options.solver_parameters = {
        'snes_type': 'newtonls',
        'ksp_type': 'gmres',
        'pc_type': 'bjacobi',
        'sub_pc_type': 'ilu',
    }
    options.update(model_options)
    solver_obj.create_equations()

    # Set up gauges
    if store_station_time_series:
        for name, data in stations.items():
            sta_lat, sta_lon = data['latlon']
            tstart, tend = data['interval']
            sta_x, sta_y = utm.from_latlon(sta_lat, sta_lon, force_zone_number=54)[:2]
            cb = TimeSeriesCallback2D(
                solver_obj, ['elev_2d'], sta_x, sta_y, name,
                append_to_log=False, start_time=tstart, end_time=tend,
            )
            solver_obj.add_callback(cb)

    # Set boundary conditions
    zero = Constant(0.0)
    solver_obj.bnd_functions['shallow_water'] = {
        100: {'un': zero, 'elev': zero},
        200: {'un': zero},
        300: {'un': zero},
    }
    return solver_obj
