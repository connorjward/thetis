# Rhine ROFI test case
# ====================
#
# Idealized Rhine river plume test case according to [2].
#
# Salinity ranges from 32 psu in ocean to 0 in river.
# Temperature is constant 10 deg Celcius. This corresponds to density
# values 1024.611 kg/m3 in the ocean and 999.702 in the river.
#
# [1] de Boer, G., Pietrzak, J., and Winterwerp, J. (2006). On the vertical
#     structure of the Rhine region of freshwater influence. Ocean Dynamics,
#     56(3):198-216.
# [2] Fischer, E., Burchard, H., and Hetland, R. (2009). Numerical
#     investigations of the turbulent kinetic energy dissipation rate in the
#     Rhine region of freshwater influence. Ocean Dynamics, 59:629-641.
#
# Tuomas Karna 2015-06-24

from cofs import *

outputdir = create_directory('outputs')
layers = 6
mesh2d = Mesh('mesh_rhineRofi_coarse.msh')
print_info('Loaded mesh '+mesh2d.name)
print_info('Exporting to '+outputdir)

# Physical parameters
etaAmplitude = 1.00  # mean (Fisher et al. 2009 tidal range 2.00 )
etaPhase = 0
H = 20  # water depth
HInlet = 5  # water depth at river inlet
Lriver = 45e3
Qriver = 3.0e3  # 1.5e3 river discharge (Fisher et al. 2009)
Sriver = 0
Ssea = 32
density_river = 999.7
density_ocean = 1024.6
Ttide = 44714.0  # M2 tidal period (Fisher et al. 2009)
Tday = 0.99726968*24*60*60  # sidereal time of Earth revolution
OmegaEarth = 2*np.pi/Tday
OmegaTide = 2*np.pi/Ttide
g = physical_constants['g_grav']
c = sqrt(g*H)  # [m/s] wave speed
latDeg = 52.5  # latitude
phi = (np.pi/180)*latDeg  # latitude in radians
coriolisF = 2*OmegaEarth*sin(phi)  # [rad/s] Coriolis parameter ~ 1.1e-4
kelvinK = OmegaTide/c  # [1/m] initial wave number of tidal wave, no friction
kelvinM = (coriolisF/c)  # [-] Cross-shore variation

dt = 8.0
T = 32*44714
TExport = 900.0  # 44714/12

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.interpolate(Expression('(x[0] > 0.0) ? H*(1-x[0]/Lriver) + HInlet*(x[0]/Lriver) : H',
                                     H=H, HInlet=HInlet, Lriver=Lriver))

# create solver
solverObj = solver.FlowSolver(mesh2d, bathymetry_2d, layers)
options = solverObj.options
options.cfl_2d = 1.0
# options.nonlin = False
options.solveSalt = True
options.solveVertDiffusion = False
options.useBottomFriction = False
options.useALEMovingMesh = False
# options.useSemiImplicit2D = False
# options.useModeSplit = False
options.baroclinic = True
options.coriolis = Constant(coriolisF)
options.useSUPG = False
options.useGJV = False
options.uvLaxFriedrichs = Constant(1.0)
options.tracerLaxFriedrichs = Constant(1.0)
Re_h = 2.0
options.smagorinskyFactor = Constant(1.0/np.sqrt(Re_h))
options.salt_jump_diffFactor = Constant(1.0)
options.saltRange = Constant(25.0)
# To keep const grid Re_h, viscosity scales with grid: nu = U dx / Re_h
# options.hViscosity = Constant(0.5*2000.0/refinement[reso_str]/Re_h)
# To keep const grid Re_h, viscosity scales with grid: nu = U dx / Re_h
# options.hViscosity = Constant(100.0/refinement[reso_str])
# options.hViscosity = Constant(10.0)
if options.useModeSplit:
    options.dt = dt
options.TExport = TExport
options.T = T
options.outputdir = outputdir
options.uAdvection = Constant(2.0)
options.checkVolConservation2d = True
options.checkVolConservation3d = True
options.checkSaltConservation = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'uv_3d',
                            'w_3d', 'w_mesh_3d', 'salt_3d',
                            'uv_dav_2d', 'uv_dav_3d', 'baroc_head_3d',
                            'baroc_head_2d', 'gjv_alpha_h_3d', 'gjv_alpha_v_3d',
                            'smag_visc_3d', 'salt_jump_diff']
options.timerLabels = []

bnd_elev = Function(P1_2d, name='Boundary elevation')
bnd_time = Constant(0)
xyz = solverObj.mesh2d.coordinates
tri = TrialFunction(P1_2d)
test = TestFunction(P1_2d)
elev = etaAmplitude*exp(xyz[0]*kelvinM)*cos(xyz[1]*kelvinK - OmegaTide*bnd_time)
a = inner(test, tri)*P1_2d.mesh()._dx
L = test*elev*P1_2d.mesh()._dx
bndElevProb = LinearVariationalProblem(a, L, bnd_elev)
bndElevSolver = LinearVariationalSolver(bndElevProb)
bndElevSolver.solve()

fs = P1_2d
bnd_v = Function(fs, name='Boundary v velocity')
tri = TrialFunction(fs)
test = TestFunction(fs)
v = -(g*kelvinK/OmegaTide)*etaAmplitude*exp(xyz[0]*kelvinM)*cos(xyz[1]*kelvinK - OmegaTide*bnd_time)
a = inner(test, tri)*fs.mesh()._dx
L = test*v*fs.mesh()._dx
bndVProb = LinearVariationalProblem(a, L, bnd_v)
bndVSolver = LinearVariationalSolver(bndVProb)
bndVSolver.solve()

river_discharge = Constant(-Qriver)
ocean_salt = Constant(density_ocean)
river_salt = Constant(density_river)
tide_elev_funcs = {'elev': bnd_elev}
tide_uv_funcs = {'un': bnd_v}
open_funcs = {'symm': None}
river_funcs = {'flux': river_discharge}
bnd_ocean_salt = {'value': ocean_salt}
bnd_river_salt = {'value': river_salt}
solverObj.bnd_functions['shallow_water'] = {1: tide_elev_funcs, 2: tide_elev_funcs,
                                            3: tide_elev_funcs, 6: river_funcs}
# solverObj.bnd_functions['momentum'] = {1: tide_funcs, 2: tide_funcs,
#                                        3: tide_funcs, 6: river_funcs}
solverObj.bnd_functions['salt'] = {1: bnd_ocean_salt, 2: bnd_ocean_salt,
                                   3: bnd_ocean_salt, 6: bnd_river_salt}

solverObj.createEquations()
bnd_elev_3d = Function(solverObj.function_spaces.P1, name='Boundary elevation 3d')
copy2dFieldTo3d(bnd_elev, bnd_elev_3d)
tide_elev_funcs_3d = {'elev': bnd_elev_3d}
solverObj.eq_momentum.bnd_functions = {1: tide_elev_funcs_3d, 2: tide_elev_funcs_3d,
                                       3: tide_elev_funcs_3d, 6: river_funcs}

elev_init = Function(solverObj.function_spaces.H_2d, name='initial elevation')
elev_init.interpolate(Expression('(x[0]<=0) ? amp*exp(x[0]*kelvinM)*cos(x[1]*kelvinK) : amp*cos(x[1]*kelvinK)',
                      amp=etaAmplitude, kelvinM=kelvinM, kelvinK=kelvinK))
elev_init2 = Function(solverObj.function_spaces.H_2d, name='initial elevation')
elev_init2.interpolate(Expression('(x[0]<=0) ? amp*exp(x[0]*kelvinM)*cos(x[1]*kelvinK) : 0.0',
                       amp=etaAmplitude, kelvinM=kelvinM, kelvinK=kelvinK))
uv_init = Function(solverObj.function_spaces.U_2d, name='initial velocity')
# uv_init.interpolate(Expression('(x[0]<=0) ? amp*exp(x[0]*kelvinM)*cos(x[1]*kelvinK) : amp*cos(x[1]*kelvinK)',
#                       amp=etaAmplitude, kelvinM=kelvinM, kelvinK=kelvinK))
tri = TrialFunction(solverObj.function_spaces.U_2d)
test = TestFunction(solverObj.function_spaces.U_2d)
a = inner(test, tri)*solverObj.eq_sw.dx
uv = (g*kelvinK/OmegaTide)*elev_init2
L = test[1]*uv*solverObj.eq_sw.dx
solve(a == L, uv_init)
salt_init3d = Function(solverObj.function_spaces.H, name='initial salinity')
salt_init3d.interpolate(Expression('d_ocean - (d_ocean - d_river)*(1 + tanh((x[0] - xoff)/sigma))/2',
                                   sigma=6000.0, d_ocean=density_ocean,
                                   d_river=density_river, xoff=20.0e3))


def update_forcings(t):
    bnd_time.assign(t)
    bndElevSolver.solve()
    copy2dFieldTo3d(bnd_elev, bnd_elev_3d)

solverObj.assignInitialConditions(elev=elev_init, salt=salt_init3d, uv_2d=uv_init)
solverObj.iterate()
