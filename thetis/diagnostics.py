"""
Classes for computing diagnostics.
"""
from .utility import *
from .configuration import *


__all__ = ["VorticityCalculator2D", "HessianRecoverer2D", "KineticEnergyCalculator"]


class DiagnosticCalculator(FrozenHasTraits):
    """
    Base class that defines the API for all diagnostic calculators.
    """
    __metaclass__ = ABCMeta

    def __call__(self):
        self.solve()

    @abstractmethod
    def solve(self):
        pass


class VorticityCalculator2D(DiagnosticCalculator):
    r"""
    Linear solver for recovering fluid vorticity,
    interpreted as a scalar field:

    .. math::
        \omega = -v_x + u_y,

    for a velocity field :math:`\mathbf{u} = (u, v)`.

    It is recommended that the vorticity is sought
    in :math:`\mathbb P1` space.
    """
    uv_2d = FiredrakeVectorExpression(
        Constant(as_vector([0.0, 0.0])), help='Horizontal velocity').tag(config=True)

    @unfrozen
    @PETSc.Log.EventDecorator("thetis.VorticityCalculator2D.__init__")
    def __init__(self, uv_2d, vorticity_2d, **kwargs):
        """
        :arg uv_2d: vector expression for the horizontal velocity.
        :arg vorticity_2d: :class:`Function` to hold calculated vorticity.
        :kwargs: to be passed to the :class:`LinearVariationalSolver`.
        """
        self.uv_2d = uv_2d
        fs = vorticity_2d.function_space()
        dim = fs.mesh().topological_dimension()
        if dim != 2:
            raise ValueError(f'Dimension {dim} not supported')
        if element_continuity(fs.ufl_element()).horizontal != 'cg':
            raise ValueError('Vorticity must be calculated in a continuous space')
        n = FacetNormal(fs.mesh())

        # Weak formulation
        test = TestFunction(fs)
        a = TrialFunction(fs)*test*dx
        L = -inner(perp(self.uv_2d), grad(test))*dx \
            + dot(perp(self.uv_2d), test*n)*ds \
            + dot(avg(perp(self.uv_2d)), jump(test, n))*dS

        # Setup vorticity solver
        prob = LinearVariationalProblem(a, L, vorticity_2d)
        kwargs.setdefault('solver_parameters', {
            "ksp_type": "cg",
            "pc_type": "bjacobi",
            "sub_pc_type": "ilu",
        })
        self.solver = LinearVariationalSolver(prob, **kwargs)

    @PETSc.Log.EventDecorator("thetis.VorticityCalculator2D.solve")
    def solve(self):
        self.solver.solve()


class HessianRecoverer2D(DiagnosticCalculator):
    r"""
    Linear solver for recovering Hessians.

    Hessians are recoved using double :math:`L^2`
    projection, which is implemented using a
    mixed finite element method.

    It is recommended that gradients and Hessians
    are sought in :math:`\mathbb P1` space of
    appropriate dimension.
    """
    field_2d = FiredrakeScalarExpression(
        Constant(0.0), help='Field to be recovered').tag(config=True)

    @unfrozen
    @PETSc.Log.EventDecorator("thetis.HessianRecoverer2D.__init__")
    def __init__(self, field_2d, hessian_2d, gradient_2d=None, **kwargs):
        """
        :arg field_2d: scalar expression to recover the Hessian of.
        :arg hessian_2d: :class:`Function` to hold recovered Hessian.
        :kwarg gradient_2d: :class:`Function` to hold recovered gradient.
        :kwargs: to be passed to the :class:`LinearVariationalSolver`.
        """
        self.field_2d = field_2d
        self.hessian_2d = hessian_2d
        self.gradient_2d = gradient_2d
        Sigma = hessian_2d.function_space()
        mesh = Sigma.mesh()
        dim = mesh.topological_dimension()
        if dim != 2:
            raise ValueError(f'Dimension {dim} not supported')
        n = FacetNormal(mesh)

        # Extract function spaces
        if element_continuity(Sigma.ufl_element()).horizontal != 'cg':
            raise ValueError('Hessians must be calculated in a continuous space')
        if Sigma.dof_dset.dim != (2, 2):
            raise ValueError('Expecting a square tensor function')
        if gradient_2d is None:
            V = get_functionspace(mesh, 'CG', 1, vector=True)
        else:
            V = gradient_2d.function_space()
            if element_continuity(V.ufl_element()).horizontal != 'cg':
                raise ValueError('Gradients must be calculated in a continuous space')
            if V.dof_dset.dim != (2,):
                raise ValueError('Expecting a 2D vector function')

        # Setup function spaces
        W = V*Sigma
        g, H = TrialFunctions(W)
        phi, tau = TestFunctions(W)
        sol = Function(W)
        self._gradient, self._hessian = sol.split()

        # The formulation is chosen such that f does not need to have any
        # finite element derivatives
        a = inner(tau, H)*dx \
            + inner(div(tau), g)*dx \
            + inner(phi, g)*dx \
            - dot(g, dot(tau, n))*ds \
            - dot(avg(g), jump(tau, n))*dS
        L = self.field_2d*dot(phi, n)*ds \
            + avg(self.field_2d)*jump(phi, n)*dS \
            - self.field_2d*div(phi)*dx

        # Apply stationary preconditioners in the Schur complement to get away
        # with applying GMRES to the whole mixed system
        sp = {
            "mat_type": "aij",
            "ksp_type": "gmres",
            "ksp_max_it": 20,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_0_fields": "1",
            "pc_fieldsplit_1_fields": "0",
            "pc_fieldsplit_schur_precondition": "selfp",
            "fieldsplit_0_ksp_type": "preonly",
            "fieldsplit_1_ksp_type": "preonly",
            "fieldsplit_1_pc_type": "gamg",
            "fieldsplit_1_mg_levels_ksp_max_it": 5,
        }
        if COMM_WORLD.size == 1:
            sp["fieldsplit_0_pc_type"] = "ilu"
            sp["fieldsplit_1_mg_levels_pc_type"] = "ilu"
        else:
            sp["fieldsplit_0_pc_type"] = "bjacobi"
            sp["fieldsplit_0_sub_ksp_type"] = "preonly"
            sp["fieldsplit_0_sub_pc_type"] = "ilu"
            sp["fieldsplit_1_mg_levels_pc_type"] = "bjacobi"
            sp["fieldsplit_1_mg_levels_sub_ksp_type"] = "preonly"
            sp["fieldsplit_1_mg_levels_sub_pc_type"] = "ilu"

        # Setup solver
        prob = LinearVariationalProblem(a, L, sol)
        kwargs.setdefault('solver_parameters', sp)
        self.solver = LinearVariationalSolver(prob, **kwargs)

    @PETSc.Log.EventDecorator("thetis.HessianRecoverer2D.solve")
    def solve(self):
        self.solver.solve()
        self.hessian_2d.assign(self._hessian)
        if self.gradient_2d is not None:
            self.gradient_2d.assign(self._gradient)


class KineticEnergyCalculator(DiagnosticCalculator):
    r"""
    Class for calculating dynamic pressure (i.e. kinetic energy),

    .. math::
        E_K = \frac12 \rho \|\mathbf{u}\|^2,

    where :math:`\rho` is the water density and :math:`\mathbf{u}`
    is the velocity.
    """
    density = FiredrakeScalarExpression(
        physical_constants['rho0'], help='Fluid density').tag(config=True)

    @unfrozen
    @PETSc.Log.EventDecorator("thetis.KineticEnergyCalculator.__init__")
    def __init__(self, uv, ke, density=None, horizontal=False, project=False):
        """
        :arg uv: scalar expression for the fluid velocity.
        :arg ke: :class:`Function` to hold calculated kinetic energy.
        :kwarg density: fluid density.
        :kwarg horizontal: consider the horizontal components of velocity only?
        :kwarg project: project, rather than interpolate?
        """
        if density is not None:
            self.density = density
        self.ke = ke
        u_sq = uv[0]*uv[0] + uv[1]*uv[1] if horizontal else dot(uv, uv)
        self.ke_expr = 0.5*self.density*u_sq
        if project:
            self.projector = Projector(self.ke_expr, self.ke)
        else:
            self.interpolator = Interpolator(self.ke_expr, self.ke)

    @PETSc.Log.EventDecorator("thetis.KineticEnergyCalculator.solve")
    def solve(self):
        if hasattr(self, 'projector'):
            self.projector.project()
        else:
            assert hasattr(self, 'interpolator')
            self.interpolator.interpolate()


class ErrorEstimator(DiagnosticCalculator):
    """
    Base class for estimating errors.
    """
    __metaclass__ = ABCMeta

    @unfrozen
    def __init__(self, solver_obj, norm_type='L2'):
        """
        :arg mesh: mesh upon which the problem is discretized
        :arg options: :class:`CommonModelOptions` object
        :kwarg norm_type: choose from 'L1' and 'L2'
        """
        self.solver_obj = solver_obj
        self.options = solver_obj.options
        if norm_type not in ('L1', 'L2'):
            raise ValueError(f'Norm type {norm_type} not supported')
        self.norm_type = norm_type

    @abstractmethod
    def residual(self):
        pass

    @abstractmethod
    def fluxes(self):
        pass


class TracerErrorEstimator2D(ErrorEstimator):
    """
    Class for estimating errors related to 2D tracer
    transport problems.
    """
    def __init__(self, mesh, options, name, norm_type='L2'):
        """
        :arg mesh: mesh upon which the problem is discretized
        :arg options: :class:`CommonModelOptions` object
        :arg name: label used for the tracer field
        :kwarg norm_type: choose from 'L1' and 'L2'
        """
        super().__init__(solver_obj, norm_type=norm_type)
        self.timestepper = self.solver_obj.timestepper.timesteppers[name]

    @property
    def residual(self):
        assert hasattr(self.timestepper, 'R')
        return assemble(self.timestepper.R)

    # TODO: fluxes
