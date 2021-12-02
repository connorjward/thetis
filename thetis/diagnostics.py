"""
Classes for computing diagnostics.
"""
from .utility import *
from .configuration import *


__all__ = ["VorticityCalculator2D", "HessianRecoverer2D", "DynamicPressureCalculator"]


class VorticityCalculator2D(FrozenHasTraits):
    r"""
    Linear solver for recovering fluid vorticity.

    It is recommended that the vorticity is sought
    in :math:`\mathbb P1` space.
    """
    uv_2d = FiredrakeVectorExpression(
        Constant(as_vector([0.0, 0.0])), help='Horizontal velocity').tag(config=True)

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
        self._isfrozen = False
        self.solver = LinearVariationalSolver(prob, **kwargs)
        self._isfrozen = True

    @PETSc.Log.EventDecorator("thetis.VorticityCalculator2D.solve")
    def solve(self):
        self.solver.solve()


class HessianRecoverer2D(FrozenHasTraits):
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

    @PETSc.Log.EventDecorator("thetis.HessianRecoverer2D.__init__")
    def __init__(self, field_2d, hessian_2d, gradient_2d=None, **kwargs):
        """
        :arg field_2d: scalar expression to recover the Hessian of.
        :arg hessian_2d: :class:`Function` to hold recovered Hessian.
        :kwarg gradient_2d: :class:`Function` to hold recovered gradient.
        :kwargs: to be passed to the :class:`LinearVariationalSolver`.
        """
        self.field_2d = field_2d
        self._isfrozen = False
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
        self._isfrozen = True

    @PETSc.Log.EventDecorator("thetis.HessianRecoverer2D.solve")
    def solve(self):
        self.solver.solve()
        self.hessian_2d.assign(self._hessian)
        if self.gradient_2d is not None:
            self.gradient_2d.assign(self._gradient)


class DynamicPressureCalculator(FrozenHasTraits):
    r"""
    Class for calculating dynamic pressure (i.e. kinetic energy).
    """
    density = FiredrakeScalarExpression(
        Constant(1030.0), help='Fluid density').tag(config=True)

    @PETSc.Log.EventDecorator("thetis.DynamicPressureCalculator.__init__")
    def __init__(self, uv, dp, density=Constant(1030.0)):
        """
        :arg uv: scalar expression for the fluid velocity.
        :arg dp: :class:`Function` to hold calculated dynamic pressure.
        :kwarg density: fluid density.
        """
        self.density = density
        self._isfrozen = False
        self.dp = dp
        self.dp_expr = 0.5*self.density*dot(uv, uv)
        self._isfrozen = True

    @PETSc.Log.EventDecorator("thetis.DynamicPressureCalculator.solve")
    def solve(self):
        self.dp.interpolate(self.dp_expr)


class TracerResidualCalculator2D(object):
    """
    Base class for evaluating residuals related to 2D tracer equations
    as piece-wise constant fields.
    """
    __metaclass__ = ABCMeta

    def __init__(self, label, residual_2d, solver_obj):
        """
        :arg label: the label for the tracer field.
        :arg residual_2d: :class:`Function` to hold the residual.
        :arg solver_obj: :class:`FlowSolver2d` object defining the problem.
        """
        element = residual_2d.ufl_element()
        if not (element.family() == 'Discontinuous Lagrange' and element.degree() == 0):
            raise ValueError(f"Residual should be P0, not {element}")
        self.residual_2d = residual_2d
        if label not in solver_obj.options.tracer:
            raise ValueError(f"{label} is not a valid tracer label")
        if label not in solver_obj.fields:
            raise ValueError(f"{label} is not a valid field label")

    @abstractmethod
    def solve(self):
        pass


class TracerStrongResidualCalculator2D(TracerResidualCalculator2D):
    """
    Class for evaluating the strong residual of a 2D tracer equation
    as a piece-wise constant field.
    """
    @PETSc.Log.EventDecorator("thetis.TracerStrongResidualCalculator2D.__init__")
    def __init__(self, label, residual_2d, solver_obj):
        """
        :arg label: the label for the tracer field.
        :arg residual_2d: :class:`Function` to hold strong residual.
        :arg solver_obj: :class:`FlowSolver2d` object defining the problem.
        """
        super().__(label, residual_2d, solver_obj)
        c = solver_obj.fields[label]

        # Get parameters
        uv = solver_obj.fields.uv_2d
        D = options.tracer[label].diffusivity
        S = options.tracer[label].source

        # Strong residual expression
        self.res_expr = -dot(uv, grad(c))
        if D is not None:
            self.res_expr += div(D*grad(c))
        if S is not None:
            self.res_expr += S

    @PETSc.Log.EventDecorator("thetis.TracerStrongResidualCalculator2D.solve")
    def solve(self):
        self.residual_2d.interpolate(self.res_expr)


class TracerFluxResidualCalculator2D(TracerResidualCalculator2D):
    """
    Class for evaluating the flux residual of a 2D tracer equation
    as a piece-wise constant field.
    """
    @PETSc.Log.EventDecorator("thetis.TracerFluxResidualCalculator2D.__init__")
    def __init__(self, label, residual_2d, solver_obj):
        """
        :arg label: the label for the tracer field.
        :arg residual_2d: :class:`Function` to hold flux residual.
        :arg solver_obj: :class:`FlowSolver2d` object defining the problem.
        """
        super().__(label, residual_2d, solver_obj)
        c = solver_obj.fields[label]

        # Get parameters
        uv = solver_obj.fields.uv_2d
        D = options.tracer[label].diffusivity
        S = options.tracer[label].source

        # Flux residual expression
        self.res_expr = 0
        raise NotImplementedError  # TODO

    @PETSc.Log.EventDecorator("thetis.TracerFluxResidualCalculator2D.solve")
    def solve(self):
        raise NotImplementedError  # TODO
