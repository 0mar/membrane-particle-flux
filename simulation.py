import numpy as np
import fenics as fe
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import mshr


class MembraneSimulator:
    """
    Solve the homogenized membrane problem in two dimensions.
    For more details, check the manuscript.
    """
    length = 1
    w = 0.25
    b = 2
    d1 = 1
    d2 = 1
    eta = 0.2
    alpha = 3

    def __init__(self):
        self.h = .4

        # Cell properties
        self.obs_length = 0.7
        self.obs_height = 0.1
        self.obs_ratio = self.obs_height / self.obs_length

        if self.h < self.eta:
            raise ValueError("Eta must be smaller than h")
        self.ref_T = 0.1

        self.eps = 2 * self.eta / MembraneSimulator.length
        self.mesh = fe.RectangleMesh(fe.Point(-1, 0), fe.Point(1, 2 * self.h / MembraneSimulator.length), 50, 50)
        self.cell_mesh = None
        self.D = np.matrix(((4 * self.ref_T * MembraneSimulator.d1 / MembraneSimulator.length ** 2, 0),
                            (0, 4 * self.ref_T * MembraneSimulator.d2 / MembraneSimulator.length ** 2)))
        # Dirichlet boundary conditions
        self.u_l = 6E-5
        self.u_r = 0
        self.u_boundary = fe.Expression('(u_l*(x[0] < - par) + u_r*(x[0] >= par))', u_l=self.u_l, u_r=self.u_r,
                                        par=MembraneSimulator.w / MembraneSimulator.length, degree=2)
        self.u_0 = self.u_boundary / 2

        # self.rhs = fe.Expression('(x[1]>par)*u',u=self.u_l*4,par=MembraneSimulator.eta/MembraneSimulator.length, degree=2)
        self.rhs = fe.Expression('-0*u',u=self.u_l,degree=2)
        self.time = 0
        self.T = 0.2
        self.dt = 0.005
        self.tol = 1E-4

        self.function_space = fe.FunctionSpace(self.mesh, 'P', 2)
        self.solution = fe.Function(self.function_space)
        self.cell_solutions = self.cell_fs = None
        self.unit_vectors = [fe.Constant((1., 0.)), fe.Constant((0., 1.))]
        self.eff_diff = np.zeros((2, 2))
        self.file = fe.File('results/solution.pvd')

    @staticmethod
    def drift_function(u):
        """
        Arbitrary polynomial
        :param u:
        :return:
        """
        # return fe.Constant(((1, 0), (0, 0))) * fe.grad(u * (1 - u))
        return -MembraneSimulator.b * MembraneSimulator.length / (2 * MembraneSimulator.d1) * fe.Constant(
            (1, MembraneSimulator.alpha)) * u * (1 - u)

    def solve_cell_problems(self, method='regularized', plot=False):
        """
        Solve the cell problems for the given PDE by changing the geometry to exclude the zone in the middle
        :return:
        """

        class PeriodicBoundary(fe.SubDomain):

            # Left boundary is "target domain" G
            def inside(self, x, on_boundary):
                par = MembraneSimulator.w / MembraneSimulator.length
                tol = 1E-4
                return on_boundary and x[1] >= - tol and x[1] <= tol

            # Map right boundary (H) to left boundary (G)
            def map(self, x, y):
                y[1] = x[1] + 2 * self.eta / MembraneSimulator.length
                y[0] = x[0]

        mesh_size = 50
        # Domain

        if method == 'circle':
            self.obstacle_radius = self.eta / MembraneSimulator.length / 2
            box_begin_point = fe.Point(-self.w / MembraneSimulator.length, 0)
            box_end_point = fe.Point(self.w / MembraneSimulator.length, 2 * self.eta / MembraneSimulator.length)
            box = mshr.Rectangle(box_begin_point, box_end_point)
            cell = box - mshr.Circle(fe.Point(0, self.eta / MembraneSimulator.length), self.obstacle_radius, mesh_size)
            self.cell_mesh = mshr.generate_mesh(cell, mesh_size)
            diff_coef = fe.Constant(((0, 0), (0, self.D[1, 1])))  # limit for regularisation below.
        # elif method == 'diff_coef':
        #     print("Haha this is going to crash. Also it is horrible in accuracy")
        #     self.cell_mesh = fe.RectangleMesh(fe.Point(-self.w / MembraneSimulator.length, 0),
        #                                       fe.Point(self.w / MembraneSimulator.length,
        #                                                2 * self.eta / MembraneSimulator.length), mesh_size, mesh_size)
        #     diff_coef = fe.Expression(
        #         (('0', '0'), ('0', 'val * ((x[0] - c_x)*(x[0] - c_x) + (x[1] - c_y)*(x[1] - c_y) > r*r)')),
        #         val=(4 * self.ref_T * MembraneSimulator.d2 / MembraneSimulator.length ** 2), c_x=0,
        #         c_y=(self.eta / MembraneSimulator.length),
        #         r=self.obstacle_radius, degree=2, domain=self.cell_mesh)
        elif method == 'regularized':
            box_begin_point = fe.Point(-self.w / MembraneSimulator.length, 0)
            box_end_point = fe.Point(self.w / MembraneSimulator.length, 2 * self.eta / MembraneSimulator.length)
            box = mshr.Rectangle(box_begin_point, box_end_point)
            obstacle_begin_point = fe.Point(-self.w / MembraneSimulator.length * self.obs_length,
                                            self.eta / MembraneSimulator.length * (1 - self.obs_height))
            obstacle_end_point = fe.Point(self.w / MembraneSimulator.length * self.obs_length,
                                          self.eta / MembraneSimulator.length * (1 + self.obs_height))
            obstacle = mshr.Rectangle(obstacle_begin_point, obstacle_end_point)
            cell = box - obstacle
            self.cell_mesh = mshr.generate_mesh(cell, mesh_size)
            diff_coef = fe.Constant(
                ((self.obs_ratio * self.D[0, 0], 0), (0, self.D[1, 1])))  # defect matrix.
        else:
            raise ValueError("%s not a valid method to solve cell problem" % method)
        self.cell_fs = fe.FunctionSpace(self.cell_mesh, 'P', 2)
        self.cell_solutions = [fe.Function(self.cell_fs), fe.Function(self.cell_fs)]
        w = fe.TrialFunction(self.cell_fs)
        phi = fe.TestFunction(self.cell_fs)
        scaled_unit_vectors = [fe.Constant((1. / np.sqrt(self.obs_ratio), 0.)), fe.Constant((0., 1.))]
        for i in range(2):
            weak_form = fe.dot(diff_coef * (fe.grad(w) + scaled_unit_vectors[i]), fe.grad(phi)) * fe.dx
            print("Solving cell problem")
            bc = fe.DirichletBC(self.cell_fs, fe.Constant(0), MembraneSimulator.cell_boundary)
            if i == 0:
                # Periodicity is applied automatically
                bc = None
            fe.solve(fe.lhs(weak_form) == fe.rhs(weak_form), self.cell_solutions[i], bc)
            if plot:
                plt.rc('text', usetex=True)
                f = fe.plot(self.cell_solutions[i])
                plt.colorbar(f)
                plt.title(r'Solution to cell problem $w_%d$' % (i + 1))
                plt.xlabel(r'$Y_1$')
                plt.ylabel(r'$Y_2$')
                print("Cell solution")
                print(np.min(self.cell_solutions[i].vector().get_local()),
                      np.max(self.cell_solutions[i].vector().get_local()))
                plt.savefig('anothercell.pdf')
                plt.show()

    def compute_effective_diffusion(self):
        """
        Computes the effective diffusion coefficient according to manuscript
        Shape of this coefficient:
        1/|Z| \int (D*(I + [[0,0],[\partial_{Y_2} w_1,\partial_{Y_2} w_2]]
        :return:
        """
        cell_size = 2 * self.eta / MembraneSimulator.length * 2 * self.w / MembraneSimulator.length
        print("Cell size is ", cell_size)
        self.eff_diff = np.zeros([2, 2])
        for i in range(2):
            grad = fe.grad(self.cell_solutions[i])
            part_div = fe.dot(self.unit_vectors[1], grad)
            integrand = fe.project(part_div, self.cell_fs)
            # integrand = fe.project(fe.dot(self.unit_vectors[1], fe.grad(self.cell_solutions[i])), self.cell_fs)
            self.eff_diff[1, i] += fe.assemble(integrand * fe.dx)
            print("integrand for cell problem", i, self.eff_diff[1, i] / cell_size)
        self.eff_diff = self.D + np.dot(self.D, self.eff_diff) / cell_size
        d1 = self.D[0, 0]
        d2 = self.D[1, 1]
        hom_d1 = self.eff_diff[0, 0]
        hom_d2 = self.eff_diff[1, 1]
        ratio = self.obs_height * self.obs_length
        print("Obstacle has a filling ratio of %.2f" % ratio)
        print("Homogenization: diff outside is %.4f, %.4f, diff inside is %.4f, %.4f" % (d1, d2, hom_d1, hom_d2))

        rad = MembraneSimulator.w / MembraneSimulator.length
        self.composed_diff_coef = fe.Expression(
            (('d1*(x[0]*x[0]>=rad*rad) + dd1*(x[0]*x[0]<rad*rad)', '0'),
             ('0', 'd2*(x[0]*x[0]>=rad*rad) + dd2*(x[0]*x[0]<rad*rad)')),
            d1=d1, d2=d2, rad=rad, dd1=hom_d1, dd2=hom_d2, degree=1)

    def plot_diffusion_coef(self, entries='01'):
        indices = [int(entry) for entry in entries]
        vector = np.zeros(2)
        vector[indices] = 1
        print(vector)
        fevec = fe.Constant(vector)
        val = fe.dot(self.composed_diff_coef, fevec)
        val = fe.dot(fevec, val)
        fe.plot(val, mesh=self.mesh)
        plt.show()

    def solve_pde(self,name='pde'):
        u = fe.Function(self.function_space)
        v = fe.TestFunction(self.function_space)
        u_old = fe.project(self.u_0, self.function_space)
        # Todo: Average right hand side if we choose it non-constant
        flux = self.dt * fe.dot(self.composed_diff_coef * (fe.grad(u) + self.drift_function(u)), fe.grad(v)) * fe.dx
        bilin_part = u * v * fe.dx + flux
        funtional_part = self.rhs * v * self.dt * fe.dx + u_old * v * fe.dx
        full_form = bilin_part - funtional_part
        num_steps = int(self.T / self.dt) + 1
        bc = fe.DirichletBC(self.function_space, self.u_boundary, MembraneSimulator.full_boundary)
        for n in range(num_steps):
            print("Step %d" % n)
            self.time += self.dt
            fe.solve(full_form == 0, u, bc)
            fe.plot(u)
            plt.savefig('images/plt%d.pdf'%n)
            plt.show()
            # print(fe.errornorm(u_old, u))
            u_old.assign(u)
            self.file << (u, self.time)
        plt.figure()
        # u=u+1E-9
        f = fe.plot(u)
        # f = fe.plot(u,norm=colors.LogNorm(vmin=1E-9,vmax=2E-4))
        plt.rc('text', usetex=True)
        plt.colorbar(f,format='%.0e')
        plt.title(r'Macroscopic density $u(x,t=1)$ for $\alpha=%.1f$'%MembraneSimulator.alpha)
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.savefig('%s.pdf'%name)
        plt.show()

    def solve_homogenized(self):
        self.solve_cell_problems(plot=False)
        self.compute_effective_diffusion()
        self.solve_pde()

    def plot_ratios(self):
        # ratios = 1 / 2 ** np.arange(16)
        ratios_0 = 1 / 4 * 1 / 2 ** np.arange(1,8+1)[::-1]
        ratios_1 = np.linspace(1 / 4, 1, 8)
        ratios = np.hstack((ratios_0, ratios_1))
        diff_coef = np.zeros_like(ratios)
        self.obs_length = 0.7
        for i, ratio in enumerate(ratios):
            print("Computing homogenised coefficient for ratio %.3e" % ratio)
            # self.obs_length = self.obs_height / ratio
            self.obs_height = self.obs_length * ratio
            self.solve_cell_problems()
            self.compute_effective_diffusion()
            diff_coef[i] = self.eff_diff[1, 1]
        print(diff_coef)
        plt.plot(ratios, diff_coef, '.-')
        plt.rc('text', usetex=True)
        plt.title(r'Value of diffusion coefficient for decreasing $\delta$')
        plt.xlabel('$\delta$')
        plt.ylabel(r'$D^*_{22}$')
        plt.savefig('ratios.eps')
        plt.show()

    def plot_cells(self):
        ratios = np.linspace(0.02, 1, 16)
        diff_coef = np.zeros_like(ratios)
        self.obs_length = 0.7
        self.obs_height = 0.1
        orig_eta = 0.2
        for i, ratio in enumerate(ratios):
            print("Computing homogenised coefficient for ratio %.3e" % ratio)
            MembraneSimulator.eta = orig_eta * ratio
            self.solve_cell_problems()
            self.compute_effective_diffusion()
            diff_coef[i] = self.eff_diff[1, 1]
        print(diff_coef)
        plt.rc('text', usetex=True)
        plt.plot(ratios, diff_coef, '.-')
        plt.title('Effective vertical diffusion for varying $\eta$')
        plt.xlabel('$\eta$')
        plt.ylabel(r'$D^*_{22}$')
        plt.savefig('cells.eps')
        plt.show()

    def test_drift(self):
        alphas = np.array([0.1,0.3,1,1.5,5,10,15])
        self.obs_length = 0.7
        for i, alpha in enumerate(alphas):
            MembraneSimulator.alpha = alpha
            self.solve_cell_problems()
            self.compute_effective_diffusion()
            self.solve_pde('alphaplot_%d'%i)

    @staticmethod
    def full_boundary(x, on_boundary):
        return on_boundary and (fe.near(x[0], -1) or fe.near(x[0], 1))

    @staticmethod
    def cell_boundary(x, on_boundary):
        par = 2 * MembraneSimulator.eta / MembraneSimulator.length
        return on_boundary and (fe.near(x[1], 0) or fe.near(x[1], par))

if __name__ == "__main__":
    solver = MembraneSimulator()
    solver.test_drift()
