import numpy as np


def Eulerstep(f,u,dt): 
    return u+dt*f(u)

def RK4step(f,u,dt):
    k1 = dt*f(u)
    k2 = dt*f(u+0.5*k1)
    k3 = dt*f(u+0.5*k2)
    k4 = dt*f(u+k3)
    return u + (k1+2*k2+2*k3+k4)/6

integrator_dict = {'Euler': Eulerstep,
                   'RK4': RK4step}

class Orbit(object):
    """
    Simulate a dimensionless orbit of a massless particle arounf a central body with varying eccentricity and demonstrate
    the performance of different integration schemes with and without regularization (coordinate transformations)

    Args:
        e (float): Eccentricity of an elliptical orbit. Must be between 0 and 1
        Integrator (str): Name of integrator to use. Options include 'RK4', 'Euler' (and more to come)
        regularized (bool): If True, uses Kustaanheimo-Steifel (KS) regularization. Otherwise, Cartesian coordinates are used throughout
        dt (float): Timestep to use for integration. When KS regularization is performed, this corresponds to the fictitious time s instead of the physical time t.
        adaptive (bool): If true, uses adaptive timestepping
        tol (float): global tolerance value to use for adaptive timestepping
    """

    def __init__(self, e=0.1, integrator='Euler', regularized=False, dt=0.1, adaptive=False, tol=1e-5):
        # nondimensionalize the problem by setting all relevant quantities to 1. Can rescale after
        # all equations implicitly assume G=M=1
        self.a = 1

        self.t = 0
        self.nstep = 0
        self.dt = dt
        self.regularized = regularized
        self.adaptive = adaptive
        self.tol = tol
        if e>=1 or e<0:
            raise ValueError("Eccentricity must be greater than or equal to 0 and less than 1")
        elif e==0:
            print("Sorry, a couple of the methods here don't play well for circular orbits, so I'm setting e=1e-8. If you wish, you can choose a smaller, nonzero value.")
            self.e = 1e-8
        else:
            self.e = e
        if integrator in integrator_dict.keys():
            self.integrator = integrator
        else:
            raise Exception("Please provide valid integrator")

        # Initialize orbit at apocenter
        self.f = np.pi

        self.threepos = None
        self.threevel = None

        self.fourpos = None
        self.fourvel = None

        # convert from a,e,f to x,v 3-vectors
        self.convert_kepler2cartesian()
        self.convert_cartesian2regular()
        print("Finished initializing orbit")
    
    # See Murray and Dermott (2000) Ch.2 for these formulae
    def convert_kepler2cartesian(self):
        n = np.power(self.a,-1.5)
        self.threepos = np.array([np.cos(self.f), np.sin(self.f), 0])*self.a*(1-self.e**2) / (1+self.e*np.cos(self.f))
        self.threevel = np.array([np.sin(self.f), self.e+np.cos(self.f), 0])*n*self.a/np.sqrt(1-self.e**2)
    
    def convert_cartesian2kepler(self):
        r = np.linalg.norm(self.threepos)
        v = np.linalg.norm(self.threevel)
        h_vec = np.cross(self.threepos,self.threevel)
        e_vec = np.cross(self.threevel,h_vec) - self.threepos/r
        e = np.linalg.norm(e_vec)
        self.a = 1/(2/r - v**2)
        self.e = e
        f = np.arccos(np.dot(e_vec,self.threepos)/(e*r))
        if np.dot(self.threepos,self.threevel) < 0:
            self.f = 2*np.pi - f
        else:
            self.f = f
    
    def convert_cartesian2regular(self):
        # the transform is non-unique, so I used the version from
        # Appendix A of https://iopscience.iop.org/article/10.1086/429546/pdf
        x,y,z = self.threepos
        r = np.linalg.norm(self.threepos)
        vx,vy,vz = self.threevel

        if x<0:
            u2 = np.sqrt(0.5*(r-x))
            u3 = 0
            u1 = 0.5*y/u2
            u4 = 0.5*z/u2
        else:
            u1 = np.sqrt(0.5*(r+x))
            u4 = 0
            u2 = 0.5*y/u1
            u3 = 0.5*z/u1
        self.fourpos = np.array([u1,u2,u3,u4])

        uv1 = 0.5*(u1*vx + u2*vy + u3*vz)
        uv2 = 0.5*(-u2*vx + u1*vy + u4*vz)
        uv3 = 0.5*(-u3*vx - u4*vy +u1*vz)
        uv4 = 0.5*(u4*vx - u3*vy + u2*vz)
        self.fourvel = np.array([uv1,uv2,uv3,uv4])
    
    def convert_regular2cartesian(self):
        u1,u2,u3,u4 = self.fourpos
        uv1,uv2,uv3,uv4 = self.fourvel
        x = u1**2 - u2**2 - u3**2 + u4**2
        y = 2*(u1*u2 - u3*u4)
        z = 2*(u1*u3 + u2*u4)
        self.threepos = np.array([x,y,z])

        r = np.linalg.norm(self.threepos)
        vx = 2*(u1*uv1 - u2*uv2 - u3*uv3 + u4*uv4)/r
        vy = 2*(u2*uv1 + u1*uv2 - u4*uv3 + u3*uv4)/r
        vz = 2*(u3*uv1 + u4*uv2 + u1*uv3 + u2*uv4)/r

        self.threevel = np.array([vx,vy,vz])

    def get_pos(self):
        return self.threepos
    
    def get_vel(self):
        return self.threevel
    
    def get_time(self):
        return self.t
    
    def get_stepcount(self):
        return self.nstep
    
    def get_kepler(self):
        return self.a, self.e, self.f

    def get_energy(self):
        return 0.5*np.linalg.norm(self.threevel)**2 - 1/np.linalg.norm(self.threepos)

    def get_angmomentum(self):
        return np.linalg.norm(np.cross(self.threepos,self.threevel))

    def dy_normal(self,u):
        # theoretically could be a static method, but must not be if you add in nonunity G and M
        x = u[:3]
        v = u[3:]
        r = np.linalg.norm(x)
        f = -1/r**3
        return np.concatenate((v, f*x), axis=None)
    
    def dy_regular(self,u):
        x = u[:4]
        v = u[4:8]
        #t = u[8]
        h = -self.get_energy()
        r = np.linalg.norm(x)**2
        return np.concatenate((v,-x*h/2,r),axis=None)

    def advance_step(self):
        if self.regularized:
            self.advance_step_regular()
        else:
            self.advance_step_normal()
    
    # the adaptive scheme in the following sections assume a second order method
    # this is not true in general, specifically for RK4
    # taken from https://en.wikipedia.org/wiki/Adaptive_step_size
    def advance_step_normal(self):
        y0 = np.concatenate((self.threepos,self.threevel),axis=None)
        stepfunc = integrator_dict[self.integrator]
        if self.adaptive:
            yerr = np.ones(2)
            while (yerr>self.tol).any():
                y1 = stepfunc(self.dy_normal,y0,self.dt)
                y05 = stepfunc(self.dy_normal,y0,self.dt/2)
                y1p = stepfunc(self.dy_normal,y05,self.dt/2)
                yerr = np.abs(y1p-y1)
                self.dt = 0.9*self.dt*np.min((np.max((np.sqrt(0.5*self.tol/(1e-10+np.max(yerr))),0.3)),2))
        else:
            y1 = stepfunc(self.dy_normal,y0,self.dt)
        self.t += self.dt
        self.nstep += 1
        self.threepos = y1[:3]
        self.threevel = y1[3:]
        self.convert_cartesian2regular()
        self.convert_cartesian2kepler()
    
    def advance_step_regular(self):
        y0 = np.concatenate((self.fourpos,self.fourvel,self.t),axis=None)
        stepfunc = integrator_dict[self.integrator]
        if self.adaptive:
            yerr = np.ones(2)
            while (yerr[:-1]>self.tol).any():
                y1 = stepfunc(self.dy_regular,y0,self.dt)
                y05 = stepfunc(self.dy_regular,y0,self.dt/2)
                y1p = stepfunc(self.dy_regular,y05,self.dt/2)
                yerr = np.abs(y1p-y1)
                self.dt = 0.9*self.dt*np.min((np.max((np.sqrt(0.5*self.tol/(1e-10+np.max(yerr))),0.3)),2))
        else:
            y1 = stepfunc(self.dy_regular,y0,self.dt)
        self.t = y1[8]
        self.nstep += 1
        self.fourpos = y1[:4]
        self.fourvel = y1[4:8]
        self.convert_regular2cartesian()
        self.convert_cartesian2kepler()

