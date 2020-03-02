# analytic_solution.py
# Python version of the analytic solution described in Appendix A of
# S.P. Veluri, C.J. Roy and E.A. Luke
# Comprehensive code verification techniques for finite volume CFD codes
# Computers & Fluids, 2012;
#
# KD, 2018
# It is essentially Rowan's original code formulated for three-dimensional solutio

from __future__ import print_function
from sympy import *
exec(open('./constants.txt').read())
# Read case no.
fp = open('case.txt', 'r');
scale_str = fp.readline().strip()
scale = int(scale_str)
fp.close()
# constants
L = 1.0
# Subsonic flow
rho0=1.0; rhox=0.15; rhoy=-0.1; rhoz=0.1; rhoxy=0.08; rhoxz=0.05; rhoyz=0.12; rhoxyz=0.1;
arhox=0.75; arhoy=0.45; arhoz=0.8; arhoxy=0.65; arhoxz=0.75; arhoyz=0.5; arhoxyz=0.2
u0=70.0; ux=7.0; uy=-15.0; uz=-10.0; uxy=7.0; uxz=4.0; uyz=-4.0; uxyz=-2.0;
aux=0.5; auy=0.85; auz=0.4; auxy=0.6; auxz=0.8; auyz=0.9; auxyz=0.5;
v0=90.0; vx=-5.0; vy=10.0; vz=5.0; vxy=-11.0; vxz=-5.0; vyz=5.0; vxyz=10.0;
avx=0.8; avy=0.8; avz=0.5; avxy=0.9; avxz=0.4; avyz=0.6; avxyz=0.2;
w0=80.0; wx=-10.0; wy=10.0; wz=12.0; wxy=-12.0; wxz=11.0; wyz=5.0; wxyz=20.0;
awx=0.85; awy=0.9; awz=0.5; awxy=0.4; awxz=0.8; awyz=0.75; awxyz=0.3;
p0=1.0e5; px=0.2e5; py=0.5e5; pz=0.2e5; pxy=-0.25e5; pxz=-0.1e5; pyz=0.1e5; pxyz=0.5e5;
apx=0.4; apy=0.45; apz=0.85; apxy=0.75; apxz=0.7; apyz=0.8; apxyz=0.3;

x, y, z, rho, u, v, w, p, S = symbols('x y z rho u v w p S')

if scale == 0:
    S = 1.0
else:
    S = exp(-16.0*((x-L/2)*(x-L/2) + (y-L/2)*(y-L/2) + (z-L/2)*(z-L/2))/(L*L))

rho = rho0 + S*rhox*sin(arhox*pi*x/L) + S*rhoy*cos(arhoy*pi*y/L) + \
   S*rhoz*cos(arhoz*pi*z/L) + S*rhoxy*cos(arhoxy*pi*x*y/(L*L)) + \
    S*rhoxz*sin(arhoxz*pi*x*z/(L*L)) + S*rhoyz*cos(arhoyz*pi*y*z/(L*L)) + \
    S*rhoxyz*sin(arhoxyz*pi*x*y*z/(L*L*L));
u =  u0 + S*ux*sin(aux*pi*x/L) + S*uy*cos(auy*pi*y/L) + S*uz*cos(auz*pi*z/L) + \
     S*uxy*cos(auxy*pi*x*y/(L*L)) + S*uxz*sin(auxz*pi*x*z/(L*L)) + \
     S*uyz*cos(auyz*pi*y*z/(L*L)) + S*uxyz*sin(auxyz*pi*x*y*z/(L*L*L));
v =  v0 + S*vx*sin(avx*pi*x/L) + S*vy*cos(avy*pi*y/L) + S*vz*cos(avz*pi*z/L) + \
      S*vxy*cos(avxy*pi*x*y/(L*L)) + S*vxz*sin(avxz*pi*x*z/(L*L)) + \
     S*vyz*sin(avyz*pi*y*z/(L*L)) + S*vxyz*cos(avxyz*pi*x*y*z/(L*L*L));
w =  w0 + S*wx*sin(awx*pi*x/L) + S*wy*cos(awy*pi*y/L) + S*wz*cos(awz*pi*z/L) + \
      S*wxy*cos(awxy*pi*x*y/(L*L)) + S*wxz*cos(awxz*pi*x*z/(L*L)) + \
     S*wyz*sin(awyz*pi*y*z/(L*L)) + S*wxyz*sin(awxyz*pi*x*y*z/(L*L*L));
p =  p0 + S*px*cos(apx*pi*x/L) + S*py*sin(apy*pi*y/L) +  S*pz*sin(apz*pi*z/L) + \
      S*pxy*sin(apxy*pi*x*y/(L*L)) + S*pxz*sin(apxz*pi*x*z/(L*L)) + \
     S*pyz*cos(apyz*pi*y*z/(L*L)) + S*pxyz*cos(apxyz*pi*x*y*z/(L*L*L));

def ref_function(x1, y1, z1, t):
    inp = {x:x1, y:y1, z:z1}
    rho1 = rho.subs(inp).evalf()
    p1 = p.subs(inp).evalf()
    T1 = p1/(rho1*R_air)
    u1 = u.subs(inp).evalf()
    v1 = v.subs(inp).evalf()
    w1 = w.subs(inp).evalf()
    return {"rho":rho1, "p":p1, "T":T1, "vel.x":u1, "vel.y":v1, "vel.z":w1}
        
if __name__ == "__main__":
    pt = {x:0.5, y:0.5, z:0.5}
    print('rho=', rho.subs(pt).evalf(), \
        'u=', u.subs(pt).evalf(), \
        'v=', v.subs(pt).evalf(), \
        'w=', w.subs(pt).evalf(), \
        'p=', p.subs(pt).evalf())
        
