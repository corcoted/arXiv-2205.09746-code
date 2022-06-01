# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python [conda env:cf] *
#     language: python
#     name: conda-env-cf-py
# ---

# %% [markdown]
"""
# Example code
for "The usefulness of homogeneous coordinates in paraxial geometric optics"

[arXiv:2205.09746](http://arxiv.org/abs/2205.09746)

Theodore Corcovilos, (c) 2022 
"""

# %% [markdown]
# This file is in jupytext "percent" script for archival purposes.
# You may find it useful to convert this to a jupyter notebook file for running and editing.
#
# `
# jupytext --to ipynb worked-examples.py
# `
# %% [markdown]
# _Required packages: sympy_

# %%
from sympy import *
init_printing()
# %%
A, B, C, D, F, G = symbols("A, B, C, D, F, G", real=True)
h, m, x, y = symbols("h, m, x, y", real=True)
d, f, u, v = symbols("d, f, u, v",real=True)
n1, n2, R = symbols("n1, n2, R", real=True)

# %% [markdown]
# ## Definitions of rays and points

# %%
# left-to-right ray
ray = lambda h, m: Matrix([[-h],[-m],[1]])
ray(h,m)

# %%
# point
point = lambda x, y: Matrix([[1],[x],[y]])
point(x,y)


# %%
# normalize point
def normalize(p):
    if p[0,0] == 0:
        return p
    else:
        return p/p[0,0]


# %%
normalize(Matrix([[x],[1],[0]]))

# %% [markdown]
# ## Definitions of ray transfer matrices

# %%
# Standard ABCD matrix
M = Matrix([[A, B, 0],[C, D, 0],[0,0,1]])
M

# %%
# thin lens
Mlens = lambda f: Matrix([[1, 0, 0],[-1/f,1,0],[0,0,1]])
Mlens(f)

# %%
# propagation
Mprop = lambda d: Matrix([[1,d,0],[0,1,0],[0,0,1]])
Mprop(d)

# %%
# plane refraction
Msnell = lambda n1, n2: Matrix([[1,0,0],[0,n1/n2,0],[0,0,1]])
Msnell(n1,n2)

# %%
# spherical refraction (R>0 for convex)
Msphr = lambda n1, n2, R: Matrix([[1,0,0],[(n1-n2)/R/n2, n1/n2, 0],[0,0,1]])
Msphr(n1,n2,R)

# %%
# plane mirror
Mmirror = Matrix([[-1,0,0],[0,1,0],[0,0,-1]])
Mmirror

# %%
# spherical mirror (R>0 is convex)
Msphm = lambda R: Matrix([[-1, 0, 0],[2/R,1,0],[0,0,-1]])
Msphm(R)

# %% [markdown]
# ## Definition of matrix adjugate

# %%
# matrix adjugate (this will fail in the unphysical case det(x)=0 )
adj = lambda x: x.det()*(x.inv().T)

# %%
for thisM in [M,Msphm(R),Msnell(n1,n2),Msphr(n1,n2,R)]:
    display(adj(thisM))

# %% [markdown]
# ## Coordinate transformations

# %%
# Rotation
Rot = lambda u: Matrix([[1,0,0],[0,cos(u),-sin(u)],[0,sin(u),cos(u)]])
Rot(u)

# %%
# Translation
T = lambda u, v: Matrix([[1,-u,-v],[0,1,0],[0,0,1]])
T(u,v)

# %%
Rot(u)*ray(h,m)

# %%
T(u,v)*ray(h,m)

# %%
simplify(adj(Rot(u)))

# %%
simplify(adj(T(u,v)))

# %%
simplify(adj(Rot(u)))*point(x,y)

# %%
simplify(adj(T(u,v)))*point(x,y)

# %%
#contrast with Gerrard
R_G = lambda u: Matrix([[1,0,0],[0,1,-u],[0,0,1]])
R_G(u)

# %%
R_G(u)*ray(h,m)

# %%
simplify(adj(R_G(u)))

# %%
simplify(adj(R_G(u)))*point(x,y)

# %% [markdown]
# ## Examples

# %% [markdown]
# ### Example 4.1
# Tilted window

# %%
M4p1 = simplify(Rot(u)*T(d,0)*Msnell(n1,1)*(T(d,0).inv())*Msnell(1,n1)*(Rot(u).inv()))
M4p1

# %%
# small u approximation
M4p1linear = M4p1.applyfunc(lambda a: series(a, x=u, x0=0, n=2).removeO())
M4p1linear

# %%
M4p1linear*ray(0,0)

# %%
up = symbols("up", real=True)

# %%
ytest = (d*sin(u-up)/cos(up)).subs(up, asin(sin(u/n1)))
ytest

# %%
# approximate arcsin with 3rd-order Taylor series, then get the series expansion for ytest
(d*sin(u-up)/cos(up)).subs(up, (sin(u/n1))+(sin(u/n1))**3/6).series(x=u,x0=0,n=4).removeO()

# %%
collect(expand(_),u)

# %%
# These agree to first order in u

# %% [markdown]
# ### Example 5.1
# Retroreflector

# %% [markdown]
# Two mirrors separated by an angle _u_, with intersection at the origin.  We'll consider an incoming ray parallel to the incoming optical axis and below it by a distance _h_.

# %%
Rot(pi/4)

# %%
M1 = simplify(Rot(u/2)*Mmirror*Rot(-u/2))
M1.subs(u,pi/2)

# %%
M2 = simplify(M1.subs(u,-u))
M2.subs(u,pi/2)

# %%
simplify(M2*M1)

# %%
simplify(M2*M1)*ray(h,m)

# %%
_.subs(u,pi/2)

# %%
M1.subs(u,pi/2)*ray(h,m)

# %%
(M2*M1).subs(u,pi/2)*ray(h,m)

# %% [markdown]
# ### Example point transfer matrices

# %%
adj(Mlens(f))

# %%
# image point through thin lens
normalize(adj(Mlens(f))*point(x,y))

# %% [markdown]
# ### Example 7.2
# Imaging a distant object

# %%
adj(Mlens(50.))*Matrix([[0],[-1],[0.01]])

# %%
normalize(_)

# %% [markdown]
# ### Example 7.3
# Analysis of a compound lens

# %%
# system RTM
M7p3 = Matrix([[0.867,1.338,0],[-0.198,0.848,0],[0,0,1]])
M7p3

# %%
# Object point
O7p3 = point(-20,0.1)
O7p3

# %%
# system PTM
adj(M7p3)

# %%
# image point
adj(M7p3)*O7p3

# %%
normalize(_)

# %%
# BFP
adj(M7p3)*Matrix([[0],[-1],[0]])

# %%
normalize(_)

# %% [markdown]
# ### Example 7.4
# Misaligned thin lens

# %%
# translated lens
M7p4a = T(0,d)*Mlens(f)*(T(0,d).inv())
M7p4a

# %%
adj(M7p4a)

# %%
adj(M7p4a)*Matrix([[0],[-1],[0]])

# %%
normalize(_)

# %%
# tilted lens
M7p4b = simplify(Rot(u)*Mlens(f)*(Rot(u).inv()))
M7p4b

# %%
adj(M7p4b)

# %%
adj(M7p4b)*Matrix([[0],[1],[0]])

# %% [markdown]
# ## Other examples
# Not in the paper

# %% [markdown]
# ### Folding mirrors
# A common setup in a laser lab is a pair of plane mirrors arranged to route the beam in a Z shape.  This provides the necessary degrees of freedom to adjust both the pointing and lateral position of the beam.
#
# We'll demonstrate this with a pair of mirrors tilted at 30 degrees relative to the axis.
#
# * Mirror 1: located at (a,0), tilted by 30 degrees
# * Mirror 2: located at (0,a*sqrt(4/3)), tilted by 210 degrees
#
# The incoming beam will be along the optical axis

# %%
a = symbols('a', real=True)

# %%
# Mirror 1
FM1 = simplify(T(a,0)*Rot(pi/3)*Mmirror*Rot(-pi/3)*T(-a,0))
FM1

# %%
# Mirror2
FM2 = simplify(T(0,a*2/sqrt(3))*Rot(4*pi/3)*Mmirror*Rot(-4*pi/3)*T(0,a*-2/sqrt(3)))
FM2

# %%
# The combination is just a translation of the beam up by $\sqrt{3}/2$
FM2*FM1

# %%
# acting on the ray
FM2*FM1*ray(0,0)

# %% [markdown]
# The end result wasn't surprising: we translate the beam up by $\sqrt{3}/2$.  Let's add in some small error in the tilt angles: $\delta_1$ and $\delta_2$

# %%
delta1, delta2 = symbols("delta1 delta2", real=True)

# %%
# Mirror 1
FM1d = simplify(T(a,0)*Rot(pi/3+delta1)*Mmirror*Rot(-pi/3-delta1)*T(-a,0))
FM1d

# %%
FM2d = simplify(T(0,a*2/sqrt(3))*Rot(4*pi/3+delta2)*Mmirror*Rot(-4*pi/3-delta2)*T(0,a*-2/sqrt(3)))
FM2d

# %%
simplify(FM2d*FM1d)

# %%
(FM2d.applyfunc(lambda x: series(x,delta2,0,2).removeO()))*(FM1d.applyfunc(lambda x: series(x,delta1,0,2).removeO()))

# %%
simplify(_)

# %%
_*ray(0,0)

# %% [markdown]
# So, in summary to first order in the &delta;'s, the new ray height depends only on $\delta_1$, but we need $\delta_2=\delta_1$ to keep the ray parallel to the axis.  This agrees with our geometric intuition.  The first mirror sets the height by picking the location on the second mirror.  The second mirror needs to be parallel to the first mirror to make the outgoing ray parallel to the axis.

# %% [markdown]
# ### Example: thick lens
# Let's try a plano-convex lens, with the curved side first, vertex at the origin, and reseting the coordinate axes at the end.
#
# * Radius of curvature: _R_ = 50 mm
# * Thickness: _d_ = 5 mm
# * Index of refraction: _n_ = 3/2

# %%
R = 50
d = 5
n = Rational(3,2)

# %%
Mpx=T(d,0)*Msnell(n,1)*T(-d,0)*Msphr(1,n,R)
Mpx

# %% [markdown]
# The effective focal length is -1/B, just like the thin lens. (Think of this as the conversion factor between the height of an incoming horizontal ray and the slope of the outgoing ray.)

# %%
EFL = -1/Mpx[1,0]
EFL

# %% [markdown]
# We can find the back focal point by imaging an ideal point in the -x direction. (Note that this will be measured from the origin, i.e. the front of the lens).

# %%
BFP=N(normalize(adj(Mpx)*Matrix([[0],[-1],[0]])))[1]
BFP

# %% [markdown]
# We can find the front focal point by reversing the system and repeating the above.  The vertex of the convex side will still be at the origin after the inversion.
#
# To reverse the system, we need to invert the system matrix and bring in the ray from the left side.

# %%
FFP=N(normalize(adj(Mpx).inv()*Matrix([[0],[-1],[0]])))[1]
FFP

# %% [markdown]
# The principal planes are a distance EFL from the focal points.

# %%
FPP = FFP+EFL
FPP

# %%
BPP = BFP-EFL
BPP

# %% [markdown]
# These results agree with the known properties of a glass plano-convex lens.  The principal planes are at the vertex of the convex surface and about 1/3 of the thickness of the lens from the vertex.
