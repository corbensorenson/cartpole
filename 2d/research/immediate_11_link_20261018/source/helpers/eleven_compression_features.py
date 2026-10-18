"""Smooth full-physics features for native compression-aware path updates."""
import numpy as np
from spatialpole.world_coordinates import native_geometry,world_coordinate_jacobian
from eleven_load_support import Loads


def physical_features(geometry,state,reference,reference_rotation,reference_omega,jacobian=False):
    rotation,omega=native_geometry(geometry,state)
    angular=np.einsum('nji,nj->ni',reference_rotation,omega-reference_omega)
    angular_weight=np.sqrt([.05,.05,.005])
    residual=np.r_[2.*(state[:2]-reference[:2]),
        np.sqrt(40.)*(rotation[:,:,2]-reference_rotation[:,:,2]).ravel(),
        np.sqrt(.05)*(state[46:48]-reference[46:48]),(angular*angular_weight).ravel()]
    if not jacobian:return residual
    local=world_coordinate_jacobian(geometry,state);J=np.zeros((70,70));J[:2]=2.*local[:2]
    for i in range(11):
        polar=local[2+2*i:4+2*i]
        J[2+3*i:5+3*i]=np.sqrt(40.)*(rotation[i,:,0,None]*polar[1]-rotation[i,:,1,None]*polar[0])
        current=local[26+3*i:29+3*i]
        J[37+3*i:40+3*i]=angular_weight[:,None]*(reference_rotation[i].T@rotation[i]@current)
    J[35:37]=np.sqrt(.05)*local[24:26]
    return residual,J


def compression_features(probe,state,action,jacobian=False):
    normalized=probe.at(state,action)/probe.weight
    residual=np.maximum(0.,-1.1-normalized)
    if not jacobian:return residual,normalized
    J=np.zeros((11,70));D=np.zeros((11,2));active=normalized<-1.1
    if active.any():
        eps=1e-5
        for j in range(70):
            delta=np.zeros(70);delta[j]=eps
            plus=probe.at(probe.env.integrate(state,delta),action)
            minus=probe.at(probe.env.integrate(state,-delta),action)
            J[active,j]=-(plus[active]-minus[active])/(2*eps)/probe.weight[active]
        _,matrix=probe.affine(state);D[active]=-matrix[active]/probe.weight[active,None]
    return residual,J,D,normalized
