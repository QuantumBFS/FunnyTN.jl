using FunnyTN.TensorNetworks: MPS, MPO 

def contract(L, K, Dcut):
    '''
    contract open square lattice Ising partition function using bounday MPS (poorman's MPS lib though)
    '''

    #construct the tensors
    c = torch.sqrt(torch.cosh(K))
    s = torch.sqrt(torch.sinh(K))
    M = torch.stack([torch.cat([c, s]), torch.cat([c, -s])])
    E = torch.tensor([[(s/c)**2, 0.], [0., (c/s)**2]], device=K.device) # observable tensor for energy
    
    T2 = torch.einsum('ai,aj->ij', (M, M))
    T3 = torch.einsum('ai,aj,ak->ijk', (M, M, M))
    T4 = torch.einsum('ai,aj,ak,al->ijkl', (M, M, M, M))

    mps = MPS(L,D=2) 
    for i in range(L):
        if (i==0 or i==L-1):
            mps[i] = T2.view(mps.bdim[i-1], 2, mps.bdim[i])
        else:
            mps[i] = T3 

    mpo = MPO(L,D=2) 
    for i in range(L):
        if (i==0 or i==L-1):
            mpo[i] = T3.view(mpo.bdim[i-1], 2, 2, mpo.bdim[i])
        else:
            mpo[i] = T4 
    
    res = 0.0
    for i in range(L//2-1):
        multiply(mpo, mps)
        res = res + compress(mps, Dcut)
    ovlp = overlap(mps, mps)
    lnZ = torch.log(ovlp) + 2*res 
    En = overlap(mps, mps, L//2, E)/ovlp  # energy measured at a bond in the system center

    return lnZ, En

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-float32", action='store_true', help="use float32")
    parser.add_argument("-cuda", type=int, default=-1, help="use GPU")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cuda<0 else "cuda:"+str(args.cuda))
    dtype = torch.float32 if args.float32 else torch.float64

    import numpy as np
    Dcut = 10
    L = 28

    for beta in np.linspace(0, 2.0, 21):
        K = torch.tensor([beta], dtype=dtype, device=device).requires_grad_()
        lnZ, En = contract(L, K, Dcut)
        lnZ.backward()
        print ('{:.1f} {:.8f} {:.8f} {:.8f}'.format(beta, lnZ/L**2, En, K.grad.item()/L**2))

        #(beta, free energy per site, energy computed at a bond via measurement, energy per site computed via BP)

