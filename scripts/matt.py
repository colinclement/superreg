import numpy as np

def intshifts(data, shifts, size=64):
    C = data.shape[-1]//2 - size//2
    #C = shifts.mean(0)

    s = shifts.astype('int')
    out = [data[0][C:C+size, C:C+size]]

    for i in range(len(shifts)):
        d = data[i+1]
        y, x = s[i]
        out.append(d[C-y:C-y+size, C-x:C-x+size])
    return np.array(out)
