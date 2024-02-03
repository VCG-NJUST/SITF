import numpy as np
import os
def addgaussiannoise():
    means=[0.0025,0.005,0.01,0.015,0.025]
    with open(os.path.join('D:/jk/PCN/train', 'gs_noise_1.txt'), 'r') as f:
        shape_names = f.readlines()
    shape_names = [x.strip() for x in shape_names]
    shape_names = list(filter(None, shape_names))
    for shape_id,shape_name in enumerate(shape_names):
        save_path=os.path.join('D:/jk/PCN/train',shape_name+'.npy')
        pts=np.load(save_path)
        bbdiag = float(np.linalg.norm(pts.max(0) - pts.min(0), 2))
        #print(pts.dtype)
        for mean in means:
            noise=np.random.randn(pts.shape[0],pts.shape[1])*bbdiag*mean
            out=pts+noise
            #print(out.dtype)
            np.save(os.path.join('D:/jk/PCN/train',shape_name+'_'+str(mean)+'.npy'),out.astype(np.float32))
            #np.save(os.path.join('D:/jk/PCN/origin', shape_name + '_' + str(mean) + '.npy'),out.astype(np.float32))

def addgaussiannoise_1(mean, noise_sub):
    bbdiag = float(np.linalg.norm(noise_sub.max(0) - noise_sub.min(0), 2))
    # print(pts.dtype)
    noise = np.random.randn(noise_sub.shape[0], noise_sub.shape[1]) * bbdiag * mean
    out = noise_sub + noise
    return out

def addpulsenoise():
    noise_levels=[0.01]
    with open(os.path.join('E:/deep learning/pf/Dataset/try', 'test.txt'), 'r') as f:
        shape_names = f.readlines()
    shape_names = [x.strip() for x in shape_names]
    shape_names = list(filter(None, shape_names))
    for shape_id,shape_name in enumerate(shape_names):
        pts=np.load(os.path.join('E:/deep learning/pf/Dataset/try',shape_name+'.npy'))
        for noise_level in noise_levels:
            pulse_intensity=noise_level*3
            p=0.2
            x=np.random.rand(pts.shape[0],pts.shape[1])
            f=np.zeros(pts.shape[0],pts.shape[1])
            f[x<p/2]=-pulse_intensity
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    if x[i, j] > p/2 and x[i, j] < p and True:
                        f[i, j] = pulse_intensity
            out=pts+f
            np.save(os.path.join('D:/jk/self-supervision/train', shape_name + '_pulse' + str(noise_level) + '.npy'), out.astype(np.float32))

def addmixnoise():
    means = [0.0025, 0.005, 0.01, 0.015, 0.025]
    noise_levels = [0.01]
    with open(os.path.join('D:/jk/self-supervision/train', 'GT0.01.txt'), 'r') as f:
        shape_names = f.readlines()
    shape_names = [x.strip() for x in shape_names]
    shape_names = list(filter(None, shape_names))
    for shape_id, shape_name in enumerate(shape_names):
        save_path = os.path.join('D:/jk/self-supervision/train', shape_name + '.npy')
        pts = np.load(save_path)
        bbdiag = float(np.linalg.norm(pts.max(0) - pts.min(0), 2))
        # print(pts.dtype)
        for mean in means:
            noise = np.random.randn(pts.shape[0], pts.shape[1]) * bbdiag * mean
            ptr = pts + noise
            for noise_level in noise_levels:
                pulse_intensity = bbdiag*noise_level * 3
                p = 0.2
                x = np.random.rand(ptr.shape[0], ptr.shape[1])
                f = np.zeros(shape=ptr.shape)
                f[x < p / 2] = -pulse_intensity
                for i in range(x.shape[0]):
                    for j in range(x.shape[1]):
                        if x[i, j] > p / 2 and x[i, j] < p and True:
                            f[i, j] = pulse_intensity
                out = ptr + f
                np.save(os.path.join('D:/jk/self-supervision/train', shape_name + '_mixgassion'+str(mean)+'pulse' + str(noise_level) + '.npy'),out.astype(np.float32))

if __name__ == '__main__':
    addgaussiannoise()