import numpy as np

def show_map(gen_object, x,y, scale=1, offset=[0,0], sampling = 1, channel=-1, octaves=1,neg_octaves=0, fade=0.5,dim3=False):
    sampling = int(sampling) #Just a check to clean up this input
    #Sampling is used when taking multiple samples per pixel, important with finer noise
    
    
    ### Remap according to scale and offset.
    #Works, but is in need of a cleanup
    #Scale should result in distance per pixel, offset in the new centre
    #x,y are the final image size. Should these affect size of mapped area?
    offset[0] = offset[0]  -x*0.5* scale +0.5* scale
    
    offset[1] = offset[1] -y*0.5* scale +0.5* scale
    offset[0] = offset[0] 
    
    offset[1] = offset[1] 
    ###
    
    output = np.zeros(shape=(x,y))
    if dim3:
        output = np.zeros(shape=(x,y,3))
    
    if channel==-1 and octaves==1 and neg_octaves==0 and fade==0.5:
        extra_args = False
    else:
        extra_args = True
    
    for i in range(x):
        for j in range(y):
            out = 0
            if sampling > 1:
                for _ in range(sampling): #Multiple (jittered) checks per pixel, reduces accuracy at low values on smooth shapes
                    if extra_args:
                        out += gen_object.get_height(i*scale+offset[0]+ (random.random()-0.5)*scale, j*scale + offset[1] + (random.random()-0.5)*scale, channel=channel, octaves=octaves,neg_octaves=neg_octaves, fade=fade)/sampling
                    else:
                        out += gen_object.get_height(i*scale+offset[0]+ (random.random()-0.5)*scale, j*scale + offset[1] + (random.random()-0.5)*scale)/sampling
            else:
                if extra_args:
                    out += gen_object.get_height(i*scale+offset[0], j*scale + offset[1],channel=channel, octaves=octaves,neg_octaves=neg_octaves, fade=fade)
                else:
                    out += gen_object.get_height(i*scale+offset[0], j*scale + offset[1])
            if dim3:
                output[j][i][0] = i
                output[j][i][1] = j
                output[j][i][2] = out
            else:
                output[j][i] = out
    return(output)

def show_map_3d(gen_obj,img_x,img_y,sca,centre_offset=[0,0],num_samples = 1,**kwargs):
    #x = np.arange(int(centre_offset[0]-0.5*img_x*sca), int(centre_offset[0]-0.5*img_x*sca)+img_x*sca, sca)
    #y = np.arange(int(centre_offset[1]-0.5*img_y*sca), int(centre_offset[1]-0.5*img_y*sca)+img_y*sca, sca)
    x = np.linspace(int(centre_offset[0]-0.5*img_x*sca), int(centre_offset[0]+0.5*img_x*sca), img_x)
    y = np.linspace(int(centre_offset[1]-0.5*img_y*sca), int(centre_offset[1]+0.5*img_y*sca), img_y)
    X, Y = np.meshgrid(x, y)
    if num_samples == 1:
        Z = np.asarray([[gen_obj.get_height(a,b,**kwargs) for a in x] for b in y])
    else:
        Z = np.zeros((img_x,img_y))
        for _ in range(num_samples):
            offset_x = np.random.rand(img_x,img_y)-0.5
            offset_x = offset_x*sca
            offset_x += X
            offset_y = np.random.rand(img_x,img_y)-0.5 
            offset_y = offset_y*sca
            offset_y += Y
            Z += np.asarray([[gen_obj.get_height(a,b,**kwargs) for a in x] for b in y])/num_samples
            
    return (X, Y, Z)