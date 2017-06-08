import javabridge
import bioformats as bf
import numpy as np
from hspiral import hspiral
javabridge.start_vm(class_path=bf.JARS)

def getImageShape(path):
    xmlimage=bf.get_omexml_metadata(path=path)
    metadata= bf.OMEXML(xmlimage)

    NX= metadata.image().Pixels.SizeX
    NY= metadata.image().Pixels.SizeY
    NZ= metadata.image().Pixels.SizeZ
    NC= metadata.image().Pixels.SizeC  
    NT= metadata.image().Pixels.SizeT
    return (NX,NY,NZ,NC,NT)


path='/home/rod/tfDeconv-data/MyrSCARlifeRRokG_8.czi'

psfpaths=['/home/rod/tfDeconv-data/myrscarlifeRRokG_8-psfc0.ome.tif','/home/rod/tfDeconv-data/myrscarlifeRRokG_8-psfc1.ome.tif']

imageshape=getImageShape(path)

psfshapes=[getImageShape(p) for p in psfpaths]

psfs = [ np.zeros(shape,np.float32) for shape in psfshapes] 

for c in range(len(psfs)):
    with bf.ImageReader(path=psfpaths[c]) as psfreader:
        for z in range(psfshapes[c][4]):
            psfs[c][:,:,0,0,z]=psfreader.read(c=0,t=z,z=0,rescale=False)
psfs = [psf.reshape((psf.shape[0],psf.shape[1],psf.shape[4])) for psf in psfs]
        


with bf.ImageReader(path=path) as reader:
    for c in range(imageshape[3]):
        for t in range(imageshape[4]):
            image=np.zeros((imageshape[0],imageshape[1],imageshape[2]))
            for z in range(imageshape[2]):
                image[:,:,z]=reader.read(c=c,t=t,z=z,rescale=False)
                
            #perform deconvolution
            result = hspiral(image,psfs[c])
            
            result = image*2
            for z in range(imageshape[2]):
                bf.write_image(pathname='result.ome.tif',pixels=result[:,:,z],pixel_type="float",z=z,c=c,t=t,size_c=imageshape[3],size_z=imageshape[2],size_t=imageshape[4])                
            
            
        