import javabridge
import bioformats as bf
import numpy as np
from hspiral import HSPIRAL
javabridge.start_vm(class_path=bf.JARS)
import tensorflow as tf

def getImageShape(path):
    xmlimage=bf.get_omexml_metadata(path=path)
    metadata= bf.OMEXML(xmlimage)

    NX= metadata.image().Pixels.SizeX
    NY= metadata.image().Pixels.SizeY
    NZ= metadata.image().Pixels.SizeZ
    NC= metadata.image().Pixels.SizeC  
    NT= metadata.image().Pixels.SizeT
    return (NX,NY,NZ,NC,NT)


path='/media/rcilla/CillaTufts/tfDeconv-data/MyrSCARlifeRRokG_8.czi'

psfpaths=['/media/rcilla/CillaTufts/tfDeconv-data/myrscarlifeRRokG_8-psfc0.ome.tif','/media/rcilla/CillaTufts/tfDeconv-data/myrscarlifeRRokG_8-psfc1.ome.tif']

imageshape=getImageShape(path)

psfshapes=[getImageShape(p) for p in psfpaths]

psfs = [ np.zeros(shape,np.float32) for shape in psfshapes] 

for c in range(len(psfs)):
    with bf.ImageReader(path=psfpaths[c]) as psfreader:
        for z in range(psfshapes[c][4]):
            psfs[c][:,:,0,0,z]=psfreader.read(c=0,t=z,z=0,rescale=False)
psfs = [psf.reshape((psf.shape[0],psf.shape[1],psf.shape[4])) for psf in psfs]
        
psfs = [psf[24:-24,24:-24,:] for psf in psfs]

deconvobj=None
with tf.Session() as session:
    with bf.ImageReader(path=path) as reader:
        for c in range(imageshape[3]):
            for t in range(imageshape[4]):
                image=np.zeros((imageshape[0],imageshape[1],imageshape[2]))
                for z in range(imageshape[2]):
                    image[:,:,z]=reader.read(c=c,t=t,z=z,rescale=False)
                
                
                if deconvobj==None:
                    deconvobj=HSPIRAL(image.shape,psfs[c].shape)
                result = deconvobj.run(image,psfs[c],session)
                
                bf.write_image(pathname='MyrSCARlifeRRokG_8-c%d-t%03d.ome.tif'%(c,t),pixels=result,pixel_type="float",c=0,t=0,size_c=1,size_z=result.shape[2],size_t=1)