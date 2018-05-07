import javabridge
import bioformats as bf
import numpy as np
from hspiral import HSPIRAL
import atexit
javabridge.start_vm(class_path=bf.JARS)
#atexit.register(javabridge.kill_vm)

path='/home/morgan/movies/LifeR_RokG_WT_3.czi'

def get_movie_shape(path):
    xml_image = bf.get_omexml_metadata(path=path)
    metadata = bf.OMEXML(xml_image)
    path = path

    NX = metadata.image().Pixels.SizeX
    NY = metadata.image().Pixels.SizeY
    NZ = metadata.image().Pixels.SizeZ
    NC = metadata.image().Pixels.SizeC
    NT = metadata.image().Pixels.SizeT
    return NX,NY,NZ,NC,NT


def movie_input_generator(path,channel=0,output_type=np.uint16):

    NX,NY,NZ,NC,NT=get_movie_shape(path)

    assert (channel < NC)

    with bf.ImageReader(path=path) as reader:
        for t in range(NT):
            image = np.zeros((NX, NY, NZ), dtype=output_type)

            for z in range(NZ):
                image[:, :, z] = reader.read(c=channel, t=t, z=z, rescale=False)

            yield image
    return

import pylab

NX, NY, NZ, NC, NT = get_movie_shape(path)

for t,frame in enumerate(movie_input_generator(path,channel=1)):
    for z in range(NZ):
        bf.write_image('./out.ome.tif',pixels=frame[:,:,z],pixel_type=bf.PT_UINT16,c=0,t=t,z=z,size_c=1,size_t=NT,size_z=NZ)

    #pylab.imshow(frame[:,:,2])
    #pylab.pause(1)

javabridge.kill_vm()