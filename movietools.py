
import bioformats as bf
import numpy as np





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


def read_psf(path,output_type=np.float32):
    NX,NY,NZ,NC,NT=get_movie_shape(path)
    #assert(NC==1)
    #assert(NT==1)
    psf=np.zeros((NX,NY,NZ),dtype=output_type)
    with bf.ImageReader(path=path) as reader:
        for z in range(NZ):
            psf[:, :, z] = reader.read(c=z, rescale=False)
    return psf