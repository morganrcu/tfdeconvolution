
import argparse
import javabridge
import bioformats as bf
import atexit

javabridge.start_vm(class_path=bf.JARS)
atexit.register(javabridge.kill_vm)

from flowdec import data as fd_data
from flowdec import psf as fd_psf
from flowdec import restoration as fd_restoration


from movietools import get_movie_shape,movie_input_generator,read_psf

parser= argparse.ArgumentParser(description='Deconvolve some movies')

parser.add_argument('input',type=str)
parser.add_argument('psf',type=str)
parser.add_argument('output_prefix',type=str)
parser.add_argument('channel',type=int,default=0)

args=parser.parse_args()
print(args)

inputpath=args.input
psfpath=args.psf
outputprefix=args.output_prefix
channel=args.channel


NX, NY, NZ, NC, NT = get_movie_shape(inputpath)
psf=read_psf(psfpath)

for t,frame in enumerate(movie_input_generator(inputpath,channel=channel)):
    print('Processing frame {0}'.format(t))
    algo = fd_restoration.RichardsonLucyDeconvolver(3).initialize()
    res = algo.run(fd_data.Acquisition(data=frame, kernel=psf), niter=30).data
    bf.write_image(outputprefix.format(t),res,z=0,t=0,c=0,size_z=NZ,size_t=1,size_c=1)
javabridge.kill_vm()
