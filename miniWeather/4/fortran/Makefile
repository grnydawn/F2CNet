
PNETCDF_ROOT := /Users/8yk/opt/pnetcdf/1.13.0

FFLAGS  := -O3 -ffree-line-length-none -I${PNETCDF_ROOT}/include
LDFLAGS := -L${PNETCDF_ROOT}/lib -lpnetcdf

PARAMS  := -D_NX=100 -D_NZ=50 -D_SIM_TIME=100 -D_OUT_FREQ=200 \
		   -D_DATA_SPEC=DATA_SPEC_THERMAL -DNO_INFORM

do: banner mpi.exe output.nc

banner:
	@echo "######## BEGIN FORTRAN ########"

mpi.exe: miniWeather_mpi.F90
	mpif90 -o mpi.exe ${FFLAGS} ${LDFLAGS} ${PARAMS} $<

output.nc: mpi.exe
	mpirun -n 2 ./$<

clean:
	rm -f mpi.exe output.nc *.o *.mod
